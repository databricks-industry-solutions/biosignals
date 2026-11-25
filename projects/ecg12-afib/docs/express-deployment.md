# Express deployment for the serving endpoints

[← ecg12-afib README](../README.md)

The two serving endpoints, `<endpoint_prefix>_arr` and `<endpoint_prefix>_afib`, started through Model
Serving's **express deployment** path in our tests on AWS and Azure. This page explains what that is,
why this project's models are eligible for it, where it happens, how to check it, and how the clouds differ.

Tested end to end on AWS and on Azure: both jobs (SETUP and DAILY), the two serving endpoints (through
express deployment), the AI/BI dashboard, and the Databricks App (Triage → Classify, with the dashboard
embedded). Not yet tested on Google Cloud, where express deployment works differently (see
[Differences between clouds](#differences-between-clouds)).

> **Reference:** [Express deployments for model serving endpoints](https://docs.databricks.com/aws/en/machine-learning/model-serving/express-deployments) (AWS) · [the same page for Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/express-deployments) · [for Google Cloud](https://docs.databricks.com/gcp/en/machine-learning/model-serving/express-deployments)

## What express deployment is

When Model Serving starts a custom model, it needs a Python environment with the model's packages.

- **Standard path:** Model Serving builds a container image for the model, running `pip install` on its
  requirements (including, for a PyTorch model, the CUDA libraries that come with `torch`). In this project
  that took about 15 minutes, and an endpoint took about 21 minutes to be ready.
- **Express path:** Model Serving prepares the environment once and reuses it. In the [AWS](https://docs.databricks.com/aws/en/machine-learning/model-serving/express-deployments) and
  [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/express-deployments) docs' words: "Databricks automatically prepares and reuses compatible model environments when you
  create or update an endpoint. No additional registration parameter or deployment-mode selection is
  required." In this project's test runs on AWS the preparation took 2 to 4 minutes, and an endpoint
  was ready 7 to 10 minutes after the update started; on Azure the preparation took 1 to 3 minutes,
  and a new endpoint was ready 5 to 7 minutes after it was created.

Express deployment is for **custom MLflow models** on CPU or GPU endpoints, such as this project's PyTorch
models. It is not the same as vLLM, which is a server for large language models; this project doesn't use
vLLM.

## Why this project's models are eligible

Express deployment has no switch. On AWS and Azure, Databricks applies it on its own, for "eligible models
on CPU or GPU serving compute" ([AWS](https://docs.databricks.com/aws/en/machine-learning/model-serving/express-deployments), [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/express-deployments)). The docs give two conditions: "The model must be logged with MLflow from
Databricks compute, either classic compute running Databricks Runtime or serverless compute", and
"Databricks also checks whether the model's recorded runtime, Python version, and dependencies are
compatible with the selected serving compute." They then list what can prevent it: missing or incompatible
runtime and Python-version metadata, a Conda-only environment or a dependency on PySpark, editable / Git /
URL / package-index entries (or bundled wheels without a content hash) in the pip requirements,
CUDA-tagged package versions, missing model-size metadata or models and environment files over the size
limits, and a customer-managed key for managed services.

The docs require exact version pins in one case only: "For models logged on serverless compute and served
on GPU, declared `torch`, `torchvision`, and `torchaudio` requirements must be exact version pins matching
the GPU serving environment." This project's endpoints serve on CPU, so that rule does not apply to them.

This project's explicit package list and training setup avoid the blockers that depend on the
model: URL / Git / editable / package-index entries, CUDA-tagged versions, a Conda-only environment,
PySpark, and missing runtime or Python-version metadata. The others are outside the project's control:
a workspace setting such as a customer-managed key for managed services, and the size checks, which
Model Serving makes on the logged model (this project's models are small).

| Where | What it does |
|---|---|
| `common/config.py` (`MODEL_PIP_REQUIREMENTS`) | One short list of exact package versions, matching the training environment, with no CUDA tags, URLs or extra packages (exact versions are not an express requirement on CPU; they make the pickled model load with the versions it was saved with) |
| `workflow/02_train_arr_model.py`, `workflow/03_train_afib_model.py` | Log each model with `pip_requirements=config.MODEL_PIP_REQUIREMENTS`, so MLflow records that list instead of guessing one from the training notebook |
| `resources/ecg12_afib_setup.job.yml` | Trains on a named serverless AI Runtime environment, so the model records its runtime and Python version |
| `workflow/05_serve_endpoints.py` | Creates or updates the endpoints as usual; on AWS and Azure, Model Serving uses the express path on its own when the models are eligible (they were in our tests on both clouds) |

MLflow also offers an explicit route, `mlflow.register_model(..., env_pack="databricks_model_serving")`,
which packs the training notebook's whole Python environment at registration. This project doesn't use
it: the training environment is a GPU environment (PyTorch with CUDA libraries), while the endpoints serve
on CPU, so it would ship gigabytes the endpoints don't need.
The automatic route above prepares a small environment from `MODEL_PIP_REQUIREMENTS` instead.

Without `pip_requirements`, MLflow infers the list from the training notebook. On GPU compute that list
included `nvidia-ml-py` and other packages the model doesn't need. In an earlier test run with that
inferred list and an older MLflow, the endpoints took the standard path; the next runs, with the explicit
list and a newer MLflow (the version every notebook that logs, loads or serves the models runs, see
`common/config.py`), took the express path. Both changed at once, so we have not tested which of the two
was needed; the explicit list stays because it avoids the blockers above and keeps the served
environment small.

## Where it happens

No notebook prepares the environment:

1. `02_` / `03_` record the model's package list when they log the model (the recipe).
2. `05_` creates or updates the endpoint (the trigger).
3. When the model is eligible, Model Serving reads the model's `requirements.txt`, prepares the
   environment and starts the endpoint (the event "Model artifact preparation").

Both models use the same package list, so on the express path Model Serving prepares **one** environment
and both endpoints reuse it ("shared by models with identical requirements"). Each endpoint still loads its own model files.

The DAILY job (`07_`, `08_`) does not use the endpoints: it loads the models directly in the job. Express
deployment does not apply there.

## How to check it

Because express deployment is automatic for eligible models, nothing in this project's configuration
records which path an endpoint took. Model Serving's own records do: the served model's build log and the endpoint's events. The
AWS and Azure pages describe no check; the [Google Cloud page](https://docs.databricks.com/gcp/en/machine-learning/model-serving/express-deployments) says: "The deployment event log for an express
model contains no container build events." The build-log line and the event names below are what Model
Serving wrote in our tests on AWS and Azure.

**In the workspace:** Serving > the endpoint > **Logs** > **Build logs**, with the served model version
selected. On the express path the log is short and starts with:

```text
This environment was prepared ahead of time by express deployment and is shared by models with identical requirements.
```

followed by a few lines of timings. On the standard path it is a long container build log with
`pip install` output instead.

**From the CLI:**

```bash
# Build log of a served model (served model name = <endpoint>-<model version>)
databricks serving-endpoints build-logs <endpoint> <endpoint>-<model-version> -p <profile>

# Endpoint events: express shows "Model artifact preparation", standard shows "Container image creation"
databricks api get /api/2.0/serving-endpoints/<endpoint>/events -p <profile>
```

**In `05_`'s output:** after the endpoints are ready, `05_` reads each endpoint's build log and events and
prints one line per endpoint, for example:

```text
bio_ecg12_arr-5: EXPRESS deployment (environment prepared in 2m47s; no container image build)
```

It also tags each endpoint `deployment` = `express` (its environment preparation completed, or was prepared
earlier and reused), `standard` (a container image build finished) or `unknown` (the build log or events could
not be read, a build had started and not finished, or they named neither path; this does not mean express
wasn't used). The tag is set by this project's `05_`, not by the platform; it shows on the Serving page next to
the endpoint's other tags.
Neither `standard` nor `unknown` fails the task; `05_` prints a warning instead.

The endpoint page itself does not show a separate express label in our tests; the build log above is the
platform's own record.

## What to expect

- **First deployment of a new package list:** extra time while Model Serving prepares the environment
  (1 to 4 minutes in our tests). Later deployments with the same list reuse it.
- **Rerunning SETUP without a new model version:** if an endpoint already serves the champion version,
  with no update in progress and no pending config, `05_` leaves it as it is (it prints
  `<endpoint> already serves <catalog>.<schema>.<model> v<n>; no update`). A new champion version, a
  previous update that failed, or a pending config left behind triggers an update.
- **AI Gateway settings:** the first time `05_` turns on inference tables and usage tracking, the endpoint
  showed one more redeploy right after it was ready (inferred from the timing); that redeploy reused the
  prepared environment. Later runs, with the settings already on, had no extra redeploy.
- **Scale-to-zero:** an idle endpoint stops; the first request afterwards waits while it starts again
  (about 5 minutes in our test). Requests sent during that time can time out; retry them.

## Differences between clouds

The express deployment page is not the same on every cloud (all three versions checked 2026-11-23):

- **[AWS](https://docs.databricks.com/aws/en/machine-learning/model-serving/express-deployments)** and **[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/express-deployments)**: express deployment is automatic for eligible models, as described above, on CPU
  or GPU serving compute. `env_pack` "remains supported" there as an explicit alternative.
- **[Google Cloud](https://docs.databricks.com/gcp/en/machine-learning/model-serving/express-deployments)**: the page still describes the explicit route only. Express deployment is turned on by
  passing `env_pack` to `mlflow.register_model`; the model "must be registered in Unity Catalog and served
  on CPU compute", and the page adds "Register from a regular serverless notebook." This project does not
  use `env_pack`, so on Google Cloud expect the standard path (a container image build): the endpoints
  should still serve, with slower updates, and `05_` would report `standard` and print a warning. This
  has not been tested yet.

## If it falls back to the standard path

Check the model's recorded environment (in the model version's artifacts: `requirements.txt`, `MLmodel`,
`python_env.yaml`) against the list of blockers above. Common causes: the model was logged without
`pip_requirements`, a requirement with a CUDA tag (`torch==2.11.0+cu130`) or an index URL, or a changed
training environment whose versions no longer match `MODEL_PIP_REQUIREMENTS`.

## Changing the training environment

`MODEL_PIP_REQUIREMENTS` must match the training environment's versions (the model is saved with
`cloudpickle` there and loaded elsewhere). When moving to another serverless environment version:

1. Read that environment's release notes and package list.
2. Update `MODEL_PIP_REQUIREMENTS` to its numpy, pandas, cloudpickle, torch and mlflow versions (torch
   without the CUDA tag; mlflow at the environment's mlflow-skinny version).
3. Don't install a package version **below** what an environment ships: on serverless, downgrading a core
   package such as `databricks-sdk` or `protobuf` can stop the notebook's Python process from restarting.
   The tests in `dev/tests` check this.
4. Run SETUP, then DAILY, and check the build log as above.

> **Reference:** [Serverless environment 6](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/six) · [AI Runtime environment 6](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/six-gpu)
