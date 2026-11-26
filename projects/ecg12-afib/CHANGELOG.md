# Changelog

Notable changes to the ECG/AFib 12-lead project. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- Training and the validation gate run on serverless AI Runtime environment 6; serving and the DAILY
  job run on serverless environment 6.
- Models are logged with explicit serving requirements (exact versions) that avoid what blocks
  Model Serving's [express deployment](https://docs.databricks.com/aws/en/machine-learning/model-serving/express-deployments);
  on AWS and Azure the endpoints used it in our tests.
- The app's `/api/generate` (a new synthetic reading) has a rate limit like `/api/analyze` and
  `/api/warmup`: 30 requests per user and 120 from all users together in any 60 s, then a 429 with
  `Retry-After` and a message, which the Live tab shows where a reading failed.
- `deploy.sh` runs `bundle plan` before deploying, and takes its values from the bundle (flags,
  `BUNDLE_VAR_<name>`, `variable-overrides.json` or defaults), so its checks match what is deployed.
  The same check runs first on the values given as flags (warehouse id included), so a bad flag
  stops the script before it builds the app.
- `deploy.sh` stops at step 1 when the profile's `workspace_id` line names another workspace than
  its `host` connects to, instead of only failing `bundle summary` after SETUP.
- `deploy.sh` stops with `<flag> needs a value` when a flag that takes a value has none, instead
  of taking the next flag as the value or stopping with no message.
- `deploy.sh` and the docs need Databricks CLI 0.286.0 or later, the first with `bundle validate
  --strict`; `deploy.sh` stops at step 1 on an older one.
- SETUP's serving step skips an endpoint that already serves the champion version, and reports
  whether each endpoint used express deployment (endpoint tag `deployment`).
- The DAILY job's scoring notebooks (`07_`, `08_`) install the CPU build of torch 2.11.0 from
  PyTorch's CPU-only index (PyPI's Linux build is the CUDA one, with about 20 nvidia / CUDA
  packages and triton the CPU tasks never use). A workspace whose serverless compute cannot reach
  `download.pytorch.org` needs that line's `--index-url=...` removed.

### Security
- MLflow 3.18.0 instead of 3.13.0, for which the OSV vulnerability database lists security
  advisories (none for 3.18.0, checked 2026-11-25): in the served models' requirements (MLflow
  3.13.0 requires `cryptography` below 49, 3.18.0 below 51), and as `mlflow-skinny` 3.18.0 in
  every notebook that logs, loads or serves the models. The training and gate notebooks (`02_` to
  `04_`) now install it over the 3.13.0 that AI Runtime environment 6 ships, and restart Python;
  `05_`, `07_` and `08_` install it over environment 6's 3.12.0. It changes no other package either
  environment ships (on environment 6 it adds `python-multipart`, which that environment lacks).
- The repository's docs publish build installs soupsieve 4.0 and tornado 6.6.1 instead of 2.10 and
  6.6, for which the OSV vulnerability database lists security advisories (none for the new
  versions, checked 2026-11-25). `.github/requirements-publish.in` (and `requirements-ci.in`) now
  give the command that generates the hash-pinned file.

### Added
- [Quickstart](docs/quickstart.md) for macOS and Windows, and [express deployment](docs/express-deployment.md) notes.
- `variable-overrides.example.json` template for bundle variables.
- The app's title links to this project's source.
- The repository's NOTICE says that the serving endpoints log every request and response to the
  `<model>_payload` inference tables in the deployer's catalog and schema, so a project adapted to
  real data should treat those tables as holding that data.

### Fixed
- Markdown that rendered wrong on GitHub (strikethrough from `~`, hidden `ARRAY<FLOAT>`).
- The inference-table delay in the app, the notebooks, the dashboard and the docs: the docs now say
  "rows are typically available within seconds" for CPU custom-model endpoints such as these (the
  "within 1 hour" best-effort delivery is for Foundation Model API, external-model and agent
  endpoints); the text quotes that and what we saw in the tested deployments (20 to 60 minutes; in
  the tested AWS deployment the `_payload` tables were plain Delta tables, not the telemetry view
  the docs describe). A request over 1 MiB is "logged as `null`", not left out.
- SETUP's serving step no longer waits out its 50-minute timeout when an endpoint update that
  reuses the config version fails before the wait sees it start. When its pending config differs
  from the one left before the update (for example by its start time, Model Serving's own
  timestamp), the step stops at once with Model Serving's message; when it looks exactly like an
  earlier failure, the step stops once that has lasted 5 minutes (at its next check of the
  endpoint, which another endpoint's AI Gateway setup can delay) saying it could not tell and
  pointing at the endpoint's events, instead of reporting a failure. A newly created endpoint whose first check comes before
  Model Serving shows the update in progress is waited for, instead of failing at once.
- The serving step also counts this update's served model recovering, or ready with the traffic
  not switched yet, as progress; stops at once saying another update is in progress (or replaced
  this one) when the endpoint's pending config definitely serves something else (a name, model or
  version that is there and differs), instead of reporting that update's result as this one's,
  and counts a poll with no endpoint state as unclear, not as progress (the endpoint showing the
  update in progress still counts as progress, whatever its served entity shows). Every field the
  wait compares (the pending config's signature, the served entity's identity, the at-target
  check) goes through one three-way comparison: a field the API left out is missing, never a
  value, so missing data alone never makes a failure, another writer's update or "did not
  complete": the wait counts it as unclear until a poll shows it whole. A value present before and
  left out now does not make a new attempt, and a config version the step could not read before the
  update is unknown, not "no endpoint yet" (the old config seen again is not another update's).
  Before changing an existing endpoint, the step checks, fail safe, that its current config is
  readable and serves only this deployment's model and that a pending config names only that model
  too, both up front and on the read the update is sent from (which narrows, not closes, the window
  before the update: one writer at a time is still assumed); anything unknown stops the step saying
  it could not tell which model it serves, not that another deployment owns it (before, names were
  pooled across the two configs, so a pending config naming this model passed for an unknown
  current one); and the deployment-path report of an endpoint whose served entity is left out is
  `unknown`, a warning, not a crash. It waits for both endpoints together under one
  deadline, so a failure of the second one shows at once instead of after the first one is
  ready. The notebook states that it assumes it is the only writer to its endpoints while it runs.
- When an endpoint update completed but the endpoint's config, serving this run's model version,
  has another size, scale-to-zero or traffic share than the one submitted (changed in the Serving
  UI, say), the serving step stops saying what differs and that it may have been changed in the
  Serving UI, instead of saying the update did not complete and the endpoint keeps serving its
  previous config, which may not be true. When that config leaves out its served entity's name,
  model or version, it is unclear, not a changed config: the step waits for a poll that shows it
  whole, or stops after 5 minutes saying it could not tell.
- The serving step's deployment-path report counts only a served model's latest environment
  preparation or image build, so an image built by an earlier, failed attempt no longer reports a
  later express deployment as `standard`; when the events cannot say which build came last, it
  reports `unknown`. When no event carries a config version, the timestamps order the builds (the
  latest start and what follows it), so an older finished image build followed by a newer finished
  express preparation is `express`, not `unknown`; a missing timestamp, or two kinds of build
  starting at the same time, still makes it `unknown`.
- A `deployment` or `removeAfter` endpoint tag that cannot be set (no permission, a tag policy
  refusing the value) is a warning with the tag to add by hand, not a failed SETUP: the endpoints
  serve either way.
- The deployment-path report names a path only from a finished build: a container image build
  that started and has not finished, an express preparation that has not completed, or a build log
  that says express is still preparing with no build in the events is `unknown` (before: `standard`,
  `express`, `express`), and so is a path whose events hold a build that names no served entity.
  A completion counts only at or after the latest start of its kind, so an older completed express
  preparation (or image build) followed by a newer start that has not finished is `unknown`, not
  `express` (or `standard`); events with a completion and no start still name the path.
- The dashboard's Ask Genie said "Genie is unavailable due to incompatible SQL expressions in the
  dashboard" after a first deployment (the dashboard was published before DAILY created its
  tables). `deploy.sh --run-daily` now publishes it again right after DAILY (step 6, before the
  app's steps, so their failure cannot skip it), with the bundle's warehouse and its
  `embed_credentials` value in an explicit JSON body (`--json`), so `false` is always sent (the
  API's default for a missing value is `true`); the docs give the command for the other paths
  ([Ask Genie](docs/dashboard.md)).

### Tested
End to end on AWS and on Azure: SETUP and DAILY jobs, both serving endpoints (express deployment),
the AI/BI dashboard and the Databricks App. Not yet tested on Google Cloud. On Windows, only WSL
was tested. Some changes above have so far been checked only by the local tests, with stand-ins for
the CLI and Model Serving, not in a workspace run: in the serving step, recovering or ready served
models as progress, another writer's update, a missing state, one deadline for both endpoints,
the finished-build rule of the deployment-path report (and its ordering of builds by start time and
of a completion after the latest start), the message for a completed config with other settings and
the tag warnings; in the app, the `/api/generate` rate limit; in `deploy.sh`, the new step order and
the checks for a flag with no value and for the CLI version. The profile `workspace_id` check ran
read-only against a wrong and a correct profile, and the JSON publish body was sent by CLI 0.280.0,
0.286.0 and 1.20.0 to a local stand-in server. The local tests run the models' torch 2.11.0 and
cloudpickle 3.1.1 (before: 2.9.0 and 3.0.0), so their pickle-and-load check runs those versions.

### Upgrading
1. Run SETUP before DAILY: DAILY loads the models with the new package versions.
2. SETUP moves each endpoint to the new `champion` version; "Until the new configuration is ready,
   the old configuration keeps serving prediction traffic"
   ([Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints)).
3. Every SETUP run sets the endpoint tag `deployment` again, overwriting the previous value (a
   tag call that fails only warns).
4. To roll back, set each model's `champion` alias back to its previous version (both models: the
   gate promotes both or neither), then run `workflow/05_serve_endpoints` on its own with the same
   parameters; it serves whatever `champion` points to. DAILY scores `champion` too. Running the whole
   SETUP job instead would retrain and re-gate. A rollback changes only later scoring (live and
   DAILY): it does not rewrite tables already written or the inference tables' history.

## 2026-11-19 — hardening ([#5](https://github.com/databricks-industry-solutions/biosignals/pull/5))
Schema ownership checks, not-for-clinical-use notice, CI hardening, rate limits, dependency updates.

## 2026-11-04 — first release
The ecg12-afib project: SETUP and DAILY jobs, two serving endpoints, AI/BI dashboard, Databricks App.
