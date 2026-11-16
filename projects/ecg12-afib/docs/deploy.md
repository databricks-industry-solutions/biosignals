# Configure, deploy, cost and teardown

[← ecg12-afib README](../README.md)

## Configuration

Bundle variables (`databricks.yml`). Pass them as `--var name=value` on every bundle command (or
export `BUNDLE_VAR_<name>`); `bundle validate` refuses to run without the required ones.

| Variable | Required | Default | Used for |
|---|---|---|---|
| `catalog` | **yes** | — | Unity Catalog catalog of every table, model and inference table (jobs: `catalog_name`; dashboard: `dataset_catalog`; app: `ECG_DEMO_REACT_CATALOG`). Letters, digits and underscores only |
| `warehouse_id` | **yes** | — | SQL warehouse the dashboard runs on |
| `schema` | no | `biosignals` | Unity Catalog schema (jobs: `schema_name`; dashboard: `dataset_schema`; app: `ECG_DEMO_REACT_UC_SCHEMA`); `01_` creates it if needed. Letters, digits and underscores only. **One schema per deployment** (below) |
| `endpoint_prefix` | no | `bio_ecg12` | Prefix of the two serving endpoints, `<endpoint_prefix>_arr` and `<endpoint_prefix>_afib` (letters, digits, `_`, `-`; SETUP job: `endpoint_prefix`; app: `ECG_DEMO_REACT_ENDPOINT_PREFIX`). Endpoint names are workspace-wide: use a value no other deployment in your workspace uses |
| `mlflow_experiment` | no | `/Users/<deployer>/biosignals-ecg12-afib` | MLflow experiment (absolute workspace path) of every SETUP-job run: training (`02_`, `03_`) and the validation gate (`04_`) (SETUP job: `mlflow_experiment`). Outlives `bundle destroy`; [Where the MLflow runs go](#where-the-mlflow-runs-go) |
| `app_name` | no | `ecg12-demo` | Name of the Databricks App (`full` target). Lowercase letters, digits and hyphens, at most 30 characters. Override it if that name is already taken in your workspace (app names are workspace-wide) |
| `viewer_group` | no | `users` | Group granted `CAN_READ` on the dashboard and `CAN_USE` on the app |
| `embed_credentials` | no | `true` | `true`: the published dashboard runs every viewer's queries with the deploying identity's credentials (fine for this synthetic data only). **Set `false` if you adapt the project to real data**, and grant the viewers access ([Dashboard](dashboard.md)) |
| `remove_after` | no | empty | Expiry date `YYYY-MM-DD`, set on both jobs as the tag `removeAfter`; [Expiry tag](#expiry-tag-removeafter) |

The workspace itself comes from your Databricks CLI profile (`-p <profile>`); no host or profile
is committed. Catalog and schema names go into SQL and model names unquoted, so the jobs and
`deploy.sh` accept only letters, digits and underscores (Unity Catalog allows more, such as
hyphens, but only inside backticks).

### Expiry tag (`removeAfter`)

Some workspaces run cleanup automation that deletes resources once the date in their
`removeAfter` tag (`YYYY-MM-DD`) has passed. To have it remove this demo's jobs, pass the date:
`--var remove_after=<YYYY-MM-DD>` (or `BUNDLE_VAR_remove_after`, or `deploy.sh --remove-after
<YYYY-MM-DD>`). The bundle sets it as `presets.tags`, so every bundle resource that takes tags
gets it: here the two jobs (next to the `dev` tag that `mode: development` adds). The tag is
opt-in: the default is empty, so the jobs carry `removeAfter` with an empty value, which names no
date and schedules nothing for removal. A workspace without such automation can ignore the variable.

To change the date later, redeploy with the new one: `databricks bundle deploy --var
remove_after=<YYYY-MM-DD>` (plus your usual `-t`, `-p` and `--var` values) updates the jobs' tags
in place, without re-running the SETUP job (`deploy.sh` would run it again). Like every variable,
pass it on each deploy: a deploy without it puts the empty value back.

The tag does not reach the two serving endpoints (the SETUP job's `05_` creates them through the
SDK, not the bundle), the app and the dashboard (bundle apps and dashboards take no tags), or the
MLflow experiment, schema, tables and models. Remove those as [Teardown](#teardown) describes.

### One schema per deployment

The model names (`bio_ecg12_arr`, `bio_ecg12_afib`), their inference tables
(`bio_ecg12_arr_payload`, `bio_ecg12_afib_payload`) and every other table name are fixed in
`common/config.py`. They are a deployment's own only if its schema is: **give every deployment
its own schema** (`--var schema=<schema>`). Two deployments in one `<catalog>.<schema>` would share
these objects: one SETUP run would overwrite the other's training data and move its `@champion`,
the other's DAILY job would then score with those models, and either teardown would delete models
the other still serves.

The jobs stop that from happening by accident. The SETUP job's first task (`01_`) records this
deployment as the schema's owner in the one-row table `ecg12_afib_deployment`, on its first run.
The first task of SETUP and of DAILY (`06_`) then checks that table before writing anything and
stops when it names another deployment ("… belongs to another deployment of this demo … Set the
bundle variable `schema` to a schema no other deployment uses"); DAILY also stops when no
deployment owns the schema yet (run SETUP first). A deployment's identity is the `deployment_id`
job parameter, `<workspace id>:<bundle root path>`; the root path is the bundle's folder in the
workspace (`/Workspace/Users/<you>/.bundle/biosignals-ecg12-afib/<target>` in development mode).
So the two targets, two users, or one user in two workspaces that share a Unity Catalog metastore
are different deployments and need different schemas.

**Upgrading a deployment made before this check existed:** its schema has no
`ecg12_afib_deployment` table yet, so after you deploy the new version its next DAILY run stops,
by design, with "… has no ecg12_afib_deployment table, so no deployment owns it yet: run this
deployment's SETUP job first". Run the SETUP job once (or re-run `deploy.sh`, whose step 5 runs
it): its first task records this deployment as the owner, and DAILY runs again from then on. If two
deployments already share the schema, the first one to run SETUP becomes its owner; move the other
one to a schema of its own.

The record is a table, not a schema property or tag: setting a schema property
(`ALTER SCHEMA … SET DBPROPERTIES`) needs ownership of the schema or `MANAGE` on it, more than the
`CREATE TABLE` the jobs already need; a workspace tag policy can restrict tag keys and values; and
the table is only ever created, never replaced, so a second deployment cannot overwrite it.

The serving endpoints are the one workspace-wide name, so they come from `endpoint_prefix`.
**`05_` never takes over another deployment's endpoint:** before updating an existing endpoint of
the same name it checks what the endpoint serves, and if that is anything but this deployment's
`<catalog>.<schema>.<model>` the SETUP job stops with "Serving endpoint … already exists and
serves …; set the bundle variable endpoint_prefix to a unique value", having changed neither
endpoint.

### Where the MLflow runs go

Every MLflow run of the SETUP job (one per training notebook, `02_` and `03_`, and one per gate run,
`04_`) is logged to **one experiment**, the `mlflow_experiment` variable. Its default,
`/Users/${workspace.current_user.userName}/biosignals-ecg12-afib`, resolves at deploy to the home
folder of whoever deploys (a user's email, or a service principal's application id), so it needs
no edit in any workspace. Before their first run, the notebooks make it the active experiment
with `common/tracking.py`, which creates what is missing: first the parent folder (the workspace
`mkdirs` call, which also creates intermediate folders), then the experiment. The folder step is
what makes the default work for an identity without a home folder: `mlflow.set_experiment` creates a missing
experiment but not a missing folder ("Parent directory ... does not exist"), and a service
principal can have no home folder yet. The helper also creates the experiment itself, because
`02_` and `03_` start in parallel: if both create it at once, MLflow's `set_experiment` hands the
losing task the experiment without making it active, and that task's run would land in its
notebook's own experiment. The registered models still go to Unity Catalog
(`mlflow.set_registry_uri("databricks-uc")`): the experiment holds the runs, Unity Catalog the
model versions, and each version links back to the run that logged it.

The experiment is deliberately **not** a bundle resource: `bundle destroy` leaves it, so run
history accrues across redeploys and a version in Unity Catalog never points at a deleted run.
(Without the variable, each notebook would log to its own experiment inside the bundle's folder,
which `bundle destroy` deletes.) The DAILY job starts no runs, so it does not take the variable.

One experiment per deployer, by default: your `no_app` and `full` targets, and deployments to
different schemas, share it. Their runs stay distinguishable: each training run's `model_name` tag
is the full `<catalog>.<schema>.<model>` name, and each gate run's `<model>.holdout_table`
parameter names the `<catalog>.<schema>` it read. To keep a deployment's runs separate, or to
share one experiment with your team, point the variable elsewhere:
`--var mlflow_experiment=/Shared/<folder>/<name>`, `deploy.sh --mlflow-experiment <path>`, or
`"mlflow_experiment"` in `variable-overrides.json` (Path 1); the path must be inside a folder,
which the SETUP job creates if it is missing and the job's identity may create folders there.

**Why the home folder is the default, not `/Shared`.** The home folder belongs to the identity
that deploys and runs the jobs, so it normally can create it (unless the workspace's folder
permissions forbid it, see below) and nobody else's runs land there; it
is also where the bundle itself deploys (`/Workspace/Users/<you>/.bundle/...`). A `/Shared/...`
default would put every deployer's runs in one experiment, and whether an identity may create
folders in `/Shared` is up to the workspace's admins. The SETUP job's identity (the deployer, in
development mode) needs permission to create the folder, or `CAN_EDIT` on an experiment someone
else created; without it the SETUP job stops at that step, before any training.

## Deploy

Two targets, which are **separate deployments** (each has its own folder and state under
`/Workspace/Users/<you>/.bundle/biosignals-ecg12-afib/<target>/`), so each needs its own schema.
Pick one per workspace; to switch in the same schema, remove the first completely
([Teardown](#teardown), including `ecg12_afib_deployment`), or give the second its own `schema`.
Both use `mode: development` (job and dashboard names get a `[dev <user>]` prefix, schedules
deploy paused; the app name is not prefixed).

**What the deploying identity needs:** `USE CATALOG` and `CREATE SCHEMA` on the catalog (or, if the
schema exists, `USE SCHEMA`, `CREATE TABLE` and `CREATE MODEL` on it); serverless jobs, including
serverless GPU; permission to create Model Serving endpoints; `CAN_USE` on the warehouse; for
`full`, permission to create apps. Adding the embedding allowlist entry needs a workspace admin.

### Path 1: jobs + dashboard from the workspace (target `no_app`, no local build)

1. Clone this repository into the workspace as a Git folder.
2. Set the required variables without editing the committed `databricks.yml`: in the Git folder,
   create the file `projects/ecg12-afib/.databricks/bundle/no_app/variable-overrides.json` with
   ```json
   {"catalog": "<catalog>", "warehouse_id": "<warehouse-id>"}
   ```
   (add `"schema": "<schema>"` for a schema of this deployment's own, and any other variable from
   the table above). The bundle reads this file for the `no_app` target, after `--var` and
   `BUNDLE_VAR_<name>` and before the defaults ([bundle variables](https://docs.databricks.com/aws/en/dev-tools/bundles/variables)).
   `.databricks/` is in the repository's `.gitignore`, so the file never ends up in a commit. If
   the bundle editor still reports a required variable without a value, set the values under
   `targets.no_app.variables` in your Git folder's copy of `databricks.yml` instead, and do not
   commit that change.
3. Open `projects/ecg12-afib/databricks.yml` in the bundle editor and deploy (target `no_app`, the
   default). Then run `ecg12_afib_setup`, then `ecg12_afib_daily` from the Deployments tab.

The same from a laptop:

```bash
cd projects/ecg12-afib
databricks bundle validate --strict -p <profile> --var catalog=<catalog> --var warehouse_id=<id>
databricks bundle deploy -p <profile> --var catalog=<catalog> --var warehouse_id=<id>
databricks bundle run ecg12_afib_setup -p <profile> --var catalog=<catalog> --var warehouse_id=<id>
databricks bundle run ecg12_afib_daily -p <profile> --var catalog=<catalog> --var warehouse_id=<id>
```

(add `--var schema=<schema>` to every command for a schema other than `biosignals`).

No app is deployed on this path; the dashboard and the serving endpoints still are.

### Path 2: everything, including the app (target `full`, `deploy.sh`)

The app's wheel and compiled UI are not committed: they are built locally from the lockfiles.
**Prerequisites:** [apx](https://github.com/databricks-solutions/apx) **0.3.8** exactly (it brings
its own bun, which builds the UI), [uv](https://docs.astral.sh/uv/), the Databricks CLI 0.281.0 or
later, `jq`, `python3`, and a CLI profile for your workspace (`databricks auth login -p <profile>`).

```bash
cd projects/ecg12-afib
./deploy.sh -p <profile> --catalog <catalog> --warehouse-id <id> [--run-daily]
```

`deploy.sh` stops with an error at the first step that fails:

1. checks the prerequisites, the required arguments (catalog and schema names included), that
   the profile authenticates and, unless `--skip-embed-allowlist`, that it can read the
   dashboard-embedding settings step 8 changes (a workspace admin setting), so a caller who cannot
   learns it before anything is built or deployed;
2. builds the app (`app/build_from_lockfiles.sh`, see [`app/README.md`](../app/README.md));
3. `databricks bundle validate --strict -t full`;
4. `databricks bundle deploy -t full`: the jobs, the dashboard and the app, with `viewer_group`
   granted `CAN_READ` on the dashboard and `CAN_USE` on the app;
5. runs the SETUP job and waits for it (`--run-daily` then also runs the DAILY job once);
6. grants the app's service principal `CAN_QUERY` on both endpoints (they exist only after the
   SETUP job, so they cannot be app resources of the first deploy);
7. starts the app (`databricks bundle run ecg12_afib_app`), which deploys the built source with
   its environment, including the dashboard and DAILY job ids;
8. adds the app's domain to the dashboard-embedding allowlist (needs a workspace admin;
   `--skip-embed-allowlist` skips it and prints the domain an admin must add). It reads the list
   with its etag (the setting's version) and writes back that list plus the domain with the same
   etag, so if another admin changes the list in between the write is refused instead of dropping
   their entry; the script then reads again and retries, and finally checks the domain is there.
   **Caveat:** a setting that has never been written has no etag, so two first-ever writes at the
   same moment can still overwrite each other (the later one wins); if the dashboard does not show
   in the app, an admin adds the domain by hand (Settings > Security > Embed dashboards).

Other options: `--schema`, `--endpoint-prefix`, `--mlflow-experiment`, `--app-name`, `--viewer-group`,
`--embed-credentials`, `--remove-after` (the bundle variables above). Re-running is safe: every
step updates in place, and the SETUP job retrains and re-gates both models.

The notebooks also run one at a time from a Git folder (they find `common/` one folder up); type
the catalog into each notebook's `catalog_name` widget, and into `02_`, `03_` and `04_`'s
`mlflow_experiment` widget the experiment path (e.g. `/Users/<you>/biosignals-ecg12-afib`).

## Cost

These are billable resources in your workspace: serverless job runs (SETUP trains and gates on
serverless GPU, A10; DAILY runs on standard serverless), two Model Serving endpoints (workload size
Small with scale-to-zero, so they cost only while they serve, and the first request after idling
waits for them to start), the SQL warehouse while the dashboard runs queries, and the app while it
is running (`databricks apps stop <app_name> -p <profile>` stops it). While someone has the app's
Live tab open, its warm-up pings keep both endpoints awake (one request each every 10 minutes);
they scale to zero again once no page is open. The DAILY schedule is paused.

## Teardown

Delete only what belongs to this deployment. The model and table names are the same in every
deployment, so first check that the schema is yours: the jobs record its owner in
`<catalog>.<schema>.ecg12_afib_deployment` ([One schema per deployment](#one-schema-per-deployment)).

**1. Check the schema's owner** (reads only; needs `CAN_USE` on the warehouse and `SELECT` on the
table). It compares the recorded owner with this deployment's identity,
`<workspace id>:<bundle root path>`, and prints `OK` or `STOP`:

```bash
cd projects/ecg12-afib
P=<profile>; T=<no_app|full>; CATALOG=<catalog>; SCHEMA=<schema>; WH=<warehouse-id>
VARS=(-t "$T" -p "$P" --var "catalog=$CATALOG" --var "schema=$SCHEMA" --var "warehouse_id=$WH")
UC="$CATALOG.$SCHEMA"
MINE="$(databricks metastores current -p "$P" -o json | jq -r .workspace_id):$(databricks bundle validate "${VARS[@]}" -o json | jq -r .workspace.root_path)"
OWNER=$(databricks api post /api/2.0/sql/statements -p "$P" --json "$(jq -n --arg wh "$WH" --arg t "$UC.ecg12_afib_deployment" \
  '{warehouse_id: $wh, wait_timeout: "50s", statement: "SELECT DISTINCT deployment_id FROM \($t)"}')" \
  | jq -r '[.result.data_array[]?[0]] | join(",")')
if [ "$OWNER" = "$MINE" ]; then echo "OK: $UC belongs to this deployment ($MINE)"
else echo "STOP: $UC belongs to '${OWNER:-no deployment}', not to this deployment ($MINE): delete nothing in it"; fi
```

On `STOP`, delete nothing in that schema (steps 3 and 4): it holds another deployment's models and
tables, or none of this demo's. Step 2 is still safe: it removes only this bundle's own resources.

**2. Remove what the bundle created:** the jobs, the dashboard and (`full`) the app. The MLflow
experiment is not one of them and stays, with its run history
([Where the MLflow runs go](#where-the-mlflow-runs-go)).

```bash
databricks bundle destroy "${VARS[@]}"
```

**3. Only after `OK`: delete this deployment's endpoints and models.** Delete the endpoints by the
names your `endpoint_prefix` gives (`<endpoint_prefix>_arr`, `<endpoint_prefix>_afib`), and only if
each one serves your own `<catalog>.<schema>` models (endpoint names are workspace-wide, so another
deployment's endpoint can have a similar name):

```bash
PREFIX=<endpoint_prefix>
# Every model the endpoint serves or is about to serve (current + pending config, incl. the
# legacy served_models field): delete it only if that is exactly this deployment's model.
SERVED='[(.config, .pending_config) | select(. != null)
         | ((.served_entities // [])[].entity_name, (.served_models // [])[].model_name)]
        | map(select(. != null)) | unique | join(",")'
for pair in arr:bio_ecg12_arr afib:bio_ecg12_afib; do
  e="${PREFIX}_${pair%%:*}"; want="$UC.${pair#*:}"
  got=$(databricks serving-endpoints get "$e" -p "$P" -o json | jq -r "$SERVED")
  if [ "$got" = "$want" ]; then databricks serving-endpoints delete "$e" -p "$P"
  else echo "Not deleting $e: it serves '$got', not '$want'"; fi
done
databricks registered-models delete "$UC.bio_ecg12_arr" -p "$P"
databricks registered-models delete "$UC.bio_ecg12_afib" -p "$P"
```

**4. Only after `OK`: drop the tables,** `ecg12_afib_deployment` last (`training_data_12l`,
`batch_data_12l_*`, `arr_triaged_12l_*`, `arr_triage_history_12l`,
`dailyOrweekly_arr_afib_inference_12l`, `bio_ecg12_*_payload`, then `ecg12_afib_deployment`), or
the whole schema if it holds nothing else (`DROP SCHEMA <catalog>.<schema> CASCADE`). Once
`ecg12_afib_deployment` is gone, the next SETUP run in that schema records a new owner. Finally,
remove the app's domain from the embedding allowlist.

**5. Optional: delete the MLflow experiment,** only if no other deployment of yours logs to it (by
default your targets and schemas share one) and you no longer need the run history:

```bash
EXP=$(databricks bundle validate "${VARS[@]}" -o json | jq -r .variables.mlflow_experiment.value)
databricks experiments delete-experiment "$(databricks experiments get-by-name "$EXP" -p "$P" -o json | jq -r .experiment.experiment_id)" -p "$P"
```
