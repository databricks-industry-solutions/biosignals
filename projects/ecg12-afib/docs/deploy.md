# Configure, deploy, cost and teardown

[← ecg12-afib README](../README.md)

## Configuration

Bundle variables (`databricks.yml`). Pass them as `--var name=value` on every bundle command (or
export `BUNDLE_VAR_<name>`); `bundle validate` refuses to run without the required ones.

| Variable | Required | Default | Used for |
|---|---|---|---|
| `catalog` | **yes** | — | Unity Catalog catalog of every table, model and inference table (jobs: `catalog_name`; dashboard: `dataset_catalog`; app: `ECG_DEMO_REACT_CATALOG`) |
| `warehouse_id` | **yes** | — | SQL warehouse the dashboard runs on |
| `schema` | no | `biosignals` | Unity Catalog schema (jobs: `schema_name`; dashboard: `dataset_schema`; app: `ECG_DEMO_REACT_UC_SCHEMA`); `01_` creates it if needed |
| `endpoint_prefix` | no | `bio_ecg12` | Prefix of the two serving endpoints, `<endpoint_prefix>_arr` and `<endpoint_prefix>_afib` (letters, digits, `_`, `-`; SETUP job: `endpoint_prefix`; app: `ECG_DEMO_REACT_ENDPOINT_PREFIX`). Endpoint names are workspace-wide: use a value no other deployment in your workspace uses |
| `app_name` | no | `ecg12-demo` | Name of the Databricks App (`full` target). Lowercase letters, digits and hyphens, at most 30 characters. Override it if that name is already taken in your workspace (app names are workspace-wide) |
| `viewer_group` | no | `users` | Group granted `CAN_READ` on the dashboard and `CAN_USE` on the app |

The workspace itself comes from your Databricks CLI profile (`-p <profile>`); no host or profile
is committed. The model names (`bio_ecg12_arr`, `bio_ecg12_afib`) and their inference tables
(`bio_ecg12_arr_payload`, `bio_ecg12_afib_payload`) are fixed in `common/config.py`: they live in
your catalog and schema, so they are this deployment's own. The serving endpoints are the one
workspace-wide name, so they come from `endpoint_prefix`. **`05_` never takes over another
deployment's endpoint:** before updating an existing endpoint of the same name it checks what the
endpoint serves, and if that is anything but this deployment's `<catalog>.<schema>.<model>` the
SETUP job stops with "Serving endpoint … already exists and serves …; set the bundle variable
endpoint_prefix to a unique value", having changed neither endpoint.

## Deploy

Two targets, which are **separate deployments** (each has its own folder and state under
`/Workspace/Users/<you>/.bundle/biosignals-ecg12-afib/<target>/`). Pick one per workspace; to
switch, `bundle destroy` the first. Both use `mode: development` (job and dashboard names get a
`[dev <user>]` prefix, schedules deploy paused; the app name is not prefixed).

**What the deploying identity needs:** `USE CATALOG` and `CREATE SCHEMA` on the catalog (or, if the
schema exists, `USE SCHEMA`, `CREATE TABLE` and `CREATE MODEL` on it); serverless jobs, including
serverless GPU; permission to create Model Serving endpoints; `CAN_USE` on the warehouse; for
`full`, permission to create apps. Adding the embedding allowlist entry needs a workspace admin.

### Path 1: jobs + dashboard from the workspace (target `no_app`, no local build)

1. Clone this repository into the workspace as a Git folder.
2. Open `projects/ecg12-afib/databricks.yml` in the bundle editor and set the required variables
   for the `no_app` target, for example in your copy:
   ```yaml
   targets:
     no_app:
       variables:
         catalog: <catalog>
         warehouse_id: <warehouse-id>
   ```
3. Deploy (target `no_app`, the default). Then run `ecg12_afib_setup`, then `ecg12_afib_daily`
   from the Deployments tab.

The same from a laptop:

```bash
cd projects/ecg12-afib
databricks bundle validate --strict -p <profile> --var catalog=<catalog> --var warehouse_id=<id>
databricks bundle deploy -p <profile> --var catalog=<catalog> --var warehouse_id=<id>
databricks bundle run ecg12_afib_setup -p <profile> --var catalog=<catalog> --var warehouse_id=<id>
databricks bundle run ecg12_afib_daily -p <profile> --var catalog=<catalog> --var warehouse_id=<id>
```

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

1. checks the prerequisites, the required arguments and that the profile authenticates;
2. builds the app (`app/build_from_lockfiles.sh`, see [`app/README.md`](../app/README.md));
3. `databricks bundle validate --strict -t full`;
4. `databricks bundle deploy -t full`: the jobs, the dashboard and the app, with `viewer_group`
   granted `CAN_READ` on the dashboard and `CAN_USE` on the app;
5. runs the SETUP job and waits for it (`--run-daily` then also runs the DAILY job once);
6. grants the app's service principal `CAN_QUERY` on both endpoints (they exist only after the
   SETUP job, so they cannot be app resources of the first deploy);
7. starts the app (`databricks bundle run ecg12_afib_app`), which deploys the built source with
   its environment, including the dashboard and DAILY job ids;
8. adds the app's domain to the dashboard-embedding allowlist (needs a workspace admin; fails with
   the domain to add, or `--skip-embed-allowlist` skips it and prints that domain).

Other options: `--schema`, `--endpoint-prefix`, `--app-name`, `--viewer-group` (the bundle
variables above). Re-running
is safe: every step updates in place, and the SETUP job retrains and re-gates both models.

The notebooks also run one at a time from a Git folder (they find `common/` one folder up); type
the catalog into each notebook's `catalog_name` widget.

## Cost

These are billable resources in your workspace: serverless job runs (SETUP trains and gates on
serverless GPU, A10; DAILY runs on standard serverless), two Model Serving endpoints (workload size
Small with scale-to-zero, so they cost only while they serve, and the first request after idling
waits for them to start), the SQL warehouse while the dashboard runs queries, and the app while it
is running (`databricks apps stop <app_name> -p <profile>` stops it). While someone has the app's
Live tab open, its warm-up pings keep both endpoints awake (one request each every 10 minutes);
they scale to zero again once no page is open. The DAILY schedule is paused.

## Teardown

```bash
cd projects/ecg12-afib
databricks bundle destroy -t <no_app|full> -p <profile> --var catalog=<catalog> --var warehouse_id=<id>
```

This removes what the bundle created: the jobs, the dashboard and (`full`) the app. It does not
remove what the jobs created; delete those yourself. Delete only this deployment's endpoints, by
the names your `endpoint_prefix` gives (`<endpoint_prefix>_arr`, `<endpoint_prefix>_afib`), and
first check that each one serves your own `<catalog>.<schema>` models (endpoint names are
workspace-wide, so another deployment's endpoint can have a similar name):

```bash
P=<profile>; PREFIX=<endpoint_prefix>; UC=<catalog>.<schema>
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

then drop the tables (`training_data_12l`, `batch_data_12l_*`, `arr_triaged_12l_*`,
`arr_triage_history_12l`, `dailyOrweekly_arr_afib_inference_12l`, `bio_ecg12_*_payload`) or the
whole schema if it holds nothing else (`DROP SCHEMA <catalog>.<schema> CASCADE`), and remove the
app's domain from the embedding allowlist.
