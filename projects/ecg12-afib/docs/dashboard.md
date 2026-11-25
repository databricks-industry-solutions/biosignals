# Dashboard

[← ecg12-afib README](../README.md)

`resources/ecg12_afib.dashboard.yml` deploys `dashboards/ecg12_afib.lvdash.json` with both targets.
Its queries name bare tables; the bundle points them at your catalog and schema
(`dataset_catalog` / `dataset_schema`) and runs them on your `warehouse_id`. Deploying publishes
it, with `embed_credentials` from the bundle variable of that name.

**Whose credentials the dashboard uses.** With `embed_credentials: true` (the default, for this
synthetic demo only) the published dashboard runs every viewer's queries with the **deploying
identity's** credentials: viewers (and the app's embed) need only `CAN_READ` on the dashboard,
which the bundle grants to `viewer_group`, and they see whatever the deployer can see. That is
acceptable here only because every row is synthetic. **If you adapt the project to real data, set
`embed_credentials` to `false`** (`--var embed_credentials=false`, or `--embed-credentials false`
with `deploy.sh`) and grant the viewers what their own queries need: `USE CATALOG` on the catalog,
`USE SCHEMA` and `SELECT` on the schema (or on the dashboard's tables), and `CAN_USE` on the SQL
warehouse. Each viewer then sees only what their own Unity Catalog grants allow; consider
narrowing `viewer_group` from `users` as well.

- **Batch monitoring** reads `arr_triage_history_12l` and `dailyOrweekly_arr_afib_inference_12l`:
  the funnel (landed → flagged → AFib-positive), arrivals per hour, triage by rhythm, the
  AFib-probability distribution, AFib by age band and the worklist, all for the **latest
  `batch_date`** (one coherent funnel), plus the daily trend over every date.
- **Live serving** reads the two payload tables: requests, model execution time and payloads not
  logged per endpoint, the returned probabilities, the latest requests. It parses only the small
  response JSON; `execution_duration_ms` is the model's inference time, not end-to-end latency.
  The app's warm-up pings (`client_request_id = 'app-warmup'`) are left out of all of these and
  counted on their own ("App warm-up pings").

**Until there is data, tiles show errors, not empty charts.** The batch tiles fail until the DAILY
job has run once (their tables do not exist before). The live tiles fail with `UNRESOLVED_COLUMN`
until AI Gateway has delivered the first inference-table logs: AI Gateway creates each payload
table with a placeholder schema (`databricks_request_id` only) when `05_` enables it, and adds the
real columns with the first delivered batch of logs.

The dashboard's live tiles fill in after the endpoints have had requests: Databricks writes the
request logs to the inference tables with a delay. For CPU custom-model endpoints such as these the
docs say "rows are typically available within seconds", through a view over a telemetry table
([inference tables for serving endpoints](https://docs.databricks.com/aws/en/ai-gateway/inference-tables-serving-endpoints));
in the tested deployments rows arrived 20 to 60 minutes after a request (in the tested AWS
deployment the tables were plain Delta tables, not that view; checked 2026-11-24). Refresh the
dashboard after that; until the first logs arrive the live tiles show an error, while the batch
tiles work as soon as DAILY has run. Nothing needs to be rebuilt or redeployed.

**Ask Genie needs a publish after the first DAILY run.** A published dashboard includes a Genie
Agent, which answers questions in plain language (**Ask Genie**), and the docs say: "When you
republish your dashboard with changes, the associated Genie Agent updates automatically to reflect
the new datasets and visualizations." ([Genie Agents with dashboards](https://docs.databricks.com/aws/en/dashboards/genie-spaces)). `bundle deploy`
publishes the dashboard before SETUP and DAILY have created its tables. What we saw in our tests on
AWS and Azure:

- Published before the DAILY job had created its tables, the dashboard's Ask Genie said "Genie is
  unavailable due to incompatible SQL expressions in the dashboard". The dashboard's SQL uses none
  of the expressions that page lists as incompatible.
- Publishing the unchanged draft again, after DAILY had run, fixed it; `bundle plan` still showed the
  dashboard unchanged afterwards.
- Published after DAILY had run, with the request-log tables still holding only their placeholder
  column, Ask Genie worked (it opened with suggested questions).

So publish the dashboard once more after the DAILY job has run once. `./deploy.sh --run-daily` does
it right after DAILY (its step 6). Without `--run-daily`, or after a plain bundle deploy (target `no_app`), run
DAILY, then:

```bash
databricks bundle summary -t <target> -p <profile> --var catalog=<catalog> --var warehouse_id=<warehouse-id> -o json | jq -r '.resources.dashboards.ecg12_afib_dashboard.id'
databricks lakeview publish <dashboard-id> --json '{"embed_credentials": true, "warehouse_id": "<warehouse-id>"}' -p <profile>
```

`<target>` is `no_app` or `full`; the first command prints `<dashboard-id>` (without `jq`, run it
without `-o json | jq ...`: the id is the part of the dashboard's URL after `/dashboardsv3/`). Write
`"embed_credentials": false` if you deploy with `embed_credentials` set to `false`, and keep the
field either way: the API's default for a missing `embed_credentials` is `true`
([publish API](https://docs.databricks.com/api/workspace/lakeview/publish)), and CLI 0.280.0 sent no
`embed_credentials` at all for `--embed-credentials=false`. (PowerShell: the JSON file pattern of
the [quickstart](quickstart.md), step 5.) In the workspace you
can instead open the dashboard and "Click Publish in the upper-right corner of the dashboard"
([Share a dashboard](https://docs.databricks.com/aws/en/dashboards/share/share)), keeping the same
credentials setting.

**Publish as the identity that deployed.** With embedded credentials, viewers' queries run as
whoever published last: `embed_credentials` is the "Flag to indicate if the publisher's
credentials should be embedded in the published dashboard", and "These embedded credentials will be
used to execute the published dashboard's queries" ([publish API](https://docs.databricks.com/api/workspace/lakeview/publish)). The same goes for
Publish in the workspace. So publish with the profile that deployed the bundle (`deploy.sh` uses its
`-p` profile), or check which identity a profile uses first (`databricks current-user me -p
<profile>`).

To change the dashboard, edit `dashboards/build_ecg_demo_dashboard.py`, run
`python dashboards/build_ecg_demo_dashboard.py` and commit the regenerated JSON (`--check` reports
whether the committed file is current). A dashboard edited in the workspace UI differs from the
file, and the next `bundle deploy` of that deployment stops with "This dashboard has been
modified remotely since the last bundle deployment." unless run with `--force`, which replaces the UI
edits with the file; `databricks bundle generate dashboard --resource ecg12_afib_dashboard --force` pulls the
UI version back into the file.

The app reads the dashboard id from `ECG_DEMO_REACT_DASHBOARD_ID`, which the `full` target sets to
the deployed dashboard's id (`${resources.dashboards.ecg12_afib_dashboard.id}`, resolved when the
app is deployed), so pointing the app at the dashboard needs no rebuild. Without an id the app's
Batch tab shows a "Dashboard not configured" note. The header's "Lakeflow" link works the same way:
`ECG_DEMO_REACT_DAILY_JOB_ID` = `${resources.jobs.ecg12_afib_daily.id}` opens the DAILY job, and
without it the link opens the workspace's jobs list. Embedding also needs the app's domain on the
workspace's dashboard-embedding allowlist (a workspace admin setting; `deploy.sh` adds it).
