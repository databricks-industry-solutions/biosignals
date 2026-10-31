# Dashboard

[← ecg12-afib README](../README.md)

## Dashboard

`resources/ecg12_afib.dashboard.yml` deploys `dashboards/ecg12_afib.lvdash.json` with both targets.
Its queries name bare tables; the bundle points them at your catalog and schema
(`dataset_catalog` / `dataset_schema`) and runs them on your `warehouse_id`. Deploying publishes
it, with `embed_credentials: true`: the published dashboard runs its queries with the deploying
identity's credentials, so viewers (and the app's embed) need only `CAN_READ` on the dashboard,
which the bundle grants to `viewer_group`. The data is synthetic; set `embed_credentials: false` to
make each viewer's own Unity Catalog grants apply instead.

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
real columns with the first delivered batch of logs, best effort, typically within about an hour of
the first app request. Nothing needs to be rebuilt or redeployed; the tiles work once the data is
there.

To change the dashboard, edit `dashboards/build_ecg_demo_dashboard.py`, run
`python dashboards/build_ecg_demo_dashboard.py` and commit the regenerated JSON (`--check` reports
whether the committed file is current). A dashboard edited in the workspace UI differs from the
file, and the next `bundle deploy` stops with "has been modified remotely" unless run with
`--force`; `databricks bundle generate dashboard --resource ecg12_afib_dashboard --force` pulls the
UI version back into the file.

The app reads the dashboard id from `ECG_DEMO_REACT_DASHBOARD_ID`, which the `full` target sets to
the deployed dashboard's id (`${resources.dashboards.ecg12_afib_dashboard.id}`, resolved when the
app is deployed), so pointing the app at the dashboard needs no rebuild. Without an id the app's
Batch tab shows a "Dashboard not configured" note. The header's "Lakeflow" link works the same way:
`ECG_DEMO_REACT_DAILY_JOB_ID` = `${resources.jobs.ecg12_afib_daily.id}` opens the DAILY job, and
without it the link opens the workspace's jobs list. Embedding also needs the app's domain on the
workspace's dashboard-embedding allowlist (a workspace admin setting; `deploy.sh` adds it).
