"""Generate the 12-lead ECG/AFib AI/BI dashboard, ``ecg12_afib.lvdash.json``: batch monitoring +
live serving.

The bundle deploys that file as a dashboard resource (``resources/ecg12_afib.dashboard.yml``),
with the catalog and schema set by the bundle variables (``dataset_catalog`` /
``dataset_schema``): every query names bare tables, so the file is the same for every
workspace. Re-run this script after changing it and commit the JSON; ``--check`` only reports
whether the committed file is current (dev/tests/test_repo_consistency.py runs it).

    python dashboards/build_ecg_demo_dashboard.py           # rewrites dashboards/ecg12_afib.lvdash.json
    python dashboards/build_ecg_demo_dashboard.py --check   # exit 1 if it would change

The demo scores ECGs on two separate paths, and the dashboard shows them in two separate
sections, each with its own source tables:

* **Batch monitoring** (DAILY job): 07_arr_triage / 08_afib_classify score with the champion models
  loaded from Unity Catalog, on the job's driver; they never call the endpoints. Source tables:
  ``arr_triage_history_12l`` (every ECG of every day, with its triage call) and
  ``dailyOrweekly_arr_afib_inference_12l`` (the flagged ECGs of every day, with the AFib call).
  Tiles: the day's funnel (landed -> flagged -> AFib-positive), arrivals per hour, triage by true
  rhythm, the AFib-probability distribution, AFib-positive share by age band and the
  flagged-patient worklist, all for the LATEST batch date in ``arr_triage_history_12l`` (so they
  form one coherent funnel), plus the daily trend over every date.
* **Live serving** (the app): every Triage -> Classify click queries the endpoints, which log each
  request and response to their AI Gateway inference tables ``bio_ecg12_arr_payload`` and
  ``bio_ecg12_afib_payload``. Tiles: requests, model execution time and payloads not logged
  (KPIs), requests and execution time per hour by endpoint, the distribution of the returned
  probabilities, and the latest requests. Only the small response JSON is parsed; the ~650 KB
  request JSON (the ECG itself) is only measured (its size), never parsed.

Until data exists the tiles show errors, not empty charts: the batch tiles until the DAILY job
has run once (the tables do not exist before), the live tiles until AI Gateway has delivered the
first inference-table logs. AI Gateway creates each inference table with a placeholder schema
(``databricks_request_id`` only) when 05_serve_endpoints enables it and adds the real columns
with the first delivered batch of logs, best effort, typically within about an hour of the first
app request; until then the live queries fail with UNRESOLVED_COLUMN.

Table, model and endpoint names come from ../common/config.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # projects/ecg12-afib -> common/
from common import config  # noqa: E402

OUT = Path(__file__).resolve().parent / "ecg12_afib.lvdash.json"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--check", action="store_true", help=f"exit 1 if {OUT.name} is not current; write nothing")
args = parser.parse_args()

# Bare table names: the bundle's dataset_catalog / dataset_schema supply catalog and schema.
TRI = config.TRIAGE_HISTORY_TABLE                                    # arr_triage_history_12l
INF = config.INFERENCE_TABLE                                         # dailyOrweekly_arr_afib_inference_12l
ARR_PAY = config.payload_table_name(config.ARR_ENDPOINT)             # bio_ecg12_arr_payload
AFIB_PAY = config.payload_table_name(config.AFIB_ENDPOINT)           # bio_ecg12_afib_payload

DAY = config.BATCH_DATE_COL
LATEST_DAY = f"{DAY} = (SELECT max({DAY}) FROM {TRI})"               # single-day tiles: the latest batch
AFIB_POS = f"afib_predictions_1 > {config.AFIB_THRESHOLD}"
RHYTHM = "CASE rhythm_label WHEN 0 THEN 'NSR' WHEN 1 THEN 'AFib' WHEN 2 THEN 'Other' END"
AGE_BAND = ("CASE WHEN age_years < 50 THEN '40-49' WHEN age_years < 60 THEN '50-59' "
            "WHEN age_years < 70 THEN '60-69' WHEN age_years < 80 THEN '70-79' ELSE '80-90' END")
ARR_LABEL = f"{config.ARR_ENDPOINT} (triage)"
AFIB_LABEL = f"{config.AFIB_ENDPOINT} (classify)"
# Response JSON of each endpoint: ARR {"predictions": [{"arr_prob": p, "arr_flag": f}, ...]},
# AFib {"predictions": [[p_not_afib, p_afib], ...]}; one entry per ECG.
ARR_RESPONSE = "'predictions ARRAY<STRUCT<arr_prob: DOUBLE, arr_flag: DOUBLE>>'"
AFIB_RESPONSE = "'predictions ARRAY<ARRAY<DOUBLE>>'"

# Categorical colors: rhythm classes (same as the notebooks' plots) and endpoints.
RHYTHM_COLORS = [{"value": "NSR", "color": "#2a78d6"}, {"value": "AFib", "color": "#eb6834"},
                 {"value": "Other", "color": "#1baf7a"}]
ENDPOINT_COLORS = [{"value": ARR_LABEL, "color": "#6250d6"}, {"value": AFIB_LABEL, "color": "#eda100"}]


def payload_rows(table, label):
    """One row per logged endpoint request (request JSON measured, not parsed)."""
    return (f"SELECT '{label}' AS endpoint, databricks_request_id, request_time, "
            "date_trunc('HOUR', request_time) AS request_hour, status_code, execution_duration_ms, "
            "ROUND(length(request) / 1024.0, 0) AS request_kb, "
            "CASE WHEN request IS NULL OR coalesce(size(logging_error_codes), 0) > 0 THEN 1 ELSE 0 END AS not_logged "
            f"FROM {table} ")


datasets = [
    # --- Batch monitoring (DAILY job: models loaded from UC, endpoints not called) ---------------
    {"name": "funnel", "displayName": "Batch: triage funnel (latest day)", "queryLines": [
        f"SELECT (SELECT COUNT(*) FROM {TRI} WHERE {LATEST_DAY}) AS total_signals, ",
        f"(SELECT COUNT(*) FROM {TRI} WHERE {LATEST_DAY} AND arr_flag = 1) AS flagged_abnormal, ",
        f"(SELECT COUNT(*) FROM {INF} WHERE {LATEST_DAY} AND {AFIB_POS}) AS afib_positive"]},
    # Each ECG's acquisition time within the day (06_generate_batch_data spreads them over clinic hours).
    {"name": "arrivals", "displayName": "Batch: ECGs landed per hour (latest day)", "queryLines": [
        f"SELECT hour(acquisition_datetime_utc) AS hour_utc, {RHYTHM} AS rhythm, COUNT(*) AS n ",
        f"FROM {TRI} WHERE {LATEST_DAY} ",
        "GROUP BY hour(acquisition_datetime_utc), rhythm_label ORDER BY hour_utc, rhythm_label"]},
    {"name": "triage_accuracy", "displayName": "Batch: triage by rhythm (latest day)", "queryLines": [
        f"SELECT {RHYTHM} AS rhythm, ",
        "CASE WHEN arr_flag = 1 THEN 'flagged (->afib)' ELSE 'not flagged' END AS triage, ",
        f"COUNT(*) AS n FROM {TRI} WHERE {LATEST_DAY} GROUP BY rhythm_label, arr_flag ORDER BY rhythm_label"]},
    {"name": "prob_hist", "displayName": "Batch: AFib probability (latest day)", "queryLines": [
        f"SELECT ROUND(afib_predictions_1, 1) AS prob_bin, COUNT(*) AS n FROM {INF} ",
        f"WHERE {LATEST_DAY} GROUP BY ROUND(afib_predictions_1, 1) ORDER BY prob_bin"]},
    {"name": "trend", "displayName": "Batch: daily trend (every day)", "queryLines": [
        f"SELECT {DAY} AS date, ",
        "COUNT(*) AS afib_classified, ",
        f"SUM(CASE WHEN {AFIB_POS} THEN 1 ELSE 0 END) AS afib_positive FROM {INF} ",
        "GROUP BY 1 ORDER BY 1"]},
    # Patient context: every screened patient of the day (TRI) by age band, with the AFib model's
    # call where they were flagged and classified (INF).
    {"name": "afib_by_age", "displayName": "Batch: AFib-positive by age band (latest day)", "queryLines": [
        "SELECT t.age_band, COUNT(*) AS screened, ",
        f"SUM(CASE WHEN i.{AFIB_POS} THEN 1 ELSE 0 END) AS afib_positive, ",
        f"ROUND(SUM(CASE WHEN i.{AFIB_POS} THEN 1 ELSE 0 END) / COUNT(*), 3) AS afib_positive_share ",
        f"FROM (SELECT {config.ID_COL}, {AGE_BAND} AS age_band FROM {TRI} WHERE {LATEST_DAY}) t ",
        f"LEFT JOIN (SELECT {config.ID_COL}, afib_predictions_1 FROM {INF} WHERE {LATEST_DAY}) i ",
        f"USING ({config.ID_COL}) GROUP BY t.age_band ORDER BY t.age_band"]},
    {"name": "worklist", "displayName": "Batch: flagged-patient worklist (latest day)", "queryLines": [
        f"SELECT {DAY} AS batch_date, mrn, age_years AS age, sex, bmi, heart_rate_bpm AS hr_bpm, ",
        "CONCAT(bp_systolic_mmhg, '/', bp_diastolic_mmhg) AS bp_mmhg, ",
        f"{RHYTHM} AS true_rhythm, ROUND(afib_predictions_1, 3) AS p_afib, ",
        f"CASE WHEN {AFIB_POS} THEN 'AFib' ELSE 'not AFib' END AS afib_call, ",
        "date_format(acquisition_datetime_utc, 'HH:mm') AS acquired_utc ",
        f"FROM {INF} WHERE {LATEST_DAY} ORDER BY afib_predictions_1 DESC LIMIT 25"]},
    # --- Live serving (the app: endpoint calls logged by AI Gateway) ------------------------------
    {"name": "live_requests", "displayName": "Live: endpoint requests (AI Gateway inference tables)", "queryLines": [
        payload_rows(ARR_PAY, ARR_LABEL), "UNION ALL ", payload_rows(AFIB_PAY, AFIB_LABEL),
        "ORDER BY request_time DESC"]},
    # One row per ECG scored live, from the logged response JSON.
    {"name": "live_predictions", "displayName": "Live: returned probabilities", "queryLines": [
        f"SELECT '{ARR_LABEL}' AS endpoint, request_time, p.arr_prob AS score, ROUND(p.arr_prob, 1) AS score_bin ",
        f"FROM (SELECT request_time, explode(from_json(response, {ARR_RESPONSE}).predictions) AS p ",
        f"FROM {ARR_PAY} WHERE status_code = 200 AND response IS NOT NULL) ",
        "UNION ALL ",
        f"SELECT '{AFIB_LABEL}' AS endpoint, request_time, p[1] AS score, ROUND(p[1], 1) AS score_bin ",
        f"FROM (SELECT request_time, explode(from_json(response, {AFIB_RESPONSE}).predictions) AS p ",
        f"FROM {AFIB_PAY} WHERE status_code = 200 AND response IS NOT NULL)"]},
]


def field(name, expression=None):
    return {"name": name, "expression": expression or f"`{name}`"}


def query(ds, fields, disaggregated=True, filters=None):
    q = {"datasetName": ds, "fields": fields, "disaggregated": disaggregated}
    if filters:
        q["filters"] = [{"expression": f} for f in filters]
    return [{"name": "main_query", "query": q}]


def place(widget, x, y, w, h):
    return {"widget": widget, "position": {"x": x, "y": y, "width": w, "height": h}}


def text(name, lines, x, y, w, h):
    return place({"name": name, "multilineTextboxSpec": {"lines": lines}}, x, y, w, h)


def counter(name, ds, value, title, x, y, *, agg=None, period=None, fmt=None, w=4, h=4):
    """KPI. ``agg`` = (field name, expression) aggregating a multi-row dataset; else one-row ``value``."""
    fields = [field(*agg)] if agg else [field(value)]
    encodings = {"value": {"fieldName": fields[0]["name"], "displayName": title}}
    if fmt:
        encodings["value"]["format"] = fmt
    if period:
        fields.append(field(*period))
        encodings["period"] = {"fieldName": period[0]}
    return place({"name": name, "queries": query(ds, fields, disaggregated=agg is None),
                  "spec": {"version": 2, "widgetType": "counter", "encodings": encodings,
                           "frame": {"showTitle": True, "title": title}}}, x, y, w, h)


def chart(name, kind, ds, x_enc, y_enc, title, pos, *, fields, color=None, disaggregated=True,
          filters=None, description=None, mark=None):
    encodings = {"x": x_enc, "y": y_enc}
    if color:
        encodings["color"] = color
    frame = {"showTitle": True, "title": title}
    if description:
        frame.update(showDescription=True, description=description)
    spec = {"version": 3, "widgetType": kind, "encodings": encodings, "frame": frame}
    if mark:
        spec["mark"] = mark
    return place({"name": name, "queries": query(ds, fields, disaggregated, filters), "spec": spec}, *pos)


def table(name, ds, columns, title, pos):
    """Table widget; ``columns`` = [(field, display name), ...]."""
    return place({"name": name, "queries": query(ds, [field(f) for f, _ in columns]),
                  "spec": {"version": 2, "widgetType": "table",
                           "encodings": {"columns": [{"fieldName": f, "displayName": d} for f, d in columns]},
                           "frame": {"showTitle": True, "title": title}}}, *pos)


def cat(name, display, mappings=None):
    scale = {"type": "categorical"}
    if mappings:
        scale["mappings"] = mappings
    return {"fieldName": name, "displayName": display, "scale": scale}


def quant(name, display, fmt=None):
    enc = {"fieldName": name, "displayName": display, "scale": {"type": "quantitative"}}
    if fmt:
        enc["format"] = fmt
    return enc


PERCENT = {"type": "number-percent", "decimalPlaces": {"type": "max", "places": 1}}

layout = [
    text("title", ["## ECG/AFib on Databricks — 12-lead triage → classify\n", "\n",
                   "Synthetic 12-lead ECGs with synthetic patient records land in Unity Catalog; an *arr* (triage) "
                   "model flags abnormal rhythms; an *AFib* model classifies the flagged. Two scoring paths, two "
                   "sections below. The single-day batch tiles show the **latest batch date**."], 0, 0, 12, 2),

    # --- Batch monitoring ------------------------------------------------------------------------
    text("hdr-batch", ["### Batch monitoring — DAILY job (models loaded from Unity Catalog @"
                       f"{config.CHAMPION_ALIAS}, scored in the job; endpoints not called)\n", "\n",
                       f"Sources: `{TRI}` (every ECG of every day) and `{INF.lower()}` (flagged ECGs of every "
                       f"day). Single-day tiles: the latest `{DAY}` in `{TRI}`. These tiles show an error until "
                       "the DAILY job has run once."], 0, 2, 12, 2),
    counter("kpi-total", "funnel", "total_signals", "ECGs landed (latest day)", 0, 4),
    counter("kpi-flagged", "funnel", "flagged_abnormal", "Flagged abnormal by triage (latest day)", 4, 4),
    counter("kpi-afib", "funnel", "afib_positive", "AFib-positive by classifier (latest day)", 8, 4),
    chart("arrivals-bar", "bar", "arrivals", cat("hour_utc", "Hour acquired (UTC)"), quant("n", "ECGs"),
          "ECGs landed per hour, by true rhythm (the day's incoming stream)", (0, 8, 6, 5),
          fields=[field("hour_utc"), field("n"), field("rhythm")],
          color=cat("rhythm", "True rhythm", RHYTHM_COLORS)),
    # x = triage OUTCOME (flagged / not flagged), bars stacked by true rhythm.
    chart("triage-bar", "bar", "triage_accuracy", cat("triage", "Triage outcome"), quant("n", "ECGs"),
          "Triage by true rhythm (arr model)", (6, 8, 6, 5),
          fields=[field("rhythm"), field("n"), field("triage")],
          color=cat("rhythm", "True rhythm", RHYTHM_COLORS)),
    chart("prob-bar", "bar", "prob_hist", quant("prob_bin", "AFib probability (bin)"), quant("n", "Flagged ECGs"),
          "AFib-probability distribution (flagged ECGs, latest day)", (0, 13, 6, 5),
          fields=[field("prob_bin"), field("n")]),
    chart("afib-age-bar", "bar", "afib_by_age", cat("age_band", "Age band (years)"),
          quant("afib_positive_share", "AFib-positive (share of screened)", PERCENT),
          "AFib-positive share by age band (latest day)", (6, 13, 6, 5),
          fields=[field("age_band"), field("afib_positive_share")]),
    chart("trend-bar", "bar", "trend", {"fieldName": "date", "displayName": "Batch date", "scale": {"type": "categorical"}},
          {"scale": {"type": "quantitative"}, "fields": [
              {"fieldName": "afib_classified", "displayName": "Classified (flagged)"},
              {"fieldName": "afib_positive", "displayName": "AFib-positive"}]},
          "Daily batch inference (every day)", (0, 18, 12, 5),
          fields=[field("date"), field("afib_classified"), field("afib_positive")], mark={"layout": "group"},
          color={"scale": {"type": "categorical", "mappings": [
              {"value": "Classified (flagged)", "color": "#2a78d6"},
              {"value": "AFib-positive", "color": "#eb6834"}]}}),
    table("worklist-table", "worklist", [
        ("batch_date", "Batch date"), ("mrn", "MRN (synthetic)"), ("age", "Age"), ("sex", "Sex"), ("bmi", "BMI"),
        ("hr_bpm", "HR (bpm)"), ("bp_mmhg", "BP (mmHg)"), ("true_rhythm", "True rhythm"),
        ("p_afib", "p(AFib)"), ("afib_call", "AFib call"), ("acquired_utc", "Acquired (UTC)")],
        "Flagged-patient worklist (latest day, top 25 by AFib probability)", (0, 23, 12, 6)),
    text("patient-note", ["**Who is being flagged.** Every ECG row carries a synthetic patient + acquisition "
                          "record (`mrn`, age, sex, height, weight, BMI, blood pressure, heart rate measured from the "
                          "ECG, acquisition time). AFib rises with age, as in real populations. The worklist ranks the "
                          "day's flagged patients by AFib probability — the queue a cardiology reviewer would work "
                          "through."], 0, 29, 12, 2),

    # --- Live serving ----------------------------------------------------------------------------
    text("hdr-live", ["### Live serving — the app (endpoint calls logged by AI Gateway)\n", "\n",
                      f"Sources: `{ARR_PAY}` and `{AFIB_PAY}`, the endpoints' inference tables. Each "
                      "*Triage → Classify* click in the app adds a row to the triage table, and one to the classify "
                      "table when the ECG is flagged. Batch scoring never appears here. Execution time is the "
                      "model's inference time only (no network overhead). **These tiles show an error until AI "
                      "Gateway has delivered the first inference-table logs** (best effort, typically within about an "
                      "hour of the first app request): until then the tables have no columns but "
                      "`databricks_request_id`."],
         0, 31, 12, 3),
]
LIVE_Y = 34                                     # first row under the live-section header
live_layout = [
    counter("kpi-requests", "live_requests", None, "Endpoint requests", 0, LIVE_Y,
            agg=("count(databricks_request_id)", "COUNT(`databricks_request_id`)"),
            period=("daily(request_time)", 'DATE_TRUNC("DAY", `request_time`)')),
    counter("kpi-exec", "live_requests", None, "Avg model execution time (ms)", 4, LIVE_Y,
            agg=("avg(execution_duration_ms)", "AVG(`execution_duration_ms`)"),
            period=("daily(request_time)", 'DATE_TRUNC("DAY", `request_time`)'),
            fmt={"type": "number", "decimalPlaces": {"type": "max", "places": 1}}),
    counter("kpi-not-logged", "live_requests", None, "Requests not logged (payload > 1 MiB)", 8, LIVE_Y,
            agg=("sum(not_logged)", "SUM(`not_logged`)")),
    chart("live-volume", "line", "live_requests",
          {"fieldName": "request_hour", "displayName": "Hour (UTC)", "scale": {"type": "temporal"}},
          quant("count(databricks_request_id)", "Requests"), "Requests per hour, by endpoint", (0, LIVE_Y + 4, 6, 5),
          fields=[field("request_hour"), field("count(databricks_request_id)", "COUNT(`databricks_request_id`)"),
                  field("endpoint")],
          color=cat("endpoint", "Endpoint", ENDPOINT_COLORS), disaggregated=False),
    chart("live-exec", "line", "live_requests",
          {"fieldName": "request_hour", "displayName": "Hour (UTC)", "scale": {"type": "temporal"}},
          quant("avg(execution_duration_ms)", "Avg model execution time (ms)"),
          "Model execution time per hour, by endpoint", (6, LIVE_Y + 4, 6, 5),
          fields=[field("request_hour"), field("avg(execution_duration_ms)", "AVG(`execution_duration_ms`)"),
                  field("endpoint")],
          color=cat("endpoint", "Endpoint", ENDPOINT_COLORS), disaggregated=False),
    chart("live-arr-prob", "bar", "live_predictions", quant("score_bin", "Abnormal probability (bin)"),
          quant("count(score)", "ECGs scored live"), "Triage endpoint: returned abnormal probability",
          (0, LIVE_Y + 9, 6, 5), fields=[field("score_bin"), field("count(score)", "COUNT(`score`)")],
          disaggregated=False, filters=[f"`endpoint` = '{ARR_LABEL}'"]),
    chart("live-afib-prob", "bar", "live_predictions", quant("score_bin", "AFib probability (bin)"),
          quant("count(score)", "ECGs scored live"), "Classify endpoint: returned AFib probability",
          (6, LIVE_Y + 9, 6, 5), fields=[field("score_bin"), field("count(score)", "COUNT(`score`)")],
          disaggregated=False, filters=[f"`endpoint` = '{AFIB_LABEL}'"]),
    table("live-table", "live_requests", [
        ("request_time", "Request time (UTC)"), ("endpoint", "Endpoint"), ("status_code", "HTTP status"),
        ("execution_duration_ms", "Model execution (ms)"), ("request_kb", "Request (KB)"),
        ("not_logged", "Not logged")],
        "Latest endpoint requests", (0, LIVE_Y + 14, 12, 6)),
]
layout += live_layout
LINEAGE_Y = max(item["position"]["y"] + item["position"]["height"] for item in live_layout)
layout += [
    text("lineage-note", ["**One governed platform.** Batch lineage: "
                          f"`{config.BATCH_TABLE_PREFIX}<date>` → `{config.TRIAGED_TABLE_PREFIX}<date>` "
                          f"(+ `{TRI}`) → `{INF.lower()}`. Live: app → endpoints `{config.ARR_ENDPOINT}` / "
                          f"`{config.AFIB_ENDPOINT}` → `{ARR_PAY}` / `{AFIB_PAY}`. Both paths use the same "
                          f"`@{config.CHAMPION_ALIAS}` "
                          "model versions, promoted by the SETUP job's validation gate. Cheap triage screens every "
                          "ECG; the costlier AFib model runs only on the flagged."], 0, LINEAGE_Y, 12, 2),
]

dashboard = {
    "datasets": datasets,
    "pages": [{"name": "overview", "displayName": "ECG/AFib 12-lead: batch + live",
               "pageType": "PAGE_TYPE_CANVAS", "layoutVersion": "GRID_V1", "layout": layout}],
    # Palette only: the counters rendered blank with custom font / background colors, so those
    # stay at the workspace default.
    "uiSettings": {"theme": {
        "selectionColor": {"light": "#2272B4", "dark": "#8ACAFF"},
        "visualizationColors": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
                                "#e34948"],
        "widgetHeaderAlignment": "LEFT"}},
}

text_out = json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n"
if args.check:
    current = OUT.read_text() if OUT.exists() else ""
    if current != text_out:
        sys.exit(f"{OUT.name} is out of date: run python dashboards/{Path(__file__).name} and commit it")
    print(f"{OUT.name} is current")
else:
    OUT.write_text(text_out)
    print(f"wrote {OUT} ({len(datasets)} datasets, {len(layout)} widgets)")
