"""Single-source-of-truth checks: the app's copy of common/, the bundle's defaults and sync rules,
the job DAGs and compute, the schema and the model / endpoint names (only in config), and the
dashboard builder's two sections."""
import io
import json
import re
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest
import yaml

from common import config
from conftest import DEMO_ROOT

APP_COPY = DEMO_ROOT / "app" / "src" / "ecg_demo_react" / "backend" / "common"
MIRRORED = ["__init__.py", "config.py", "ecg_signal_gen.py", "patient_info.py"]

APP_PREFIX = "ecg12-demo-"
APP_TEMPLATE = APP_PREFIX + "${var.app_suffix}"     # the only shape the app resource name may take
APP_NAME_MAX = 30                                   # Databricks app-name length limit
# Apps deployed in the workspace on 2026-10-30 (`databricks apps list -p fevm-mmt-aws-usw2`),
# including the live 8-lead demo `ekg-demo` and the earlier manual deploy of this demo under the
# old prefix, `ekg12-demo-mmt`.
DEPLOYED_APPS = ["app-demo", "dbx-unifiedchat-app-prod", "ekg-demo", "ekg12-demo-mmt", "glucosphere-app",
                 "glucosphere-fw-v2-mmt", "gwb-app", "mcp-genesis-workbench"]
# The earlier run's (and the live 8-lead demo's) schema.
OLD_SCHEMA = re.compile(r"ekg_afib")
# The first 12-lead run's model and endpoint names (schema ekg_afib): prose may mention them, code may not.
OLD_MODEL_NAMES = re.compile(r"arr_model_12l|afib_demo_12l")
SKIP_DIRS = {"tests", ".build", "node_modules", "__dist__", ".venv"}
# Live resources this demo does not own keep their real (German-spelled) names, and prose may name
# them: the 8-lead app and its deployed source folder, the earlier run's schema, and the earlier
# manual deploy of this demo. Everything else is spelled ECG.
LIVE_RESOURCE_NAMES = re.compile(r"ekg12-demo-mmt|ekg_demo_react_build|ekg-demo|ekg_afib")


@pytest.mark.parametrize("module", MIRRORED)
def test_app_copy_of_common_is_byte_identical(module):
    assert (APP_COPY / module).read_bytes() == (DEMO_ROOT / "common" / module).read_bytes()


def test_app_copy_has_only_the_mirrored_modules():
    assert sorted(p.name for p in APP_COPY.glob("*.py")) == sorted(MIRRORED)


def _bundle():
    return yaml.safe_load((DEMO_ROOT / "databricks.yml").read_text())


def test_dev_folder_is_never_deployed():
    """dev/ holds developer analysis scripts and these tests: excluded from bundle sync, outside
    the app's wheel (built from app/ only), and imported by nothing that ships."""
    assert (DEMO_ROOT / "dev" / "README.md").is_file()
    assert (DEMO_ROOT / "dev" / "signal_variety" / "analyze_signal_variety.py").is_file()
    assert "dev/**" in _bundle()["sync"]["exclude"]
    shipped = [DEMO_ROOT / "databricks.yml", DEMO_ROOT / "app" / "pyproject.toml"] + [
        p for folder in ("common", "workflow", "dashboards", "resources", "app/src")
        for p in (DEMO_ROOT / folder).rglob("*")
        if p.suffix in {".py", ".ts", ".tsx", ".yml"} and not SKIP_DIRS & set(p.parts)]
    assert len(shipped) > 30
    for path in shipped:
        assert "signal_variety" not in path.read_text(), path.relative_to(DEMO_ROOT)


def test_bundle_defaults_match_config():
    bundle = _bundle()
    assert bundle["bundle"]["name"] == "biosignals-ecg12-afib"
    variables = bundle["variables"]
    assert variables["catalog"]["default"] == config.CATALOG
    assert variables["schema"]["default"] == config.SCHEMA == "biosignals"
    app = bundle["resources"]["apps"]["ecg12_afib_app"]
    assert app["source_code_path"] == "./app/.build"       # the bundle's own build folder
    dev = bundle["targets"]["dev"]
    assert dev["mode"] == "development"
    assert dev["workspace"]["profile"] == "fevm-mmt-aws-usw2"


def test_app_suffix_is_required_and_rendered_name_never_matches_a_deployed_app():
    """The user gives only the suffix; the rendered name is always `ecg12-demo-` + that suffix.

    So `--var app_suffix=ekg-demo` renders `ecg12-demo-ekg-demo`, and NO value can render to
    `ekg-demo` or any other deployed app: a rendered name starts with the prefix, those don't."""
    bundle = _bundle()
    assert "app_name" not in bundle["variables"]                        # old variable is gone
    assert "default" not in bundle["variables"]["app_suffix"]           # never silently defaulted
    apps = bundle["resources"]["apps"]
    assert list(apps) == ["ecg12_afib_app"]                             # exactly one app resource
    assert apps["ecg12_afib_app"]["name"] == APP_TEMPLATE
    for name, target in bundle["targets"].items():                     # no target swaps the name
        assert "apps" not in (target.get("resources") or {}), name
    for path in sorted((DEMO_ROOT / "resources").glob("*.yml")):       # no included file adds one
        assert "apps" not in (yaml.safe_load(path.read_text()).get("resources") or {}), path.name

    def render(suffix):
        return APP_TEMPLATE.replace("${var.app_suffix}", suffix)

    for app in DEPLOYED_APPS:
        assert not app.startswith(APP_PREFIX), app                      # so no suffix can collide
        assert render(app) != app                                       # e.g. app_suffix=ekg-demo
    assert render("ekg-demo") == "ecg12-demo-ekg-demo"
    assert len(APP_PREFIX) == 11
    assert render("mmt") == "ecg12-demo-mmt" and len(render("mmt")) == 14  # suggested suffix
    assert len(render("x" * 19)) == APP_NAME_MAX                        # longest suffix that fits
    assert len(render("x" * 20)) > APP_NAME_MAX


def _jobs():
    jobs = {}
    for path in sorted((DEMO_ROOT / "resources").glob("*.job.yml")):
        for key, job in yaml.safe_load(path.read_text())["resources"]["jobs"].items():
            jobs[key] = (path, job)
    return jobs


def test_two_jobs_with_the_expected_dags():
    """SETUP: data -> {train arr, train afib} -> validation gate -> serve. DAILY: generate -> triage -> classify."""
    jobs = _jobs()
    assert sorted(jobs) == ["ecg12_afib_daily", "ecg12_afib_setup"]
    dags = {key: {t["task_key"]: (Path(t["notebook_task"]["notebook_path"]).name,
                                  sorted(d["task_key"] for d in t.get("depends_on", [])))
                  for t in job["tasks"]}
            for key, (_, job) in jobs.items()}
    assert dags["ecg12_afib_setup"] == {
        "generate_training_data": ("01_generate_training_data.py", []),
        "train_arr_model": ("02_train_arr_model.py", ["generate_training_data"]),
        "train_afib_model": ("03_train_afib_model.py", ["generate_training_data"]),
        "validate_and_promote": ("04_validate_and_promote.py", ["train_afib_model", "train_arr_model"]),
        "serve_endpoints": ("05_serve_endpoints.py", ["validate_and_promote"]),
    }
    assert dags["ecg12_afib_daily"] == {
        "simulate_incoming_batch": ("06_generate_batch_data.py", []),
        "arr_triage": ("07_arr_triage.py", ["simulate_incoming_batch"]),
        "afib_classify": ("08_afib_classify.py", ["arr_triage"]),
    }
    # The gate and serving run only when everything upstream succeeded (no run_if override).
    assert all("run_if" not in t for _, job in jobs.values() for t in job["tasks"])


def test_job_tasks_point_at_existing_notebooks_on_serverless():
    for path, job in _jobs().values():
        assert "job_clusters" not in job
        for task in job["tasks"]:
            assert (path.parent / task["notebook_task"]["notebook_path"]).is_file()
            assert not {"existing_cluster_id", "new_cluster", "job_cluster_key"} & set(task)
    setup, daily = _jobs()["ecg12_afib_setup"][1], _jobs()["ecg12_afib_daily"][1]
    # SETUP trains + gates on serverless GPU; DAILY scores with spark_udf, whose Spark workers need
    # torch: standard serverless with the ML environment (torch CPU build), no GPU.
    assert {e["environment_key"]: e["spec"] for e in setup["environments"]} == {
        "aiv5": {"base_environment": "databricks_ai_v5"}}
    assert {e["environment_key"]: e["spec"] for e in daily["environments"]} == {
        "ml_v5": {"base_environment": "databricks_ml_v5"}}
    params = {p["name"]: p["default"] for p in daily["parameters"]}
    assert params[config.RUN_DATE_PARAM] == "{{job.start_time.iso_date}}"
    assert daily["schedule"]["pause_status"] == "PAUSED"

    def accelerators(job):
        return {t["task_key"]: (t.get("environment_key"), t.get("compute", {}).get("hardware_accelerator"))
                for t in job["tasks"]}

    gpu, ml, standard = ("aiv5", "GPU_1xA10"), ("ml_v5", None), (None, None)
    assert accelerators(daily) == {"simulate_incoming_batch": standard, "arr_triage": ml, "afib_classify": ml}
    assert accelerators(setup) == {
        "generate_training_data": standard, "train_arr_model": gpu, "train_afib_model": gpu,
        "validate_and_promote": gpu, "serve_endpoints": standard}


def _string_literals(path):
    """String literals of a Python file (f-string text too, which Python >= 3.12 tokenizes apart)."""
    kinds = {tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING)}
    tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
    return [t.string for t in tokens if t.type in kinds]


def _yaml_scalars(node):
    if isinstance(node, dict):
        return [s for v in node.values() for s in _yaml_scalars(v)]
    if isinstance(node, list):
        return [s for v in node for s in _yaml_scalars(v)]
    return [str(node)]


def test_model_and_endpoint_names_come_only_from_config():
    """No code, job, bundle or app file names a model or endpoint: notebooks, dashboard builder and
    app read config.ARR_MODEL_NAME / AFIB_MODEL_NAME / *_ENDPOINT. The first 12-lead run's names
    (arr_model_12l, afib_demo_12l) may appear only in prose (comments, docstrings, markdown)."""
    names = {config.ARR_MODEL_NAME, config.AFIB_MODEL_NAME, config.ARR_ENDPOINT, config.AFIB_ENDPOINT}
    assert names == {"bio_ecg12_arr", "bio_ecg12_afib"}                       # each endpoint = its model
    py_files = [p for p in DEMO_ROOT.rglob("*.py") if not SKIP_DIRS & set(p.relative_to(DEMO_ROOT).parts)]
    for path in py_files:
        for literal in _string_literals(path):
            if literal.startswith(('"""', "'''")):
                continue                                                     # docstring prose
            assert not OLD_MODEL_NAMES.search(literal), f"{path.relative_to(DEMO_ROOT)}: {literal[:80]}"
            if path.name != "config.py":
                assert not names & set(re.findall(r"\w+", literal)), f"{path.relative_to(DEMO_ROOT)}: {literal[:80]}"
    for path in [DEMO_ROOT / "databricks.yml", *sorted((DEMO_ROOT / "resources").glob("*.yml"))]:
        for value in _yaml_scalars(yaml.safe_load(path.read_text())):
            assert not OLD_MODEL_NAMES.search(value) and not names & set(re.findall(r"\w+", value)), path.name
    ui_sources = [p for p in (APP_COPY.parent.parent / "ui").rglob("*.ts*") if "node_modules" not in p.parts]
    assert ui_sources
    for path in ui_sources:
        source = path.read_text()
        assert not OLD_MODEL_NAMES.search(source) and not names & set(re.findall(r"\w+", source)), path.name


def test_no_hardcoded_schema_outside_config():
    """The UC schema is config.SCHEMA everywhere: no code, job or app file names a schema itself.
    The earlier run's schema `ekg_afib` may appear only in prose (comments, docstrings, markdown)."""
    py_files = [p for p in DEMO_ROOT.rglob("*.py") if not SKIP_DIRS & set(p.relative_to(DEMO_ROOT).parts)]
    assert len(py_files) > 15
    for path in py_files:
        for literal in _string_literals(path):
            if literal.startswith(('"""', "'''")):
                continue                                                     # docstring prose
            assert not OLD_SCHEMA.search(literal), f"{path.relative_to(DEMO_ROOT)}: {literal[:80]}"
            if path.name != "config.py":
                assert literal.strip("'\"") != config.SCHEMA, f"{path.relative_to(DEMO_ROOT)}: {literal}"

    for path in [DEMO_ROOT / "databricks.yml", *sorted((DEMO_ROOT / "resources").glob("*.yml"))]:
        values = _yaml_scalars(yaml.safe_load(path.read_text()))
        assert not [v for v in values if OLD_SCHEMA.search(v)], path.name
        assert (config.SCHEMA in values) == (path.name == "databricks.yml"), path.name   # bundle default only
    for path in (APP_COPY.parent.parent / "ui").rglob("*.ts*"):
        source = path.read_text()
        assert not OLD_SCHEMA.search(source) and f'"{config.SCHEMA}"' not in source, path.name


def test_ecg_spelling_everywhere_except_live_resource_names():
    """Docs, code, config and app files spell the demo ECG. The old spelling may appear only as part
    of a LIVE_RESOURCE_NAMES name. This file is skipped: it defines the check."""
    generated = {".build", "node_modules", "__dist__", ".venv", ".databricks", ".pytest_cache", "__pycache__"}
    suffixes = {".py", ".md", ".yml", ".toml", ".json", ".ts", ".tsx", ".html", ".css", ".sh"}
    repo_root = DEMO_ROOT.parent.parent
    paths = [repo_root / "README.md"] + [
        p for p in sorted(DEMO_ROOT.rglob("*"))
        if p.is_file() and (p.suffix in suffixes or p.name == ".gitignore")
        and not generated & set(p.relative_to(DEMO_ROOT).parts) and p != Path(__file__).resolve()]
    assert len(paths) > 60
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            rest = LIVE_RESOURCE_NAMES.sub("", line).lower()
            assert "ekg" not in rest, f"{path.relative_to(repo_root)}:{number}: {line.strip()[:100]}"


def _dashboard(tmp_path, *extra):
    out = tmp_path / "body.json"
    builder = DEMO_ROOT / "dashboards" / "build_ecg_demo_dashboard.py"
    subprocess.run([sys.executable, str(builder), str(out), "--date", "2026-10-29", *extra], check=True,
                   capture_output=True, text=True)
    return json.loads(json.loads(out.read_text())["serialized_dashboard"])


def test_dashboard_has_separate_batch_and_live_sections(tmp_path):
    dashboard = _dashboard(tmp_path)
    sql = {d["name"]: "".join(d["queryLines"]) for d in dashboard["datasets"]}
    batch = {"funnel", "arrivals", "triage_accuracy", "prob_hist", "trend", "afib_by_age", "worklist"}
    live = {"live_requests", "live_predictions"}
    assert set(sql) == batch | live
    location = f"{config.CATALOG}.{config.SCHEMA}."
    for name, query in sql.items():
        assert location in query and "ekg_afib." not in query, name               # the configured schema
        assert not re.search(r"dailyOrweekly_arr_afib_inference(?!_12l)", query), name  # never 8-lead
        assert not re.search(r"\b(batch_data|arr_triaged)_\d{8}\b", query), name
        # Batch tiles read only the batch tables; live tiles only the endpoints' payload tables.
        reads_payload = "_payload" in query
        assert reads_payload == (name in live), name
    assert "arr_triaged_12l_20261029" in sql["funnel"] and "acquisition_datetime_utc" in sql["arrivals"]
    assert "dailyOrweekly_arr_afib_inference_12l" in sql["trend"]
    assert "mrn" in sql["worklist"] and "batch_date = DATE'2026-10-29'" in sql["worklist"]
    for endpoint in (config.ARR_ENDPOINT, config.AFIB_ENDPOINT):                   # bio_ecg12_*_payload
        table = config.payload_table(endpoint)
        assert table.endswith(("bio_ecg12_arr_payload", "bio_ecg12_afib_payload"))
        assert table in sql["live_requests"] and table in sql["live_predictions"]
    assert not OLD_MODEL_NAMES.search(" ".join(sql.values()))
    # Only the response JSON is parsed; the request JSON (the ECG) is only measured.
    assert "from_json(request" not in sql["live_requests"] + sql["live_predictions"]
    assert "length(request)" in sql["live_requests"]
    assert "from_json(response, 'predictions ARRAY<STRUCT<arr_prob: DOUBLE" in sql["live_predictions"]
    assert "from_json(response, 'predictions ARRAY<ARRAY<DOUBLE>>')" in sql["live_predictions"]

    page = dashboard["pages"][0]
    assert page["pageType"] == "PAGE_TYPE_CANVAS" and page["layoutVersion"] == "GRID_V1"
    layout = page["layout"]
    headers = {item["widget"]["name"]: "".join(item["widget"]["multilineTextboxSpec"]["lines"])
               for item in layout if "multilineTextboxSpec" in item["widget"]}
    assert headers["hdr-batch"].startswith("### Batch monitoring") and "endpoints not called" in headers["hdr-batch"]
    assert headers["hdr-live"].startswith("### Live serving") and "_payload" in headers["hdr-live"]
    widgets = {item["widget"]["name"]: item for item in layout}
    assert {"triage-bar", "afib-age-bar", "worklist-table", "arrivals-bar", "live-volume", "live-exec",
            "live-arr-prob", "live-afib-prob", "live-table"} <= set(widgets)
    # Every batch widget sits above the live header, every live widget below it.
    live_y = widgets["hdr-live"]["position"]["y"]
    for name, item in widgets.items():
        datasets = {q["query"]["datasetName"] for q in item["widget"].get("queries", [])}
        if datasets:
            assert (item["position"]["y"] > live_y) == bool(datasets & live), name

    rows = {}
    for item in layout:                                                          # 12-column rows, no gaps
        rows[item["position"]["y"]] = rows.get(item["position"]["y"], 0) + item["position"]["width"]
    assert set(rows.values()) == {12}
    datasets_used = set()
    for item in layout:                                                          # field bindings match
        widget = item["widget"]
        if "spec" not in widget:
            continue
        names = {f["name"] for q in widget["queries"] for f in q["query"]["fields"]}
        datasets_used |= {q["query"]["datasetName"] for q in widget["queries"]}
        used = set()
        for key, enc in widget["spec"]["encodings"].items():
            if key == "columns":
                used |= {c["fieldName"] for c in enc}
                continue
            if "fieldName" in enc:
                used.add(enc["fieldName"])
            used |= {f["fieldName"] for f in enc.get("fields", [])}
        assert used <= names, (widget["name"], used - names)
        version = {"counter": 2, "table": 2, "bar": 3, "line": 3}[widget["spec"]["widgetType"]]
        assert widget["spec"]["version"] == version, widget["name"]
    assert datasets_used == set(sql)                                             # no orphan datasets


def test_dashboard_location_can_be_overridden(tmp_path):
    sql = "".join("".join(d["queryLines"]) for d in _dashboard(tmp_path, "--schema", "other")["datasets"])
    assert f"{config.CATALOG}.other.arr_triaged_12l_20261029" in sql and f".{config.SCHEMA}." not in sql
