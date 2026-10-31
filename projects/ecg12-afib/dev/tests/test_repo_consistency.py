"""Single-source-of-truth and portability checks: the app's copy of common/, the bundle's
variables, targets and resources, the job DAGs and compute, model / endpoint / schema names only
in config, no workspace-specific values in any committed file, and the generated dashboard."""
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
APP_NAME_PATTERN = re.compile(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?")   # lowercase letters, digits, hyphens
APP_NAME_MAX = 30                                                # Databricks app-name length limit
SKIP_DIRS = {"tests", ".build", "node_modules", "__dist__", ".venv", ".databricks", "__pycache__",
             ".pytest_cache", ".apx"}
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".json", ".ts", ".tsx", ".html", ".css", ".sh",
                 ".lock", ".txt"}


@pytest.mark.parametrize("module", MIRRORED)
def test_app_copy_of_common_is_byte_identical(module):
    assert (APP_COPY / module).read_bytes() == (DEMO_ROOT / "common" / module).read_bytes()


def test_app_copy_has_only_the_mirrored_modules():
    assert sorted(p.name for p in APP_COPY.glob("*.py")) == sorted(MIRRORED)


def _bundle():
    return yaml.safe_load((DEMO_ROOT / "databricks.yml").read_text())


def _project_files():
    """Committed-style text files of the project (generated and dependency folders skipped)."""
    return [p for p in sorted(DEMO_ROOT.rglob("*"))
            if p.is_file() and (p.suffix in TEXT_SUFFIXES or p.name in {".gitignore", ".python-version"})
            and not (SKIP_DIRS - {"tests"}) & set(p.relative_to(DEMO_ROOT).parts)]


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


def test_bundle_variables_have_no_workspace_defaults():
    """The catalog and warehouse are required (no default); the schema defaults to config.SCHEMA;
    the app name is the generic ecg12-demo; no target names a host or profile."""
    bundle = _bundle()
    assert bundle["bundle"]["name"] == "biosignals-ecg12-afib"
    variables = bundle["variables"]
    assert set(variables) == {"catalog", "schema", "warehouse_id", "endpoint_prefix", "app_name", "viewer_group"}
    for required in ("catalog", "warehouse_id"):
        assert not {"default", "lookup"} & set(variables[required]), required
        assert variables[required]["description"].startswith("REQUIRED."), required
    assert variables["schema"]["default"] == config.SCHEMA == "biosignals"
    app_name = variables["app_name"]["default"]
    assert app_name == "ecg12-demo"
    assert APP_NAME_PATTERN.fullmatch(app_name) and len(app_name) <= APP_NAME_MAX
    assert "workspace-wide" in variables["app_name"]["description"]
    assert variables["viewer_group"]["default"] == "users"
    # Workspace-wide endpoint names: one prefix, default = config's (the app and 05_ fall back to it).
    assert variables["endpoint_prefix"]["default"] == config.ENDPOINT_PREFIX
    assert "workspace-wide" in variables["endpoint_prefix"]["description"]
    config.endpoint_names(variables["endpoint_prefix"]["default"])          # a valid prefix
    for name, target in bundle["targets"].items():
        assert target["mode"] == "development", name
        assert not {"host", "profile"} & set(target.get("workspace") or {}), name
        assert "variables" not in target, name                  # no target pins a catalog or warehouse
    assert "workspace" not in bundle


def test_no_app_target_needs_no_local_build_and_full_adds_the_app():
    """no_app (default): jobs + dashboard only, deployable from the workspace UI (no artifact build,
    no app). full: the app with its locally built source, artifact build and environment."""
    bundle = _bundle()
    targets = bundle["targets"]
    assert set(targets) == {"no_app", "full"}
    assert targets["no_app"].get("default") is True and "default" not in targets["full"]
    assert "artifacts" not in bundle and "apps" not in (bundle.get("resources") or {})
    assert not {"artifacts", "resources"} & set(targets["no_app"])
    assert "app/**" in targets["no_app"]["sync"]["exclude"]
    for path in sorted((DEMO_ROOT / "resources").glob("*.yml")):        # no included file adds an app
        assert "apps" not in (yaml.safe_load(path.read_text()).get("resources") or {}), path.name

    full = targets["full"]
    assert full["sync"]["include"] == ["app/.build"]
    assert full["artifacts"] == {"ecg12_afib_app": {"path": "./app", "build": "./build_from_lockfiles.sh"}}
    apps = full["resources"]["apps"]
    assert list(apps) == ["ecg12_afib_app"]
    app = apps["ecg12_afib_app"]
    assert app["name"] == "${var.app_name}" and app["source_code_path"] == "./app/.build"
    # The bundle's command overrides app.yml's; keep them the same.
    assert app["config"]["command"] == yaml.safe_load((DEMO_ROOT / "app" / "app.yml").read_text())["command"]
    env = {e["name"]: e["value"] for e in app["config"]["env"]}
    assert env == {"ECG_DEMO_REACT_CATALOG": "${var.catalog}", "ECG_DEMO_REACT_UC_SCHEMA": "${var.schema}",
                   "ECG_DEMO_REACT_ENDPOINT_PREFIX": "${var.endpoint_prefix}",
                   "ECG_DEMO_REACT_DASHBOARD_ID": "${resources.dashboards.ecg12_afib_dashboard.id}",
                   "ECG_DEMO_REACT_DAILY_JOB_ID": "${resources.jobs.ecg12_afib_daily.id}"}
    assert "ecg12_afib_daily" in _jobs()                                     # the job that id refers to
    assert app["permissions"] == [{"group_name": "${var.viewer_group}", "level": "CAN_USE"}]
    # Those env names are the backend's AppConfig fields (env prefix ECG_DEMO_REACT_).
    app_config = (DEMO_ROOT / "app" / "src" / "ecg_demo_react" / "backend" / "core" / "_config.py").read_text()
    for field in ("catalog: str", "uc_schema: str", "endpoint_prefix: str", "dashboard_id: str | None",
                  "daily_job_id: str | None"):
        assert f"    {field}" in app_config, field
    pyproject = (DEMO_ROOT / "app" / "pyproject.toml").read_text()
    assert 'app-slug = "ecg_demo_react"' in pyproject                        # -> prefix ECG_DEMO_REACT_


def test_dashboard_is_a_bundle_resource_on_the_bundle_variables():
    dashboards = yaml.safe_load((DEMO_ROOT / "resources" / "ecg12_afib.dashboard.yml").read_text())["resources"]["dashboards"]
    assert list(dashboards) == ["ecg12_afib_dashboard"]
    dash = dashboards["ecg12_afib_dashboard"]
    assert (DEMO_ROOT / "resources" / dash["file_path"]).resolve() == DEMO_ROOT / "dashboards" / "ecg12_afib.lvdash.json"
    assert (dash["warehouse_id"], dash["dataset_catalog"], dash["dataset_schema"]) == (
        "${var.warehouse_id}", "${var.catalog}", "${var.schema}")
    assert dash["embed_credentials"] is True and "parent_path" not in dash
    assert dash["permissions"] == [{"group_name": "${var.viewer_group}", "level": "CAN_READ"}]


def test_app_bounds_every_endpoint_call_and_tags_its_warmup_pings():
    """Read, not run (the test environment has no FastAPI): one serving client bounded by
    SERVING_TIMEOUT_S for every live call, /api/warmup tagged with WARMUP_REQUEST_ID, and the UI's
    limit on Triage -> Classify built on the backend's bound (2 x (30 + 10) = 80 s)."""
    assert (config.SERVING_TIMEOUT_S, config.WARMUP_INTERVAL_S, config.WARMUP_REQUEST_ID) == (30, 600, "app-warmup")
    assert config.WARMUP_INTERVAL_S < 30 * 60                     # inside the endpoints' 30 min idle window
    backend = DEMO_ROOT / "app" / "src" / "ecg_demo_react" / "backend"
    router = (backend / "router.py").read_text()
    assert ("http_timeout_seconds=config.SERVING_TIMEOUT_S,\n"
            "            retry_timeout_seconds=config.SERVING_TIMEOUT_S," in router)
    assert router.count("serving.serving_endpoints.query(") == 1             # every call goes through _predict
    assert "ws.serving_endpoints" not in router
    assert '@router.post("/warmup"' in router and "client_request_id=config.WARMUP_REQUEST_ID" in router
    for status in (429, 502, 503):
        assert f"_endpoint_error({status}," in router, status
    for field in ("daily_job_id=settings.daily_job_id", "serving_timeout_s=config.SERVING_TIMEOUT_S",
                  "warmup_interval_s=config.WARMUP_INTERVAL_S", "arr_endpoint=settings.endpoints[0]",
                  "afib_endpoint=settings.endpoints[1]", "arr_endpoint, afib_endpoint = settings.endpoints",
                  "endpoints = list(settings.endpoints)"):
        assert field in router, field                                       # endpoint names: endpoint_prefix
    assert "return endpoint_names(self.endpoint_prefix)" in (backend / "core" / "_config.py").read_text()
    ui = (DEMO_ROOT / "app" / "src" / "ecg_demo_react" / "ui" / "routes" / "index.tsx").read_text()
    assert "const UI_MARGIN_S = 10;" in ui and "2 * (cfg.serving_timeout_s + UI_MARGIN_S)" in ui
    for text in ("Cancel</Button>", "Retry</Button>", "New reading", "Reading…", "Each ECG is read in and scored live",
                 "Synthetic, non-PII", 'document.addEventListener("visibilitychange"', "c.daily_job_id"):
        assert text in ui, text
    api = (DEMO_ROOT / "app" / "src" / "ecg_demo_react" / "ui" / "lib" / "api.ts").read_text()
    for generated in ("export const warmup = async", "daily_job_id: string | null;", "serving_timeout_s: number;",
                      "export interface ServingError {"):
        assert generated in api, generated                                 # the TS client was regenerated


def test_deploy_script_is_valid_bash_and_reads_endpoint_names_from_config():
    script = (DEMO_ROOT / "deploy.sh").read_text()
    subprocess.run(["bash", "-n", str(DEMO_ROOT / "deploy.sh")], check=True)
    assert script.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in script
    assert "config.endpoint_names(sys.argv[1])" in script and '"$ENDPOINT_PREFIX"' in script
    assert '--var "endpoint_prefix=$ENDPOINT_PREFIX"' in script
    assert not set(config.endpoint_names()) & set(re.findall(r"\w+", script))   # (the help text shows the default prefix)
    # --skip-embed-allowlist makes no settings call: the skip branch comes before the first one.
    step8 = script[script.index('step "8/8'):]
    assert step8.index("SKIP_ALLOWLIST -eq 1") < step8.index("databricks settings")
    for step in ("bundle validate --strict", "bundle deploy", "bundle run ecg12_afib_setup",
                 "serving-endpoints update-permissions", "CAN_QUERY", "bundle run ecg12_afib_app",
                 "aibi-dashboard-embedding-approved-domains"):
        assert step in script, step


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
    # SETUP trains + gates on serverless GPU; DAILY scores on the driver with standard serverless and
    # the ML v5 environment (torch CPU build), no GPU.
    assert {e["environment_key"]: e["spec"] for e in setup["environments"]} == {
        "aiv5": {"base_environment": "databricks_ai_v5"}}
    assert {e["environment_key"]: e["spec"] for e in daily["environments"]} == {
        "ml_v5": {"base_environment": "databricks_ml_v5"}}
    # Only SETUP serves endpoints (05_), so only SETUP gets the endpoint prefix.
    assert {p["name"]: p["default"] for p in setup["parameters"]}["endpoint_prefix"] == "${var.endpoint_prefix}"
    assert "endpoint_prefix" not in {p["name"] for p in daily["parameters"]}
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
    """No code, job, bundle or app file names a model or endpoint: notebooks, dashboard builder,
    deploy script and app read config.ARR_MODEL_NAME / AFIB_MODEL_NAME and config.endpoint_names
    of the endpoint_prefix parameter. The only literal prefix outside config is the bundle
    variable's default."""
    names = {config.ARR_MODEL_NAME, config.AFIB_MODEL_NAME, config.ENDPOINT_PREFIX, *config.endpoint_names()}
    assert names == {"bio_ecg12", "bio_ecg12_arr", "bio_ecg12_afib"}
    py_files = [p for p in DEMO_ROOT.rglob("*.py") if not SKIP_DIRS & set(p.relative_to(DEMO_ROOT).parts)]
    for path in py_files:
        for literal in _string_literals(path):
            if literal.startswith(('"""', "'''")) or path.name == "config.py":
                continue                                                     # docstring prose
            assert not names & set(re.findall(r"\w+", literal)), f"{path.relative_to(DEMO_ROOT)}: {literal[:80]}"
    for path in [DEMO_ROOT / "databricks.yml", *sorted((DEMO_ROOT / "resources").glob("*.yml"))]:
        bundle = yaml.safe_load(path.read_text())
        if path.name == "databricks.yml":
            bundle["variables"]["endpoint_prefix"].pop("default")             # checked against config above
        for value in _yaml_scalars(bundle):
            assert not names & set(re.findall(r"\w+", value)), path.name
    ui_sources = [p for p in (APP_COPY.parent.parent / "ui").rglob("*.ts*") if "node_modules" not in p.parts]
    assert ui_sources
    for path in ui_sources:
        assert not names & set(re.findall(r"\w+", path.read_text())), path.name


def test_no_hardcoded_schema_outside_config():
    """The UC schema is config.SCHEMA everywhere: no code, job or app file names a schema itself;
    the bundle's `schema` variable defaults to it."""
    py_files = [p for p in DEMO_ROOT.rglob("*.py") if not SKIP_DIRS & set(p.relative_to(DEMO_ROOT).parts)]
    assert len(py_files) > 15
    for path in py_files:
        if path.name == "config.py":
            continue
        for literal in _string_literals(path):
            if not literal.startswith(('"""', "'''")):
                assert literal.strip("'\"") != config.SCHEMA, f"{path.relative_to(DEMO_ROOT)}: {literal}"
    for path in [DEMO_ROOT / "databricks.yml", *sorted((DEMO_ROOT / "resources").glob("*.yml"))]:
        values = _yaml_scalars(yaml.safe_load(path.read_text()))
        assert (config.SCHEMA in values) == (path.name == "databricks.yml"), path.name   # bundle default only
    for path in (APP_COPY.parent.parent / "ui").rglob("*.ts*"):
        assert f'"{config.SCHEMA}"' not in path.read_text(), path.name


# Values that tie a file to one workspace or person. This file is skipped: it defines them.
WORKSPACE_SPECIFIC = {
    "workspace host": re.compile(r"https?://[a-z0-9-]+\.(?:cloud\.databricks\.com|azuredatabricks\.net|gcp\.databricks\.com)"),
    "personal workspace path": re.compile(r"/Users/[^/<\s\"'`]+@"),
    "Databricks email": re.compile(r"[\w.+-]+@databricks\.com"),
    "internal package mirror": re.compile(r"(?:pypi|npm)-proxy"),
    "workspace or dashboard id": re.compile(r"\b(?:\d{15,16}|[0-9a-f]{32})\b"),
    "old EKG spelling": re.compile(r"ekg", re.IGNORECASE),
}


def test_no_workspace_specific_values_in_committed_files():
    """No host, personal path, email, internal mirror, workspace / dashboard id or old spelling in any
    project file, lockfiles included (ids are not checked in lockfiles: their hashes look alike)."""
    paths = [p for p in _project_files() if p != Path(__file__).resolve()]
    assert len(paths) > 60 and DEMO_ROOT / "app" / "uv.lock" in paths and DEMO_ROOT / "app" / "bun.lock" in paths
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            for what, pattern in WORKSPACE_SPECIFIC.items():
                if what == "workspace or dashboard id" and path.suffix in {".lock", ".json"}:
                    continue
                assert not pattern.search(line), f"{what}: {path.relative_to(DEMO_ROOT)}:{number}: {line.strip()[:100]}"


DASHBOARD = DEMO_ROOT / "dashboards" / "ecg12_afib.lvdash.json"
BUILDER = DEMO_ROOT / "dashboards" / "build_ecg_demo_dashboard.py"


def test_committed_dashboard_is_what_the_builder_generates():
    result = subprocess.run([sys.executable, "-B", str(BUILDER), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_dashboard_has_separate_batch_and_live_sections_on_bare_tables():
    dashboard = json.loads(DASHBOARD.read_text())
    sql = {d["name"]: "".join(d["queryLines"]) for d in dashboard["datasets"]}
    batch = {"funnel", "arrivals", "triage_accuracy", "prob_hist", "trend", "afib_by_age", "worklist"}
    live = {"live_requests", "live_predictions", "live_warmups"}
    assert set(sql) == batch | live
    assert not any({"catalog", "schema"} & set(d) for d in dashboard["datasets"])   # the bundle sets them
    history, results = config.TRIAGE_HISTORY_TABLE, config.INFERENCE_TABLE
    payloads = {config.payload_table_name(m) for m in (config.ARR_MODEL_NAME, config.AFIB_MODEL_NAME)}
    assert payloads == {"bio_ecg12_arr_payload", "bio_ecg12_afib_payload"}
    for name, query in sql.items():
        tables = set(re.findall(r"\b(?:FROM|JOIN)\s+([\w.`]+)", query))
        tables.discard("(SELECT")
        assert tables and all("." not in t for t in tables), (name, tables)      # bare names only
        assert not re.search(r"_\d{8}\b", query), name                         # no per-date table
        # Batch tiles read only the batch tables; live tiles only the endpoints' payload tables.
        assert tables <= ({history, results} if name in batch else payloads), (name, tables)
    latest = f"batch_date = (SELECT max(batch_date) FROM {history})"
    for name in batch - {"trend"}:
        assert latest in sql[name], name                                         # single-day tiles: latest day
    assert latest not in sql["trend"] and results in sql["trend"]                # the trend: every day
    assert "acquisition_datetime_utc" in sql["arrivals"] and "mrn" in sql["worklist"]
    for table in payloads:
        assert table in sql["live_requests"] and table in sql["live_predictions"] and table in sql["live_warmups"]
    # The app's warm-up pings: left out of every live tile (NULL client_request_id kept), counted apart.
    not_warmup = f"client_request_id IS DISTINCT FROM '{config.WARMUP_REQUEST_ID}'"
    assert sql["live_requests"].count(not_warmup) == sql["live_predictions"].count(not_warmup) == 2
    assert sql["live_warmups"].count(f"client_request_id = '{config.WARMUP_REQUEST_ID}'") == 2
    assert not_warmup not in sql["live_warmups"]
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
    # The no-data limitation is stated on the dashboard itself.
    assert "until the DAILY job has run once" in headers["hdr-batch"]
    assert "until AI Gateway has delivered the first inference-table logs" in headers["hdr-live"]
    widgets = {item["widget"]["name"]: item for item in layout}
    assert {"triage-bar", "afib-age-bar", "worklist-table", "arrivals-bar", "live-volume", "live-exec",
            "live-arr-prob", "live-afib-prob", "live-table", "kpi-warmups"} <= set(widgets)
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
