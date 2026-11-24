"""Single-source-of-truth and portability checks: the app's copy of common/, the bundle's
variables, targets and resources, the job DAGs and compute, model / endpoint / schema names only
in config, no workspace-specific values in any committed file, and the generated dashboard."""
import io
import json
import re
import shutil
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest
import yaml
from conftest import DEMO_ROOT, MLFLOW_SKINNY_REQUIRES, PIP_LINES, PREINSTALLED, SP_EXPERIMENT, pip_lines, pip_pins
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from common import config

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
    assert set(variables) == {"catalog", "schema", "warehouse_id", "endpoint_prefix", "mlflow_experiment",
                              "app_name", "viewer_group", "embed_credentials", "remove_after"}
    for required in ("catalog", "warehouse_id"):
        assert not {"default", "lookup"} & set(variables[required]), required
        assert variables[required]["description"].startswith("REQUIRED."), required
    assert variables["schema"]["default"] == config.SCHEMA == "biosignals"
    app_name = variables["app_name"]["default"]
    assert app_name == "ecg12-demo"
    assert APP_NAME_PATTERN.fullmatch(app_name) and len(app_name) <= APP_NAME_MAX
    assert "workspace-wide" in variables["app_name"]["description"]
    assert variables["viewer_group"]["default"] == "users"
    # The deployer's credentials for every dashboard viewer: on for this synthetic data, off for real data.
    assert variables["embed_credentials"]["default"] is True
    assert "real data" in variables["embed_credentials"]["description"]
    # One schema per deployment: the schema's description says so.
    assert "One schema per deployment" in variables["schema"]["description"]
    # Workspace-wide endpoint names: one prefix, default = config's (the app and 05_ fall back to it).
    assert variables["endpoint_prefix"]["default"] == config.ENDPOINT_PREFIX
    assert "workspace-wide" in variables["endpoint_prefix"]["description"]
    config.endpoint_names(variables["endpoint_prefix"]["default"])          # a valid prefix
    # One MLflow experiment per deploying identity, in its home folder, named after the bundle; no user
    # is hardcoded. The folder need not exist (a service principal may have none): the notebooks
    # create it before the experiment (tracking.use_experiment, test_tracking.py).
    experiment = variables["mlflow_experiment"]["default"]
    assert experiment == f"/Users/${{workspace.current_user.userName}}/{bundle['bundle']['name']}"
    assert experiment.replace("${workspace.current_user.userName}", "<id>") == SP_EXPERIMENT.replace(
        SP_EXPERIMENT.split("/")[2], "<id>")                       # the path the SP tests exercise
    assert "bundle destroy" in variables["mlflow_experiment"]["description"]
    for name, target in bundle["targets"].items():
        assert target["mode"] == "development", name
        assert not {"host", "profile"} & set(target.get("workspace") or {}), name
        assert "variables" not in target, name                  # no target pins a catalog or warehouse
    assert "workspace" not in bundle


def test_variable_overrides_template_holds_placeholders_for_the_required_variables():
    """variable-overrides.example.json (copied to .databricks/bundle/<target>/variable-overrides.json)
    names bundle variables only, has every required one, and holds only <...> placeholders, so no
    workspace value can be committed in it. It lists no variable with a default: the defaults stay
    in databricks.yml only (docs/deploy.md, Configuration, shows how to add one)."""
    path = DEMO_ROOT / "variable-overrides.example.json"
    template = json.loads(path.read_text())
    variables = _bundle()["variables"]
    required = {name for name, variable in variables.items() if "default" not in variable}
    assert required == {"catalog", "warehouse_id"}
    assert set(template) <= set(variables), set(template) - set(variables)
    assert required <= set(template) and not {name for name in template if "default" in variables[name]}
    assert all(isinstance(value, str) and re.fullmatch(r"<[a-z][a-z_-]*>", value) for value in template.values()), template
    assert path in _project_files()                    # the leak check below reads it too
    assert ".databricks/" in (DEMO_ROOT.parent.parent / ".gitignore").read_text().splitlines()   # the copy is never committed


def test_no_app_target_needs_no_local_build_and_full_adds_the_app():
    """no_app (default): jobs + dashboard only, deployable from the workspace UI (no artifact build,
    no app). full: the app with its locally built source, artifact build and environment."""
    bundle = _bundle()
    targets = bundle["targets"]
    assert set(targets) == {"no_app", "full"}
    assert targets["no_app"].get("default") is True and "default" not in targets["full"]
    assert "artifacts" not in bundle and "apps" not in (bundle.get("resources") or {})
    assert "artifacts" not in targets["no_app"]
    assert "app/**" in targets["no_app"]["sync"]["exclude"]
    # no_app renames its dashboard (and changes nothing else) so the two deployments' dashboards differ.
    dashboard = yaml.safe_load((DEMO_ROOT / "resources" / "ecg12_afib.dashboard.yml").read_text())
    base_name = dashboard["resources"]["dashboards"]["ecg12_afib_dashboard"]["display_name"]
    assert targets["no_app"]["resources"] == {
        "dashboards": {"ecg12_afib_dashboard": {"display_name": f"{base_name} [no_app]"}}}
    assert "dashboards" not in targets["full"]["resources"]
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
    assert dash["embed_credentials"] == "${var.embed_credentials}" and "parent_path" not in dash
    assert dash["permissions"] == [{"group_name": "${var.viewer_group}", "level": "CAN_READ"}]


def test_app_bounds_every_endpoint_call_and_tags_its_warmup_pings():
    """Read, not run (this environment has no FastAPI; test_app_backend.py runs the routes in the
    app's own): one serving client bounded by SERVING_TIMEOUT_S for every live call, /api/warmup
    tagged with WARMUP_REQUEST_ID, the UI's limit on Triage -> Classify built on the backend's bound
    (2 x (30 + 10) = 80 s), and its not-for-clinical-use notices."""
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
                 "Synthetic, non-PII", 'document.addEventListener("visibilitychange"', "c.daily_job_id",
                 "<strong>Not for clinical use.</strong>", "{analysis.notice}", "retryAfterS(err)",
                 "<Ext href={SOURCE_REPO_URL}", 'title="Source code on GitHub"'):
        assert text in ui, text
    assert ui.count("const SOURCE_REPO_URL =") == 1                         # one constant, not app config
    assert re.search(r'ECG/AFib<sup aria-hidden="true"[^>]*>↗</sup>', ui)   # screen readers skip the arrow
    api = (DEMO_ROOT / "app" / "src" / "ecg_demo_react" / "ui" / "lib" / "api.ts").read_text()
    for generated in ("export const warmup = async", "daily_job_id: string | null;", "serving_timeout_s: number;",
                      "export interface ServingError {", "export interface RateLimited {", "notice?: string;"):
        assert generated in api, generated                                 # the TS client was regenerated


def test_deploy_script_is_valid_bash_and_reads_endpoint_names_from_config():
    script = (DEMO_ROOT / "deploy.sh").read_text()
    subprocess.run(["bash", "-n", str(DEMO_ROOT / "deploy.sh")], check=True)
    assert script.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in script
    assert "config.endpoint_names(sys.argv[1])" in script and '"$ENDPOINT_PREFIX"' in script
    assert '--var "endpoint_prefix=$ENDPOINT_PREFIX"' in script
    assert not set(config.endpoint_names()) & set(re.findall(r"\w+", script))   # (the help text shows the default prefix)
    # --skip-embed-allowlist makes no settings call: the skip branch comes before the first one.
    step8 = script[script.index('step "8/9'):]
    assert step8.index("SKIP_ALLOWLIST -eq 1") < step8.index("databricks settings")
    # Step 1 reads the embedding settings (unless skipped), so a caller who cannot stops before
    # anything is built or deployed; step 8 writes back with the etag it read (optimistic concurrency).
    step1 = script[:script.index('step "2/9')]
    assert step1.index("SKIP_ALLOWLIST -eq 0") < step1.index("aibi-dashboard-embedding-access-policy get")
    assert step1.index("SKIP_ALLOWLIST -eq 0") < step1.index("read_approved_domains >/dev/null")
    assert "--arg etag" in step8 and "{etag: $etag}" in step8 and "attempt" in step8
    assert '--var "embed_credentials=$EMBED_CREDENTIALS"' in script
    assert '--var "mlflow_experiment=$MLFLOW_EXPERIMENT"' in script
    for step in ("bundle validate --strict", "bundle plan", "bundle deploy", "bundle run ecg12_afib_setup",
                 "serving-endpoints update-permissions", "CAN_QUERY", "bundle run ecg12_afib_app",
                 "aibi-dashboard-embedding-approved-domains"):
        assert step in script, step
    # Step 3: validate (its resolved configuration), then every value the script uses read from it
    # and checked with the notebooks' own checks, then the plan, all with the same arguments and
    # before the deploy. A failed validate or plan prints its hint, then the step's ERR summary.
    step3 = script[script.index('step "3/9'):script.index('step "4/9')]
    validate = step3.index('resolved=$(databricks bundle validate --strict -o json "${BUNDLE[@]}")')
    reads = [step3.index(f"{name}=$(resolved_var {var})") for name, var in RESOLVED_VARS.items()]
    checks = [step3.index(check) for check in ("config.endpoint_names(sys.argv[1])", "config.uc_location(sys.argv[1], sys.argv[2])",
                                               "config.resolve_mlflow_experiment(sys.argv[1])")]
    plan = step3.index('databricks bundle plan "${BUNDLE[@]}"')
    assert validate < min(reads) and max(reads) < min(checks) and max(checks) < plan
    assert '"$CATALOG" "$SCHEMA"' in step3 and '"$MLFLOW_EXPERIMENT" 2>&1' in step3
    assert step3.index("|| hint", plan) < step3.index(PLAN_HINT, plan)
    assert "hint() { echo \"ERROR: $*\" >&2; return 1; }" in step3         # no exit: the ERR trap reports the step
    # Each check and lookup runs once, on the resolved values: none before step 3, where only the
    # flags exist (a value from BUNDLE_VAR_<name> or the overrides file would be missed).
    for check in ("config.endpoint_names(", "config.uc_location(", "config.resolve_mlflow_experiment("):
        assert script.count(check) == 1 and check in step3, check
    assert script.index('step "4/9') < script.index('apps get "$APP_NAME"')


# The values deploy.sh reads back from `bundle validate -o json` (shell variable -> bundle variable).
RESOLVED_VARS = {"CATALOG": "catalog", "SCHEMA": "schema", "WAREHOUSE_ID": "warehouse_id", "ENDPOINT_PREFIX": "endpoint_prefix",
                 "MLFLOW_EXPERIMENT": "mlflow_experiment", "APP_NAME": "app_name",
                 "EMBED_CREDENTIALS": "embed_credentials", "REMOVE_AFTER": "remove_after"}
# Printed by deploy.sh when `databricks bundle plan` fails (for example "lineage mismatch in state
# files": the folder's deployment state belongs to another workspace).
PLAN_HINT = ("If you deployed this folder to another workspace before, use a fresh clone or delete "
             ".databricks/bundle/full/")
# A stand-in for the Databricks CLI, so deploy.sh runs end to end without a workspace. It logs every
# call, and resolves the bundle variables as the CLI does: --var, then BUNDLE_VAR_<name>, then the
# variable-overrides file (FAKE_OVERRIDES, the file's JSON), then databricks.yml's defaults
# (FAKE_DEFAULTS); `bundle validate` fails on a required variable with no value, and writes
# FAKE_VALIDATE_WARNING to stderr when set. `bundle plan` fails when PLAN_FAILS=1; `apps get` /
# `serving-endpoints get` answer for any name; `bundle summary` names the deployed dashboard
# FAKE_DASHBOARD_ID (none when it is empty) and fails when SUMMARY_FAILS=1; `lakeview publish`
# fails when PUBLISH_FAILS=1.
STUB_DATABRICKS = r"""#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ["DATABRICKS_LOG"], "a") as log:
    log.write(" ".join(args) + "\n")
def out(value):
    print(json.dumps(value))
if args == ["--version"]:
    print("Databricks CLI v0.281.0")
elif args[:2] == ["current-user", "me"]:
    out({"userName": "someone@example.org"})
elif args[:2] == ["bundle", "validate"]:
    if os.environ.get("FAKE_VALIDATE_WARNING"):
        print(os.environ["FAKE_VALIDATE_WARNING"], file=sys.stderr)
    values = {k: v for k, v in json.loads(os.environ["FAKE_DEFAULTS"]).items() if v is not None}
    values.update(json.loads(os.environ.get("FAKE_OVERRIDES") or "{}"))
    values.update({k[len("BUNDLE_VAR_"):]: v for k, v in os.environ.items() if k.startswith("BUNDLE_VAR_")})
    values.update(dict(a.split("=", 1) for a, flag in zip(args[1:], args) if flag == "--var"))
    for name in json.loads(os.environ["FAKE_DEFAULTS"]):
        if name not in values:
            sys.exit(f"Error: no value assigned to required variable {name}.")
    values = {k: v.replace("${workspace.current_user.userName}", "someone@example.org") if isinstance(v, str) else v
              for k, v in values.items()}
    out({"variables": {k: {"value": v} for k, v in values.items()}}) if "json" in args else print("Validation OK!")
elif args[:2] == ["bundle", "plan"]:
    if os.environ.get("PLAN_FAILS") == "1":
        sys.exit("Error: lineage mismatch in state files")
    print("Plan: 0 to add, 0 to change, 0 to delete, 4 unchanged")
elif args[:2] in (["bundle", "deploy"], ["bundle", "run"]) or args[:2] == ["serving-endpoints", "update-permissions"]:
    pass
elif args[:2] == ["bundle", "summary"]:
    if os.environ.get("SUMMARY_FAILS") == "1":
        sys.exit("Error: failed to load the deployment state")
    dashboard_id = os.environ.get("FAKE_DASHBOARD_ID", "dash-1")
    out({"resources": {"dashboards": {"ecg12_afib_dashboard": {"id": dashboard_id} if dashboard_id else {}}}})
elif args[:2] == ["lakeview", "publish"]:
    if os.environ.get("PUBLISH_FAILS") == "1":
        sys.exit("Error: PERMISSION_DENIED: cannot publish")
    out({"display_name": "dashboard"})
elif args[:2] == ["apps", "get"]:
    out({"service_principal_client_id": "sp-1", "url": f"https://{args[2]}.example.org"})
elif args[:2] == ["serving-endpoints", "get"]:
    out({"id": f"id-{args[2]}"})
else:
    sys.exit(f"stand-in: unexpected call: {' '.join(args)}")
"""
STUB_APX = '#!/usr/bin/env bash\n[[ "$1" == bun ]] && echo 1.0.0 || echo "apx 0.3.8"\n'
STUB_UV = '#!/usr/bin/env bash\necho "uv 0.0.0"\n'


def _run_deploy_script(tmp_path, *flags, overrides=None, plan_fails=False, env=None):
    """deploy.sh on a copy of itself and common/, with stand-ins for the CLI, apx, uv and the app
    build; --skip-embed-allowlist, so it makes no settings call. `env` adds environment variables
    (BUNDLE_VAR_<name>, FAKE_VALIDATE_WARNING). Returns (result, CLI calls)."""
    root, bin_dir = tmp_path / "project", tmp_path / "bin"
    shutil.copytree(DEMO_ROOT / "common", root / "common", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(DEMO_ROOT / "deploy.sh", root / "deploy.sh")
    for path, text in ((bin_dir / "databricks", STUB_DATABRICKS), (bin_dir / "apx", STUB_APX),
                       (bin_dir / "uv", STUB_UV), (root / "app" / "build_from_lockfiles.sh", "#!/usr/bin/env bash\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        path.chmod(0o755)
    log = tmp_path / "databricks.log"
    log.write_text("")
    defaults = {name: variable.get("default") for name, variable in _bundle()["variables"].items()}
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin:{Path(shutil.which('jq')).parent}:{Path(sys.executable).parent}",
           "HOME": str(tmp_path), "DATABRICKS_LOG": str(log), "PLAN_FAILS": "1" if plan_fails else "0",
           "FAKE_DEFAULTS": json.dumps(defaults), "FAKE_OVERRIDES": json.dumps(overrides or {}), **(env or {})}
    result = subprocess.run(["bash", str(root / "deploy.sh"), "-p", "prof", "--skip-embed-allowlist", *flags],
                            capture_output=True, text=True, env=env, timeout=60, check=False)
    return result, log.read_text().splitlines()


@pytest.mark.skipif(shutil.which("jq") is None, reason="deploy.sh needs jq")
@pytest.mark.parametrize("plan_fails", [True, False])
def test_deploy_script_stops_on_a_failed_plan(tmp_path, plan_fails):
    """deploy.sh runs `bundle plan` with the validate / deploy arguments right after validate: a failed
    plan prints the hint, then the step's usual "failed during" summary, and never deploys; a plan
    that passes goes on to the deploy and the rest of the install."""
    result, calls = _run_deploy_script(tmp_path, "--catalog", "cat", "--warehouse-id", "wh", plan_fails=plan_fails)
    args = "-t full -p prof --var catalog=cat --var warehouse_id=wh"
    assert calls.index(f"bundle validate --strict -o json {args}") + 1 == calls.index(f"bundle plan {args}"), result.stderr
    if plan_fails:
        assert result.returncode == 1
        assert "lineage mismatch in state files" in result.stderr
        assert "ERROR: databricks bundle plan failed" in result.stderr and PLAN_HINT in result.stderr
        assert "ERROR: deploy.sh failed during: 3/9 bundle validate and plan" in result.stderr
        assert result.stderr.index(PLAN_HINT) < result.stderr.index("deploy.sh failed during")
        assert not [c for c in calls if c.startswith("bundle deploy")]
    else:
        assert result.returncode == 0, result.stderr
        assert "Plan: 0 to add" in result.stdout and PLAN_HINT not in result.stderr
        assert calls.index(f"bundle plan {args}") + 1 == calls.index(f"bundle deploy {args}")
        assert "Done. App: https://ecg12-demo.example.org" in result.stdout


@pytest.mark.skipif(shutil.which("jq") is None, reason="deploy.sh needs jq")
def test_deploy_script_uses_the_bundles_resolved_values(tmp_path):
    """A value only in .databricks/bundle/full/variable-overrides.json is the one the script's own
    checks and lookups use (the app it reads, the endpoints it grants), as for the bundle; a
    BUNDLE_VAR_<name> wins over the file, and a flag (passed as --var) over both."""
    overrides = {"catalog": "file_cat", "warehouse_id": "file_wh", "app_name": "file-app", "endpoint_prefix": "file_pfx"}
    result, calls = _run_deploy_script(tmp_path / "file", overrides=overrides)
    assert result.returncode == 0, result.stderr
    assert "bundle validate --strict -o json -t full -p prof" in calls           # no --var: all from the file
    assert "apps get file-app -p prof -o json" in calls and "Done. App: https://file-app.example.org" in result.stdout
    assert {"serving-endpoints get file_pfx_arr -p prof -o json", "serving-endpoints get file_pfx_afib -p prof -o json"} <= set(calls)
    assert "catalog.schema: file_cat.biosignals" in result.stdout

    result, calls = _run_deploy_script(tmp_path / "flags", "--app-name", "flag-app", "--endpoint-prefix", "flag_pfx",
                                       overrides=overrides)
    assert result.returncode == 0, result.stderr
    assert "apps get flag-app -p prof -o json" in calls and not [c for c in calls if "file-app" in c]
    assert "serving-endpoints get flag_pfx_arr -p prof -o json" in calls and not [c for c in calls if "file_pfx" in c]

    result, calls = _run_deploy_script(tmp_path / "env", overrides=overrides, env={"BUNDLE_VAR_app_name": "env-app"})
    assert result.returncode == 0, result.stderr
    assert "apps get env-app -p prof -o json" in calls and not [c for c in calls if "file-app" in c]
    assert "serving-endpoints get file_pfx_arr -p prof -o json" in calls          # no BUNDLE_VAR: still the file's

    result, calls = _run_deploy_script(tmp_path / "env-and-flag", "--app-name", "flag-app", overrides=overrides,
                                       env={"BUNDLE_VAR_app_name": "env-app"})
    assert result.returncode == 0, result.stderr
    assert "apps get flag-app -p prof -o json" in calls and not [c for c in calls if "env-app" in c]


@pytest.mark.skipif(shutil.which("jq") is None, reason="deploy.sh needs jq")
def test_deploy_script_keeps_a_validate_warning_visible_and_out_of_the_values(tmp_path):
    """A warning `bundle validate` writes to stderr reaches the terminal, and the values the script
    reads come from the JSON on stdout, unaffected by it."""
    warning = "Warning: a made-up validate warning on stderr"
    result, calls = _run_deploy_script(tmp_path, "--catalog", "cat", "--warehouse-id", "wh", "--app-name", "warned-app",
                                       env={"FAKE_VALIDATE_WARNING": warning})
    assert result.returncode == 0, result.stderr
    assert warning in result.stderr and warning not in result.stdout
    assert "catalog.schema: cat.biosignals; app: warned-app" in result.stdout
    assert "apps get warned-app -p prof -o json" in calls


@pytest.mark.skipif(shutil.which("jq") is None, reason="deploy.sh needs jq")
@pytest.mark.parametrize("embed", [True, False], ids=["embed-credentials", "viewer-credentials"])
def test_deploy_script_republishes_the_dashboard_after_daily(tmp_path, embed):
    """With --run-daily, the last step republishes the bundle's dashboard once, after the DAILY run
    and after the app's steps, so its Ask Genie sees the tables SETUP and DAILY created: the id from
    `bundle summary`, the resolved warehouse (here only in the overrides file) and --embed-credentials
    exactly when embed_credentials is true."""
    overrides = {"catalog": "cat", "warehouse_id": "file_wh", "embed_credentials": embed}
    result, calls = _run_deploy_script(tmp_path, "--run-daily", overrides=overrides, env={"FAKE_DASHBOARD_ID": "dash-9"})
    assert result.returncode == 0, result.stderr
    publishes = [c for c in calls if c.startswith("lakeview publish")]
    expected = "lakeview publish dash-9 --warehouse-id file_wh" + (" --embed-credentials" if embed else "") + " -p prof"
    assert publishes == [expected], publishes
    daily = calls.index("bundle run ecg12_afib_daily -t full -p prof")
    assert daily < calls.index("bundle summary -o json -t full -p prof") < calls.index(expected)
    assert calls.index("apps get ecg12-demo -p prof -o json") < calls.index(expected)
    assert "republished dashboard dash-9, so its Ask Genie sees the tables SETUP and DAILY created" in result.stdout
    assert "republish once" not in result.stdout          # nothing to repeat after it


@pytest.mark.skipif(shutil.which("jq") is None, reason="deploy.sh needs jq")
def test_deploy_script_republish_failure_only_warns(tmp_path):
    """A failed republish warns with the command to run by hand and the script still finishes."""
    result, calls = _run_deploy_script(tmp_path, "--catalog", "cat", "--warehouse-id", "wh", "--run-daily",
                                       env={"PUBLISH_FAILS": "1", "FAKE_DASHBOARD_ID": "dash-9"})
    assert result.returncode == 0, result.stderr
    assert len([c for c in calls if c.startswith("lakeview publish")]) == 1
    assert "PERMISSION_DENIED: cannot publish" in result.stderr
    assert "WARNING: republishing the dashboard failed" in result.stderr
    assert "  databricks lakeview publish dash-9 --warehouse-id wh --embed-credentials -p prof" in result.stderr
    assert "deploy.sh failed during" not in result.stderr and "Done. App: " in result.stdout


@pytest.mark.skipif(shutil.which("jq") is None, reason="deploy.sh needs jq")
@pytest.mark.parametrize("env", [{"SUMMARY_FAILS": "1"}, {"FAKE_DASHBOARD_ID": ""}], ids=["summary-fails", "no-dashboard-id"])
def test_deploy_script_without_a_dashboard_id_warns_with_the_manual_command(tmp_path, env):
    """`bundle summary` failing, or naming no dashboard id: no publish call, a warning with the manual
    command (a placeholder id and how to get it), and the script still finishes."""
    result, calls = _run_deploy_script(tmp_path, "--catalog", "cat", "--warehouse-id", "wh", "--run-daily", env=env)
    assert result.returncode == 0, result.stderr
    assert not [c for c in calls if c.startswith("lakeview publish")]
    assert "WARNING: could not read the dashboard's id from the bundle, so it was not republished" in result.stderr
    assert "  databricks lakeview publish <dashboard-id> --warehouse-id wh --embed-credentials -p prof" in result.stderr
    assert "(<dashboard-id>: databricks bundle summary -t full -p prof -o json | jq -r '.resources.dashboards." in result.stderr
    assert "deploy.sh failed during" not in result.stderr and "Done. App: " in result.stdout


@pytest.mark.skipif(shutil.which("jq") is None, reason="deploy.sh needs jq")
def test_deploy_script_without_daily_prints_the_republish_command(tmp_path):
    """Without --run-daily the DAILY job's tables may not exist yet: no publish call, and the command
    to run once DAILY has run is printed."""
    result, calls = _run_deploy_script(tmp_path, "--catalog", "cat", "--warehouse-id", "wh",
                                       env={"FAKE_DASHBOARD_ID": "dash-9"})
    assert result.returncode == 0, result.stderr
    assert not [c for c in calls if c.startswith("lakeview publish")]
    assert "Once DAILY has run (now, if it ran before), republish once so Ask Genie works:" in result.stdout
    assert "  databricks lakeview publish dash-9 --warehouse-id wh --embed-credentials -p prof" in result.stdout


@pytest.mark.skipif(shutil.which("jq") is None, reason="deploy.sh needs jq")
def test_deploy_script_checks_the_resolved_values_before_deploying(tmp_path):
    """No catalog anywhere: validate fails, with the hint and the step's summary. A schema from the
    overrides file that the jobs would refuse stops the script before the plan and the deploy."""
    result, calls = _run_deploy_script(tmp_path / "none", "--warehouse-id", "wh")
    assert result.returncode == 1
    assert "no value assigned to required variable catalog" in result.stderr
    assert "--catalog / --warehouse-id, BUNDLE_VAR_catalog" in result.stderr
    assert "ERROR: deploy.sh failed during: 3/9 bundle validate and plan" in result.stderr
    result, calls = _run_deploy_script(tmp_path / "bad", overrides={"catalog": "c", "warehouse_id": "w", "schema": "my-schema"})
    assert result.returncode == 1 and "invalid catalog / schema" in result.stderr, result.stderr
    assert not [c for c in calls if c.startswith(("bundle plan", "bundle deploy"))]


def test_remove_after_tags_the_jobs_with_no_date_committed():
    """remove_after (empty by default: no date ships) is the removeAfter tag on every taggable bundle
    resource, set once in presets.tags; deploy.sh --remove-after forwards a YYYY-MM-DD date to it.
    The SDK-created serving endpoints get it from the SETUP job: 05_ checks the date's format before
    changing any endpoint, tags both endpoints only when it is set, and never deletes the tag."""
    bundle = _bundle()
    variable = bundle["variables"]["remove_after"]
    assert variable["default"] == "" and "type" not in variable          # a plain string: --var / BUNDLE_VAR_
    assert "YYYY-MM-DD" in variable["description"]
    assert bundle["presets"] == {"tags": {"removeAfter": "${var.remove_after}"}}
    assert all("presets" not in target for target in bundle["targets"].values())
    for path, job in _jobs().values():
        assert "tags" not in job, path.name                     # the jobs take presets.tags as they are
    setup, daily = _jobs()["ecg12_afib_setup"][1], _jobs()["ecg12_afib_daily"][1]
    assert {p["name"]: p["default"] for p in setup["parameters"]}["remove_after"] == "${var.remove_after}"
    assert "remove_after" not in {p["name"] for p in daily["parameters"]}      # DAILY creates no endpoint
    serve = "\n".join("" if line.startswith("# MAGIC") else line
                      for line in (DEMO_ROOT / "workflow" / "05_serve_endpoints.py").read_text().splitlines())
    assert 'dbutils.widgets.text("remove_after", "", ' in serve               # optional: empty is valid
    # Read from the parameter, and only there: a hard-coded value can't silently disable the tagging.
    assert re.findall(r"^remove_after\s*=.*$", serve, re.MULTILINE) == ['remove_after = dbutils.widgets.get("remove_after").strip()']
    check = 'if remove_after and not re.fullmatch(r"\\d{4}-\\d{2}-\\d{2}", remove_after):'
    assert check in serve and serve.index(check) < serve.index("deployed[endpoint_name] = deploy(model_name, endpoint_name, existing)")
    tag = 'w.serving_endpoints.patch(name=endpoint_name, add_tags=[EndpointTag(key="removeAfter", value=remove_after)])'
    assert serve.count(tag) == 1 and "delete_tags" not in serve
    tagging = serve[serve.index("\nif remove_after:\n"):]
    assert tagging.index("for _, endpoint_name in SERVED:") < tagging.index(tag) < tagging.index("\nelse:\n")
    assert serve.index("w.serving_endpoints.put_ai_gateway(") < serve.index(tag)     # after the endpoints exist
    # No committed file sets the tag to a date (the docs show a placeholder).
    dated = re.compile(r"remove[_-]?after\W{0,4}\d{4}-\d{2}-\d{2}", re.IGNORECASE)
    for path in _project_files():
        assert not dated.search(path.read_text()), path.relative_to(DEMO_ROOT)

    script = (DEMO_ROOT / "deploy.sh").read_text()
    assert "--remove-after YYYY-MM-DD" in script and '--var "remove_after=$REMOVE_AFTER"' in script
    for value in ("2030-1-31", "20300131", "2030-01-31x", "31-01-2030"):
        # Fails in the argument check, before any command or CLI call.
        result = subprocess.run(["bash", str(DEMO_ROOT / "deploy.sh"), "-p", "unused", "--catalog", "c",
                                 "--warehouse-id", "w", "--remove-after", value],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 2, value
        assert f"--remove-after takes a date YYYY-MM-DD, not '{value}'" in result.stderr, value


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
    # SETUP trains + gates on serverless GPU with the AI v6 environment; every other task, DAILY's
    # driver-side scoring included, runs on standard serverless environment 6, no GPU. Each is named,
    # not the workspace default (which moves to new releases).
    assert {e["environment_key"]: e["spec"] for e in setup["environments"]} == {
        "aiv6": {"base_environment": "workspace-base-environments/databricks_ai_v6"},
        "standard_v6": {"environment_version": "6"}}
    assert {e["environment_key"]: e["spec"] for e in daily["environments"]} == {
        "standard_v6": {"environment_version": "6"}}
    # Only SETUP serves endpoints (05_), so only SETUP gets the endpoint prefix.
    assert {p["name"]: p["default"] for p in setup["parameters"]}["endpoint_prefix"] == "${var.endpoint_prefix}"
    assert "endpoint_prefix" not in {p["name"] for p in daily["parameters"]}
    # Only SETUP logs MLflow runs (02_, 03_, 04_), so only SETUP gets the experiment. It is a plain
    # workspace path, not a bundle-managed experiment, so `bundle destroy` leaves the run history.
    assert {p["name"]: p["default"] for p in setup["parameters"]}[config.MLFLOW_EXPERIMENT_PARAM] == (
        "${var.mlflow_experiment}")
    assert config.MLFLOW_EXPERIMENT_PARAM not in {p["name"] for p in daily["parameters"]}
    for path in [DEMO_ROOT / "databricks.yml", *sorted((DEMO_ROOT / "resources").glob("*.yml"))]:
        resources = (yaml.safe_load(path.read_text()) or {}).get("resources") or {}
        assert "experiments" not in resources, path.name
    params = {p["name"]: p["default"] for p in daily["parameters"]}
    assert params[config.RUN_DATE_PARAM] == "{{job.start_time.iso_date}}"
    # One schema per deployment: both jobs pass the same deployment identity, which 01_ records and
    # 01_ / 06_ (each job's first task) check.
    for job in (setup, daily):
        assert {p["name"]: p["default"] for p in job["parameters"]}[config.DEPLOYMENT_ID_PARAM] == (
            "{{workspace.id}}:${workspace.root_path}")
    for notebook, call in (("01_generate_training_data.py", "uc_io.claim_schema("),
                           ("06_generate_batch_data.py", "uc_io.check_schema_owner(")):
        code = (DEMO_ROOT / "workflow" / notebook).read_text()
        owner_check = code.index(call)
        assert code.index("config.resolve_deployment_id(") < owner_check, notebook
        assert owner_check < min(code.index(write) for write in ("overwrite_table(", "write_date_slice(")
                                 if write in code), notebook
    assert daily["schedule"]["pause_status"] == "PAUSED"

    def accelerators(job):
        return {t["task_key"]: (t.get("environment_key"), t.get("compute", {}).get("hardware_accelerator"))
                for t in job["tasks"]}

    gpu, standard = ("aiv6", "GPU_1xA10"), ("standard_v6", None)
    assert accelerators(daily) == {"simulate_incoming_batch": standard, "arr_triage": standard,
                                   "afib_classify": standard}
    assert accelerators(setup) == {
        "generate_training_data": standard, "train_arr_model": gpu, "train_afib_model": gpu,
        "validate_and_promote": gpu, "serve_endpoints": standard}


def _task_environments():
    """(notebook file name, environment spec as one (key, value) pair) for every job task."""
    for _, job in _jobs().values():
        specs = {e["environment_key"]: e["spec"] for e in job["environments"]}
        for task in job["tasks"]:
            # A task without an environment gets the workspace default, which moves to new releases.
            assert "environment_key" in task, task["task_key"]
            (spec,) = specs[task["environment_key"]].items()
            yield Path(task["notebook_task"]["notebook_path"]).name, spec


def test_no_pip_line_downgrades_what_the_task_environment_ships():
    """Each %pip line only adds packages or moves them up from what the task's environment ships
    (conftest.PREINSTALLED, from the release notes): a lower databricks-sdk or protobuf than the
    environment's stops the Python kernel from restarting (05_ on environment 6 with databricks-sdk
    0.102.0 / protobuf 5.29.6 did). Every package a line names has a recorded version there."""
    checked = set()
    for notebook, spec in _task_environments():
        assert spec in PREINSTALLED, f"{notebook}: record what {spec} ships in conftest.PREINSTALLED"
        for line in pip_lines(DEMO_ROOT / "workflow" / notebook):
            for package, version in pip_pins(line):
                assert package in PREINSTALLED[spec], f"{notebook}: record {package} for {spec}"
                shipped = PREINSTALLED[spec][package]
                assert shipped is None or Version(version) >= Version(shipped), (
                    f"{notebook}: {package}=={version} is below the {shipped} that {spec} ships")
                checked.add(notebook)
    assert checked == {name for name, lines in PIP_LINES.items() if lines}


def _runs(notebook, spec, package):
    """The version of a package a notebook runs: its %pip line's pin, else what its environment ships."""
    pinned = dict(p for line in pip_lines(DEMO_ROOT / "workflow" / notebook) for p in pip_pins(line))
    return pinned.get(package, PREINSTALLED[spec][package])


def test_every_notebook_using_mlflow_runs_the_models_mlflow_version():
    """The training notebooks log with exactly the mlflow version the models' requirements name;
    the gate, serving and batch notebooks run that version or a newer one. Each gets it from its
    %pip line or, with none naming mlflow-skinny, from what its environment ships (the training
    notebooks too, when their environment ships exactly the models' version)."""
    (model_mlflow,) = [r.removeprefix("mlflow==") for r in config.MODEL_PIP_REQUIREMENTS if r.startswith("mlflow==")]
    loggers = {"02_train_arr_model.py", "03_train_afib_model.py"}
    users = loggers | {"04_validate_and_promote.py", "05_serve_endpoints.py", "07_arr_triage.py", "08_afib_classify.py"}
    seen = set()
    for notebook, spec in _task_environments():
        if notebook not in users:
            continue
        runs = _runs(notebook, spec, "mlflow-skinny")
        if notebook in loggers:
            assert runs == model_mlflow, f"{notebook} logs with mlflow-skinny {runs}, the models name {model_mlflow}"
        assert Version(runs) >= Version(model_mlflow), f"{notebook} runs mlflow-skinny {runs}"
        seen.add(notebook)
    assert seen == users


def test_every_notebook_running_the_models_runs_their_torch_version():
    """The training notebooks pickle the models with, and the gate and batch notebooks load them
    with, exactly the torch version the models' requirements name (a CUDA build of it counts, as
    in pip: torch==2.11.0 is met by 2.11.0+cu130). 05_ only calls the endpoints."""
    (model_torch,) = [r.removeprefix("torch==") for r in config.MODEL_PIP_REQUIREMENTS if r.startswith("torch==")]
    users = {"02_train_arr_model.py", "03_train_afib_model.py", "04_validate_and_promote.py",
             "07_arr_triage.py", "08_afib_classify.py"}
    seen = set()
    for notebook, spec in _task_environments():
        if notebook in users:
            runs = _runs(notebook, spec, "torch")
            assert runs is not None and Version(runs).public == model_torch, f"{notebook} runs torch {runs}"
            seen.add(notebook)
    assert seen == users


def test_installed_mlflow_skinny_accepts_the_environments_sdk_and_protobuf():
    """A %pip line that installs mlflow-skinny keeps the environment's databricks-sdk and protobuf
    only when that mlflow-skinny version accepts them (conftest.MLFLOW_SKINNY_REQUIRES, from its
    package metadata); otherwise pip would move them, which stops the Python kernel from restarting
    when it moves them down."""
    seen = set()
    for notebook, spec in _task_environments():
        for line in pip_lines(DEMO_ROOT / "workflow" / notebook):
            version = dict(pip_pins(line)).get("mlflow-skinny")
            if version is None:
                continue
            assert version in MLFLOW_SKINNY_REQUIRES, f"{notebook}: record what mlflow-skinny {version} requires"
            for package, allowed in MLFLOW_SKINNY_REQUIRES[version].items():
                runs = _runs(notebook, spec, package)
                assert runs in SpecifierSet(allowed), (
                    f"{notebook}: mlflow-skinny {version} requires {package}{allowed}, {spec} has {runs}")
            seen.add(notebook)
    assert seen == {"05_serve_endpoints.py", "07_arr_triage.py", "08_afib_classify.py"}


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
    # Not the digits of a decimal number (the 15 after "14." in a float, or before ".5"); an id
    # followed by a plain "." still counts, as in an app URL "<app>-<workspace id>.aws...".
    "workspace or dashboard id": re.compile(r"(?<!\d\.)\b(?:\d{15,16}|[0-9a-f]{32})\b(?!\.\d)"),
}


def test_no_workspace_specific_values_in_committed_files():
    """No host, personal path, email, internal mirror or workspace / dashboard id in any project file,
    lockfiles included (ids are not checked in lockfiles: their hashes look alike)."""
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
    result = subprocess.run([sys.executable, "-B", str(BUILDER), "--check"], capture_output=True, text=True,
                            check=False)
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
