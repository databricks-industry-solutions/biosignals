#!/usr/bin/env bash
# Full install of the 12-lead ECG/AFib demo into YOUR workspace: the bundle's `full` target (SETUP
# + DAILY jobs, AI/BI dashboard, Databricks App), then the steps a bundle deploy cannot do alone.
#
#   ./deploy.sh -p <profile> --catalog <catalog> --warehouse-id <id> [options]
#
# The bundle variables come from the flags (passed as --var, so they win), BUNDLE_VAR_<name>,
# .databricks/bundle/full/variable-overrides.json and databricks.yml's defaults, in that order.
# Every value the script itself uses is read back from the bundle's own resolution (step 3), so the
# script and the bundle never disagree about a value.
#
# Steps (each one fails loudly and stops the script):
#   1. prerequisites: apx 0.3.8 (with its bun), uv, Databricks CLI >= 0.286.0, jq, python3, a
#      profile that authenticates and whose workspace_id line, if it has one, is the workspace its
#      host connects to (else step 6's bundle summary fails), and (unless --skip-embed-allowlist)
#      that it can read the workspace's dashboard-embedding settings step 9 changes: a caller who
#      cannot learns it here, before anything is built or deployed
#   2. build the app from the lockfiles (app/build_from_lockfiles.sh)
#   3. databricks bundle validate --strict -t full (fails when catalog or warehouse_id has no
#      value), then the resolved values are read and checked (catalog and schema: letters, digits
#      and underscores; the warehouse id, the endpoint prefix, the MLflow experiment,
#      embed_credentials, remove_after), with the same check the flags get before step 1,
#      then databricks bundle plan -t full (prints what the deploy will add, change or delete; stops
#      with a hint when the folder's deployment state belongs to another workspace)
#   4. databricks bundle deploy -t full (creates the jobs, the dashboard and the app; grants the
#      viewer group CAN_READ on the dashboard and CAN_USE on the app, from databricks.yml)
#   5. run the SETUP job and wait for it: training data, both models, the validation gate, the two
#      serving endpoints (about 30-60 minutes: GPU training, then endpoint creation)
#      (--run-daily also runs the DAILY job once, so the dashboard's batch tiles have data)
#   6. republish the dashboard (with --run-daily), so its Ask Genie sees the tables SETUP and DAILY
#      created: step 4 published it before they existed. It comes right after DAILY, so a failure
#      in the app's steps cannot skip it. Without --run-daily the DAILY job's tables may not exist
#      yet, so it prints the command to run once DAILY has run instead (again at the end). A
#      failed republish only warns, with the command to run by hand.
#   7. grant the app's service principal CAN_QUERY on both serving endpoints (they exist only
#      after step 5, so they cannot be app resources of the first deploy)
#   8. start the app: databricks bundle run ecg12_afib_app (deploys the built source with its
#      environment, including the dashboard id)
#   9. add the app's domain to the workspace's dashboard-embedding allowlist (workspace admin
#      setting; done after step 8 because the app's URL exists once the app is deployed). It reads
#      the current list with its etag and writes back that list plus the domain with the same etag,
#      so the write fails instead of dropping an entry another admin added in between; it then
#      reads again and retries, and finally checks the domain is there. Caveat: a setting never
#      written has no etag, so two first-ever writes at the same moment can still overwrite each
#      other (the later one wins). --skip-embed-allowlist makes no settings call at all and prints
#      what an admin must add.
#
# Re-running is safe: every step updates in place (SETUP retrains and re-gates the models).
# Remove it as docs/deploy.md, Teardown, describes: check the schema's owner first; `databricks
# bundle destroy -t full ...` removes only the jobs, the dashboard and the app.
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  cat <<'USAGE'
Usage: ./deploy.sh -p PROFILE [--catalog CATALOG] [--warehouse-id ID] [options]

Required flag:
  -p, --profile PROFILE       Databricks CLI profile of the target workspace (~/.databrickscfg)

Required values (from a flag, BUNDLE_VAR_<name> or .databricks/bundle/full/variable-overrides.json;
a flag wins, then BUNDLE_VAR_<name>, then the file):
  --catalog CATALOG           Unity Catalog catalog (the SETUP job creates the schema in it)
  --warehouse-id ID           SQL warehouse for the AI/BI dashboard

Options (bundle variables too, read the same way; otherwise databricks.yml's default):
  --schema SCHEMA             Unity Catalog schema (default: biosignals); one schema per
                              deployment: the SETUP job stops if another deployment owns it
  --endpoint-prefix PREFIX    serving endpoints PREFIX_arr and PREFIX_afib (default: bio_ecg12);
                              endpoint names are workspace-wide, so use a value no other
                              deployment in the workspace uses
  --mlflow-experiment PATH    MLflow experiment (absolute workspace path) of the SETUP job's runs
                              (default: /Users/<you>/biosignals-ecg12-afib); outlives bundle destroy
  --app-name NAME             app name (default: ecg12-demo; app names are workspace-wide)
  --viewer-group GROUP        group that may view the dashboard and use the app (default: users)
  --embed-credentials BOOL    true (default, synthetic data only): dashboard viewers query with the
                              deployer's credentials; false: with their own grants (real data)
  --remove-after YYYY-MM-DD   tag both jobs and both serving endpoints removeAfter=YYYY-MM-DD, for a
                              workspace whose cleanup automation deletes resources after that date
                              (default: no date)
  --run-daily                 also run the DAILY job once after SETUP, then republish the dashboard
                              so its Ask Genie sees the tables SETUP and DAILY created
  --skip-embed-allowlist      do not change the dashboard-embedding allowlist (not a workspace
                              admin); the script prints the domain an admin must add
  -h, --help                  show this help
USAGE
}

PROFILE="" CATALOG="" WAREHOUSE_ID="" SCHEMA="" ENDPOINT_PREFIX="" MLFLOW_EXPERIMENT="" APP_NAME="" VIEWER_GROUP=""
EMBED_CREDENTIALS="" REMOVE_AFTER="" RUN_DAILY=0 SKIP_ALLOWLIST=0
# A flag that takes a value takes the next argument, which must be there, not empty and not another
# flag: otherwise the script stops here, instead of taking the next flag as the value.
need_value() {
  [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { echo "ERROR: $1 needs a value" >&2; usage >&2; exit 2; }
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--profile) need_value "$@"; PROFILE="$2"; shift 2 ;;
    --catalog) need_value "$@"; CATALOG="$2"; shift 2 ;;
    --warehouse-id) need_value "$@"; WAREHOUSE_ID="$2"; shift 2 ;;
    --schema) need_value "$@"; SCHEMA="$2"; shift 2 ;;
    --endpoint-prefix) need_value "$@"; ENDPOINT_PREFIX="$2"; shift 2 ;;
    --mlflow-experiment) need_value "$@"; MLFLOW_EXPERIMENT="$2"; shift 2 ;;
    --app-name) need_value "$@"; APP_NAME="$2"; shift 2 ;;
    --viewer-group) need_value "$@"; VIEWER_GROUP="$2"; shift 2 ;;
    --embed-credentials) need_value "$@"; EMBED_CREDENTIALS="$2"; shift 2 ;;
    --remove-after) need_value "$@"; REMOVE_AFTER="$2"; shift 2 ;;
    --run-daily) RUN_DAILY=1; shift ;;
    --skip-embed-allowlist) SKIP_ALLOWLIST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

STEP="argument check"
trap 'echo "ERROR: deploy.sh failed during: $STEP" >&2' ERR
step() { STEP="$1"; echo; echo "==> $1"; }
fail() { echo "ERROR: $*" >&2; exit 1; }

if [[ -z "$PROFILE" ]]; then
  echo "ERROR: missing required argument: -p/--profile" >&2
  usage >&2
  exit 2
fi
# The one check of the values the script and the jobs use, as NAME=VALUE arguments: catalog and
# schema are plain identifiers and the endpoint prefix and MLflow experiment are what the notebooks
# accept (common/config.py, the notebooks' own checks, which import only the standard library); the
# warehouse id is letters, digits, '-' and '_' (so a name with spaces or a URL stops here),
# embed_credentials true or false, and remove_after empty or a date YYYY-MM-DD. Prints "ERROR: invalid <name>: ..." for each bad value and fails. With --flags
# first, an empty value means "not given as a flag" and is not checked. It runs twice: here on the
# flags, before any command (so a bad value stops the script before the app is built), and in step
# 3 on the values the bundle resolved (from a flag, BUNDLE_VAR_<name>, the overrides file or a default).
check_values() {
  python3 - "$@" <<'PY'
import re
import sys

sys.path.insert(0, ".")
from common import config


def one_of(*allowed):
    def check(name, value):
        if value not in allowed:
            raise ValueError(f"{name}={value!r}: takes {' or '.join(allowed)}")
    return check


def matches(pattern, what):
    def check(name, value):
        if not re.fullmatch(pattern, value):
            raise ValueError(f"{name}={value!r}: takes {what}")
    return check


def identifier(name, value):
    config.check_identifier(value, name)


CHECKS = {
    "catalog": identifier,
    "schema": identifier,
    "endpoint_prefix": lambda name, value: config.endpoint_names(value),
    "mlflow_experiment": lambda name, value: config.resolve_mlflow_experiment(value),
    "warehouse_id": matches(r"[A-Za-z0-9_-]+", "a SQL warehouse id (letters, digits, '-' and '_'; the id, not the name)"),
    "embed_credentials": one_of("true", "false"),
    "remove_after": matches(r"([0-9]{4}-[0-9]{2}-[0-9]{2})?", "a date YYYY-MM-DD, or nothing"),
}
args = sys.argv[1:]
flags_only = args[:1] == ["--flags"]
errors = []
for name, value in (arg.split("=", 1) for arg in args[flags_only:]):
    if flags_only and not value:
        continue
    try:
        CHECKS[name](name, value)
    except ValueError as error:
        errors.append(f"ERROR: invalid {name}: {error}")
sys.exit("\n".join(errors) or None)
PY
}
command -v python3 >/dev/null || { echo "ERROR: 'python3' is not on PATH (see docs/deploy.md, Path 2 prerequisites)." >&2; exit 2; }
check_values --flags "catalog=$CATALOG" "schema=$SCHEMA" "warehouse_id=$WAREHOUSE_ID" "endpoint_prefix=$ENDPOINT_PREFIX" \
  "mlflow_experiment=$MLFLOW_EXPERIMENT" "embed_credentials=$EMBED_CREDENTIALS" "remove_after=$REMOVE_AFTER" || exit 2

# Every bundle command gets the same target, profile and flags (as --var). A variable without a
# flag comes from BUNDLE_VAR_<name>, .databricks/bundle/full/variable-overrides.json or its default;
# step 3 reads the resolved values back, and the script uses only those.
BUNDLE=(-t full -p "$PROFILE")
[[ -z "$CATALOG" ]] || BUNDLE+=(--var "catalog=$CATALOG")
[[ -z "$WAREHOUSE_ID" ]] || BUNDLE+=(--var "warehouse_id=$WAREHOUSE_ID")
[[ -z "$SCHEMA" ]] || BUNDLE+=(--var "schema=$SCHEMA")
[[ -z "$ENDPOINT_PREFIX" ]] || BUNDLE+=(--var "endpoint_prefix=$ENDPOINT_PREFIX")
[[ -z "$MLFLOW_EXPERIMENT" ]] || BUNDLE+=(--var "mlflow_experiment=$MLFLOW_EXPERIMENT")
[[ -z "$APP_NAME" ]] || BUNDLE+=(--var "app_name=$APP_NAME")
[[ -z "$VIEWER_GROUP" ]] || BUNDLE+=(--var "viewer_group=$VIEWER_GROUP")
[[ -z "$EMBED_CREDENTIALS" ]] || BUNDLE+=(--var "embed_credentials=$EMBED_CREDENTIALS")
[[ -z "$REMOVE_AFTER" ]] || BUNDLE+=(--var "remove_after=$REMOVE_AFTER")

# ---------------------------------------------------------------------------------------------
step "1/9 prerequisites"
for cmd in apx uv databricks jq python3; do
  command -v "$cmd" >/dev/null || fail "'$cmd' is not on PATH (see docs/deploy.md, Path 2 prerequisites)."
done
apx_version=$(apx --version 2>/dev/null | grep -Eo 'apx [0-9]+\.[0-9]+\.[0-9]+[^[:space:]]*' | head -n 1 || true)
[[ "$apx_version" == "apx 0.3.8" ]] || fail "need apx 0.3.8 exactly, found '${apx_version:-none}'."
apx bun --version >/dev/null 2>&1 || fail "apx's bun does not run ('apx bun --version'); reinstall apx 0.3.8."
cli_version=$(databricks --version | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
# The oldest CLI with every command and flag used here and in the docs: `bundle validate --strict`
# is in 0.286.0 and later (dataset_catalog / dataset_schema on dashboard resources, `lakeview
# publish` sending embed_credentials false, and a public `bundle plan` are older).
python3 - "$cli_version" <<'PY' || fail "need Databricks CLI >= 0.286.0, found $cli_version."
import sys
sys.exit(0 if tuple(map(int, sys.argv[1].split("."))) >= (0, 286, 0) else 1)
PY
me=$(databricks current-user me -p "$PROFILE" -o json | jq -r '.userName') \
  || fail "profile '$PROFILE' does not authenticate: run 'databricks auth login -p $PROFILE'."
echo "apx 0.3.8, uv $(uv --version | cut -d ' ' -f 2), Databricks CLI $cli_version; deploying as $me"

# A workspace_id line in the profile that names another workspace than the one its host connects to
# breaks only `bundle summary` (step 6): it compares the two and stops ("workspace_id ... does not
# match the connected workspace"), while every other command here works. Run that same CLI check
# now, on an empty bundle in a temporary folder, so it needs no build or deployment and leaves this
# folder's deployment state alone. Another failure only warns: step 6 only warns too.
id_check_dir=$(mktemp -d)
printf 'bundle:\n  name: workspace-id-check\n' >"$id_check_dir/databricks.yml"
id_check_status=0
id_check=$(cd "$id_check_dir" && databricks bundle summary -p "$PROFILE" -o json 2>&1 >/dev/null) || id_check_status=$?
rm -rf "$id_check_dir"
if [[ $id_check_status -ne 0 ]]; then
  [[ "$id_check" != *"does not match the connected workspace"* ]] \
    || fail "profile '$PROFILE': its workspace_id does not match the workspace its host connects to. Delete the workspace_id line from [$PROFILE] in ~/.databrickscfg, or set it to the ID the CLI names, then rerun. The CLI says: ${id_check#Error: }"
  echo "WARNING: could not check the profile's workspace_id (databricks bundle summary on an empty bundle): $id_check. Going on; step 6 may then only print the republish command." >&2
fi

# The dashboard-embedding settings step 9 changes are workspace admin settings. Read them now, so a
# caller who cannot (not a workspace admin) or a workspace that blocks embedding stops the script
# here, before anything is built or deployed, not after every resource exists.
# The approved domains and the setting's etag, as {"etag": ..., "domains": [...]}; a setting never
# written yet reads as no domains and no etag. The CLI's stderr is kept apart, so a warning on stderr
# cannot corrupt the JSON reply.
read_approved_domains() {
  local err reply message
  err=$(mktemp)
  if ! reply=$(databricks settings aibi-dashboard-embedding-approved-domains get -p "$PROFILE" -o json 2>"$err"); then
    if grep -q -E 'RESOURCE_DOES_NOT_EXIST|NOT_FOUND' "$err"; then
      reply='{}'                                   # never set: no approved domains yet
    else
      message=$(cat "$err"); rm -f "$err"
      fail "cannot read the approved dashboard-embedding domains (needs a workspace admin; rerun with --skip-embed-allowlist, and the script prints the domain an admin must add): $message"
    fi
  fi
  rm -f "$err"
  jq -c '{etag: (.etag // ""), domains: (.aibi_dashboard_embedding_approved_domains.approved_domains // [])}' <<<"$reply" \
    || fail "unexpected reply when reading the approved embedding domains: $reply"
}
EMBED_POLICY=""
if [[ $SKIP_ALLOWLIST -eq 0 ]]; then
  policy_err=$(mktemp)
  EMBED_POLICY=$(databricks settings aibi-dashboard-embedding-access-policy get -p "$PROFILE" -o json 2>"$policy_err" \
    | jq -r '.aibi_dashboard_embedding_access_policy.access_policy_type // empty') || true
  policy_message=$(cat "$policy_err"); rm -f "$policy_err"
  [[ -n "$EMBED_POLICY" ]] || fail "cannot read the dashboard-embedding policy that step 9 needs (a workspace admin setting): ${policy_message:-empty reply}. Rerun with --skip-embed-allowlist; the script then prints the app domain a workspace admin must add."
  case "$EMBED_POLICY" in
    ALLOW_ALL_DOMAINS) ;;
    ALLOW_APPROVED_DOMAINS) read_approved_domains >/dev/null ;;     # the list step 9 updates is readable too
    DENY_ALL_DOMAINS)
      fail "dashboard embedding is disabled in this workspace (DENY_ALL_DOMAINS), so the app could not show the dashboard. A workspace admin must allow approved domains (Settings > Security > Embed dashboards), or rerun with --skip-embed-allowlist." ;;
    *)
      fail "unexpected dashboard-embedding policy '$EMBED_POLICY'." ;;
  esac
  echo "dashboard-embedding policy: $EMBED_POLICY (readable; step 9 adds the app's domain if needed)"
fi

# ---------------------------------------------------------------------------------------------
step "2/9 build the app from the lockfiles"
# Before step 3: for the full target, `bundle validate` and `bundle plan` both stop with
# "stat app/.build: no such file or directory" until the build has written app/.build.
app/build_from_lockfiles.sh

# ---------------------------------------------------------------------------------------------
step "3/9 bundle validate and plan"
# What to do about a failed bundle command (its own message is above), then the step's usual
# "deploy.sh failed during: <step>" from the ERR trap.
hint() { echo "ERROR: $*" >&2; return 1; }
resolved=$(databricks bundle validate --strict -o json "${BUNDLE[@]}") \
  || hint "databricks bundle validate failed (its message is above). If it names a required variable with no value (catalog or warehouse_id), give it as --catalog / --warehouse-id, BUNDLE_VAR_catalog / BUNDLE_VAR_warehouse_id, or in .databricks/bundle/full/variable-overrides.json."
echo "Validation OK"
# The values this script uses, as the bundle resolved them (a flag, BUNDLE_VAR_<name>,
# variable-overrides.json or the default): one source for the script's checks and lookups and
# for what the bundle deploys.
resolved_var() { jq -r --arg name "$1" '.variables[$name].value | if . == null then "" else tostring end' <<<"$resolved"; }
CATALOG=$(resolved_var catalog)
SCHEMA=$(resolved_var schema)
ENDPOINT_PREFIX=$(resolved_var endpoint_prefix)
MLFLOW_EXPERIMENT=$(resolved_var mlflow_experiment)
APP_NAME=$(resolved_var app_name)
EMBED_CREDENTIALS=$(resolved_var embed_credentials)
WAREHOUSE_ID=$(resolved_var warehouse_id)
REMOVE_AFTER=$(resolved_var remove_after)
echo "catalog.schema: $CATALOG.$SCHEMA; app: $APP_NAME; MLflow experiment: $MLFLOW_EXPERIMENT"
# The resolved values, with the same check as the flags above (check_values), so a value from
# BUNDLE_VAR_<name> or the overrides file that the jobs would refuse stops here, before anything
# is deployed.
check_values "catalog=$CATALOG" "schema=$SCHEMA" "warehouse_id=$WAREHOUSE_ID" "endpoint_prefix=$ENDPOINT_PREFIX" \
  "mlflow_experiment=$MLFLOW_EXPERIMENT" "embed_credentials=$EMBED_CREDENTIALS" "remove_after=$REMOVE_AFTER" || exit 1
# Endpoint names from the single source of truth, common/config.py: <prefix>_arr / <prefix>_afib.
endpoints=$(python3 -c 'import sys; sys.path.insert(0, "."); from common import config
print(*config.endpoint_names(sys.argv[1]))' "$ENDPOINT_PREFIX")
read -r ARR_ENDPOINT AFIB_ENDPOINT <<<"$endpoints"
echo "serving endpoints: $ARR_ENDPOINT, $AFIB_ENDPOINT (workspace-wide names; the SETUP job stops if another deployment owns them)"
# The plan shows what step 4 will change, before it changes anything. A folder deployed to another
# workspace before keeps that workspace's deployment state in .databricks/bundle/full/, and plan can
# stop on it (for example "lineage mismatch in state files").
databricks bundle plan "${BUNDLE[@]}" \
  || hint "databricks bundle plan failed (its message is above). If you deployed this folder to another workspace before, use a fresh clone or delete .databricks/bundle/full/ (move out a variable-overrides.json first if you keep one there)."

# ---------------------------------------------------------------------------------------------
step "4/9 bundle deploy (jobs, dashboard, app; viewer permissions)"
databricks bundle deploy "${BUNDLE[@]}"

# ---------------------------------------------------------------------------------------------
step "5/9 run the SETUP job (train, gate, promote, serve; waits until it finishes)"
databricks bundle run ecg12_afib_setup "${BUNDLE[@]}"
if [[ $RUN_DAILY -eq 1 ]]; then
  STEP="5/9 run the DAILY job once"
  echo "--> DAILY job (one simulated day, so the dashboard's batch tiles have data)"
  databricks bundle run ecg12_afib_daily "${BUNDLE[@]}"
fi

# ---------------------------------------------------------------------------------------------
step "6/9 republish the dashboard (so its Ask Genie sees the tables SETUP and DAILY created)"
# Step 4 published the dashboard before SETUP and DAILY created its tables. In our tests on AWS and
# Azure, its Ask Genie then said "Genie is unavailable due to incompatible SQL expressions in the
# dashboard" until the dashboard was published again after DAILY had run: republishing the unchanged
# draft fixed it, and `bundle plan` still showed the dashboard unchanged afterwards. Request-log
# tables that still had only their placeholder column did not stop Ask Genie. Same warehouse and
# credentials setting as the bundle's; the dashboard id comes from the bundle (`bundle summary`),
# not from state files. Nothing here fails the script: the dashboard itself works either way. It
# runs right after DAILY, before the app's steps, so a failure in one of those cannot skip it.
# The request body is given whole (--json), embed_credentials true or false included: the API's
# default for a missing embed_credentials is true, and CLI 0.280.0 sent no embed_credentials for
# --embed-credentials=false. The warehouse id is letters, digits, '-' and '_' (step 3), so the body
# needs no shell quoting beyond the single quotes of the printed command.
publish_body=$(jq -cn --arg wh "$WAREHOUSE_ID" --argjson embed "$EMBED_CREDENTIALS" '{embed_credentials: $embed, warehouse_id: $wh}')
publish_flags=(--json "$publish_body")
dashboard_id=""
if summary=$(databricks bundle summary -o json "${BUNDLE[@]}"); then
  dashboard_id=$(jq -r '.resources.dashboards.ecg12_afib_dashboard.id // empty' <<<"$summary" 2>/dev/null) || dashboard_id=""
fi
publish_cmd="databricks lakeview publish ${dashboard_id:-<dashboard-id>} --json '$publish_body' -p $(printf '%q' "$PROFILE")"
# The same bundle arguments as every bundle command here (--var included), so the command also
# works for a deployment whose values were given only as flags.
[[ -n "$dashboard_id" ]] || publish_cmd+="   (<dashboard-id>: databricks bundle summary -o json$(printf ' %q' "${BUNDLE[@]}") | jq -r '.resources.dashboards.ecg12_afib_dashboard.id')"
if [[ $RUN_DAILY -eq 0 ]]; then
  echo "not republished: without --run-daily, the DAILY job's tables may not exist yet. Once DAILY has run (now, if it ran before), republish once so Ask Genie works:"
  echo "  $publish_cmd"
elif [[ -z "$dashboard_id" ]]; then
  echo "WARNING: could not read the dashboard's id from the bundle, so it was not republished; Ask Genie can show 'Genie is unavailable due to incompatible SQL expressions in the dashboard'. Republish by hand:" >&2
  echo "  $publish_cmd" >&2
elif databricks lakeview publish "$dashboard_id" "${publish_flags[@]}" -p "$PROFILE" >/dev/null; then
  echo "republished dashboard $dashboard_id, so its Ask Genie sees the tables SETUP and DAILY created"
else
  echo "WARNING: republishing the dashboard failed (its message is above); Ask Genie can show 'Genie is unavailable due to incompatible SQL expressions in the dashboard'. Republish by hand:" >&2
  echo "  $publish_cmd" >&2
fi

# ---------------------------------------------------------------------------------------------
step "7/9 grant the app's service principal CAN_QUERY on both endpoints"
app_sp=$(databricks apps get "$APP_NAME" -p "$PROFILE" -o json | jq -r '.service_principal_client_id // empty')
[[ -n "$app_sp" ]] || fail "app '$APP_NAME' has no service principal yet (databricks apps get $APP_NAME)."
for endpoint in "$ARR_ENDPOINT" "$AFIB_ENDPOINT"; do
  # Permissions APIs take the endpoint's id, not its name.
  endpoint_id=$(databricks serving-endpoints get "$endpoint" -p "$PROFILE" -o json | jq -r '.id // empty')
  [[ -n "$endpoint_id" ]] || fail "serving endpoint '$endpoint' not found: did the SETUP job finish?"
  databricks serving-endpoints update-permissions "$endpoint_id" -p "$PROFILE" -o json --json "$(jq -n \
    --arg sp "$app_sp" '{access_control_list: [{service_principal_name: $sp, permission_level: "CAN_QUERY"}]}')" \
    >/dev/null
  echo "CAN_QUERY on $endpoint -> app service principal $app_sp"
done

# ---------------------------------------------------------------------------------------------
step "8/9 start the app (deploy its source and environment)"
databricks bundle run ecg12_afib_app "${BUNDLE[@]}"
app_url=$(databricks apps get "$APP_NAME" -p "$PROFILE" -o json | jq -r '.url // empty')
[[ -n "$app_url" ]] || fail "app '$APP_NAME' has no URL after its deployment."

# ---------------------------------------------------------------------------------------------
step "9/9 dashboard-embedding allowlist (workspace admin setting)"
app_domain=${app_url#*://}
app_domain=${app_domain%%/*}
admin_help="A workspace admin must allow embedding on '$app_domain': Settings > Security > Embed dashboards (approved domains)."
if [[ $SKIP_ALLOWLIST -eq 1 ]]; then
  # No settings API call at all: reading these settings needs a workspace admin too.
  echo "SKIPPED (--skip-embed-allowlist). $admin_help"
elif [[ "$EMBED_POLICY" == ALLOW_ALL_DOMAINS ]]; then
  echo "embedding is allowed on every domain; nothing to add"
else
  # ALLOW_APPROVED_DOMAINS (checked in step 1). Optimistic concurrency with the setting's etag: write
  # back exactly the list just read plus this domain, sent with that read's etag, so if another admin
  # changed the list in between the update is refused instead of dropping their entry;
  # then read again and retry.
  # Caveat: a setting never written has no etag to send (and no entries to lose), so two first-ever
  # writes at the same moment are not protected: the later one wins and can drop the earlier domain.
  update_err=$(mktemp)
  result=""
  for attempt in 1 2 3 4 5; do
    current=$(read_approved_domains)
    domains=$(jq -c '.domains' <<<"$current")
    etag=$(jq -r '.etag' <<<"$current")
    if jq -e --arg d "$app_domain" 'index($d) != null' <<<"$domains" >/dev/null; then
      result="'$app_domain' is already an approved embedding domain"
      break
    fi
    body=$(jq -n --arg d "$app_domain" --arg etag "$etag" --argjson current "$domains" '{
      allow_missing: true,
      field_mask: "aibi_dashboard_embedding_approved_domains.approved_domains",
      setting: ({aibi_dashboard_embedding_approved_domains: {approved_domains: ($current + [$d])}}
                + (if $etag == "" then {} else {etag: $etag} end))}')
    if databricks settings aibi-dashboard-embedding-approved-domains update -p "$PROFILE" --json "$body" >/dev/null 2>"$update_err"; then
      result="added '$app_domain' to the $(jq length <<<"$domains") approved embedding domain(s) already there"
      break
    fi
    # Refused: a changed etag means another write landed in between (retry on a fresh read);
    # an unchanged one means a real error.
    [[ "$(read_approved_domains | jq -r '.etag')" != "$etag" ]] \
      || fail "could not add '$app_domain' to the approved embedding domains: $(cat "$update_err"). $admin_help"
    echo "the approved embedding domains changed while updating them (attempt $attempt); reading them again"
  done
  rm -f "$update_err"
  [[ -n "$result" ]] || fail "the approved embedding domains kept changing during 5 attempts; rerun step 9 later. $admin_help"
  # Check the result: the domain must be there now. An entry of the list read above that is gone was
  # removed by someone else after this update (it was sent with the etag of that read).
  after=$(read_approved_domains | jq -c '.domains')
  jq -e --arg d "$app_domain" 'index($d) != null' <<<"$after" >/dev/null \
    || fail "'$app_domain' is not among the approved embedding domains after the update ($after). $admin_help"
  gone=$(jq -c --argjson before "$domains" '$before - .' <<<"$after")
  [[ "$gone" == "[]" ]] || echo "note: since this update, someone else removed $gone from the approved embedding domains"
  echo "$result"
fi

trap - ERR
echo
echo "Done. App: $app_url"
echo "The DAILY job's schedule (06:00 UTC) is paused; run it with:"
echo "  databricks bundle run ecg12_afib_daily$(printf ' %q' "${BUNDLE[@]}")"
[[ $RUN_DAILY -eq 1 ]] || { echo "and once it has run, republish the dashboard so its Ask Genie works (step 6):"; echo "  $publish_cmd"; }
