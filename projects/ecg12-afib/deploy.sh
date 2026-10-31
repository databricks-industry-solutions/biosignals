#!/usr/bin/env bash
# Full install of the 12-lead ECG/AFib demo into YOUR workspace: the bundle's `full` target (SETUP
# + DAILY jobs, AI/BI dashboard, Databricks App), then the steps a bundle deploy cannot do alone.
#
#   ./deploy.sh -p <profile> --catalog <catalog> --warehouse-id <id> [options]
#
# Steps (each one fails loudly and stops the script):
#   1. prerequisites: apx 0.3.8 (with its bun), uv, Databricks CLI >= 0.281.0, jq, python3, the
#      required variables, and a profile that authenticates
#   2. build the app from the lockfiles (app/build_from_lockfiles.sh)
#   3. databricks bundle validate --strict -t full
#   4. databricks bundle deploy -t full (creates the jobs, the dashboard and the app; grants the
#      viewer group CAN_READ on the dashboard and CAN_USE on the app, from databricks.yml)
#   5. run the SETUP job and wait for it: training data, both models, the validation gate, the two
#      serving endpoints (about 30-60 minutes: GPU training, then endpoint creation)
#      (--run-daily also runs the DAILY job once, so the dashboard's batch tiles have data)
#   6. grant the app's service principal CAN_QUERY on both serving endpoints (they exist only
#      after step 5, so they cannot be app resources of the first deploy)
#   7. start the app: databricks bundle run ecg12_afib_app (deploys the built source with its
#      environment, including the dashboard id)
#   8. add the app's domain to the workspace's dashboard-embedding allowlist (workspace admin
#      setting; done after step 7 because the app's URL exists once the app is deployed). It reads
#      the current list and writes it back with the domain added, keeping every entry already
#      there. --skip-embed-allowlist makes no settings call at all and prints what an admin must add.
#
# Re-running is safe: every step updates in place (SETUP retrains and re-gates the models).
# Remove everything with: databricks bundle destroy -t full -p <profile> --var ... (see README.md).
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  cat <<'USAGE'
Usage: ./deploy.sh -p PROFILE --catalog CATALOG --warehouse-id ID [options]

Required:
  -p, --profile PROFILE       Databricks CLI profile of the target workspace (~/.databrickscfg)
  --catalog CATALOG           Unity Catalog catalog (the SETUP job creates the schema in it)
  --warehouse-id ID           SQL warehouse for the AI/BI dashboard

Options:
  --schema SCHEMA             Unity Catalog schema (default: biosignals)
  --app-name NAME             app name (default: ecg12-demo; app names are workspace-wide)
  --viewer-group GROUP        group that may view the dashboard and use the app (default: users)
  --run-daily                 also run the DAILY job once after SETUP
  --skip-embed-allowlist      do not change the dashboard-embedding allowlist (not a workspace
                              admin); the script prints the domain an admin must add
  -h, --help                  show this help
USAGE
}

PROFILE="" CATALOG="" WAREHOUSE_ID="" SCHEMA="" APP_NAME="" VIEWER_GROUP=""
RUN_DAILY=0 SKIP_ALLOWLIST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--profile) PROFILE="${2:-}"; shift 2 ;;
    --catalog) CATALOG="${2:-}"; shift 2 ;;
    --warehouse-id) WAREHOUSE_ID="${2:-}"; shift 2 ;;
    --schema) SCHEMA="${2:-}"; shift 2 ;;
    --app-name) APP_NAME="${2:-}"; shift 2 ;;
    --viewer-group) VIEWER_GROUP="${2:-}"; shift 2 ;;
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

missing=()
[[ -n "$PROFILE" ]] || missing+=("-p/--profile")
[[ -n "$CATALOG" ]] || missing+=("--catalog")
[[ -n "$WAREHOUSE_ID" ]] || missing+=("--warehouse-id")
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "ERROR: missing required argument(s): ${missing[*]}" >&2
  usage >&2
  exit 2
fi

# Every bundle command gets the same target, profile and variables.
BUNDLE=(-t full -p "$PROFILE" --var "catalog=$CATALOG" --var "warehouse_id=$WAREHOUSE_ID")
[[ -z "$SCHEMA" ]] || BUNDLE+=(--var "schema=$SCHEMA")
[[ -z "$APP_NAME" ]] || BUNDLE+=(--var "app_name=$APP_NAME")
[[ -z "$VIEWER_GROUP" ]] || BUNDLE+=(--var "viewer_group=$VIEWER_GROUP")
APP_NAME="${APP_NAME:-ecg12-demo}"

# ---------------------------------------------------------------------------------------------
step "1/8 prerequisites"
for cmd in apx uv databricks jq python3; do
  command -v "$cmd" >/dev/null || fail "'$cmd' is not on PATH (see README.md, Prerequisites)."
done
apx_version=$(apx --version 2>/dev/null | grep -Eo 'apx [0-9]+\.[0-9]+\.[0-9]+[^[:space:]]*' | head -n 1 || true)
[[ "$apx_version" == "apx 0.3.8" ]] || fail "need apx 0.3.8 exactly, found '${apx_version:-none}'."
apx bun --version >/dev/null 2>&1 || fail "apx's bun does not run ('apx bun --version'); reinstall apx 0.3.8."
cli_version=$(databricks --version | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
# dataset_catalog / dataset_schema on dashboard resources need CLI 0.281.0 or later.
python3 - "$cli_version" <<'PY' || fail "need Databricks CLI >= 0.281.0, found $cli_version."
import sys
sys.exit(0 if tuple(map(int, sys.argv[1].split("."))) >= (0, 281, 0) else 1)
PY
me=$(databricks current-user me -p "$PROFILE" -o json | jq -r '.userName') \
  || fail "profile '$PROFILE' does not authenticate: run 'databricks auth login -p $PROFILE'."
echo "apx 0.3.8, uv $(uv --version | cut -d ' ' -f 2), Databricks CLI $cli_version; deploying as $me"
# Endpoint names come from the single source of truth (common/config.py imports only datetime).
read -r ARR_ENDPOINT AFIB_ENDPOINT < <(python3 -c \
  'import sys; sys.path.insert(0, "."); from common import config; print(config.ARR_ENDPOINT, config.AFIB_ENDPOINT)')

# ---------------------------------------------------------------------------------------------
step "2/8 build the app from the lockfiles"
app/build_from_lockfiles.sh

# ---------------------------------------------------------------------------------------------
step "3/8 bundle validate"
databricks bundle validate --strict "${BUNDLE[@]}"

# ---------------------------------------------------------------------------------------------
step "4/8 bundle deploy (jobs, dashboard, app; viewer permissions)"
databricks bundle deploy "${BUNDLE[@]}"

# ---------------------------------------------------------------------------------------------
step "5/8 run the SETUP job (train, gate, promote, serve; waits until it finishes)"
databricks bundle run ecg12_afib_setup "${BUNDLE[@]}"
if [[ $RUN_DAILY -eq 1 ]]; then
  STEP="5/8 run the DAILY job once"
  echo "--> DAILY job (one simulated day, so the dashboard's batch tiles have data)"
  databricks bundle run ecg12_afib_daily "${BUNDLE[@]}"
fi

# ---------------------------------------------------------------------------------------------
step "6/8 grant the app's service principal CAN_QUERY on both endpoints"
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
step "7/8 start the app (deploy its source and environment)"
databricks bundle run ecg12_afib_app "${BUNDLE[@]}"
app_url=$(databricks apps get "$APP_NAME" -p "$PROFILE" -o json | jq -r '.url // empty')
[[ -n "$app_url" ]] || fail "app '$APP_NAME' has no URL after its deployment."

# ---------------------------------------------------------------------------------------------
step "8/8 dashboard-embedding allowlist (workspace admin setting)"
app_domain=${app_url#*://}
app_domain=${app_domain%%/*}
admin_help="A workspace admin must allow embedding on '$app_domain': Settings > Security > Embed dashboards (approved domains)."
if [[ $SKIP_ALLOWLIST -eq 1 ]]; then
  # No settings API call at all: reading these settings needs a workspace admin too.
  echo "SKIPPED (--skip-embed-allowlist). $admin_help"
else
  policy=$(databricks settings aibi-dashboard-embedding-access-policy get -p "$PROFILE" -o json \
    | jq -r '.aibi_dashboard_embedding_access_policy.access_policy_type') \
    || fail "cannot read the dashboard-embedding policy (needs a workspace admin; rerun with --skip-embed-allowlist to skip this step). $admin_help"
  case "$policy" in
    ALLOW_ALL_DOMAINS)
      echo "embedding is allowed on every domain; nothing to add" ;;
    DENY_ALL_DOMAINS)
      fail "dashboard embedding is disabled in this workspace (DENY_ALL_DOMAINS), so the app cannot show the dashboard. $admin_help" ;;
    ALLOW_APPROVED_DOMAINS)
      # Read-modify-write: keep every approved domain already there and add this app's. The CLI's
      # stderr is kept apart, so a warning on stderr cannot corrupt the JSON reply.
      get_err=$(mktemp)
      if ! current=$(databricks settings aibi-dashboard-embedding-approved-domains get -p "$PROFILE" -o json 2>"$get_err"); then
        if grep -q -E 'RESOURCE_DOES_NOT_EXIST|NOT_FOUND' "$get_err"; then
          current='{}'                                   # never set: no approved domains yet
        else
          fail "cannot read the approved embedding domains (needs a workspace admin; rerun with --skip-embed-allowlist to skip this step): $(cat "$get_err"). $admin_help"
        fi
      fi
      rm -f "$get_err"
      domains=$(jq -c '.aibi_dashboard_embedding_approved_domains.approved_domains // []' <<<"$current") \
        || fail "unexpected reply when reading the approved embedding domains: $current"
      if jq -e --arg d "$app_domain" 'index($d) != null' <<<"$domains" >/dev/null; then
        echo "'$app_domain' is already an approved embedding domain"
      else
        body=$(jq -n --arg d "$app_domain" --argjson current "$domains" '{
          allow_missing: true,
          field_mask: "aibi_dashboard_embedding_approved_domains.approved_domains",
          setting: {aibi_dashboard_embedding_approved_domains: {approved_domains: ($current + [$d])}}}')
        databricks settings aibi-dashboard-embedding-approved-domains update -p "$PROFILE" --json "$body" >/dev/null \
          || fail "could not add '$app_domain' to the approved embedding domains (needs a workspace admin; rerun with --skip-embed-allowlist to skip this step). $admin_help"
        echo "added '$app_domain' to the $(jq length <<<"$domains") approved embedding domain(s) already there"
      fi ;;
    *)
      fail "unexpected dashboard-embedding policy '$policy'. $admin_help" ;;
  esac
fi

trap - ERR
echo
echo "Done. App: $app_url"
echo "The DAILY job's schedule (06:00 UTC) is paused; run it with:"
echo "  databricks bundle run ecg12_afib_daily -t full -p $PROFILE --var catalog=$CATALOG --var warehouse_id=$WAREHOUSE_ID"
