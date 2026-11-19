#!/usr/bin/env bash
# Full install of the 12-lead ECG/AFib demo into YOUR workspace: the bundle's `full` target (SETUP
# + DAILY jobs, AI/BI dashboard, Databricks App), then the steps a bundle deploy cannot do alone.
#
#   ./deploy.sh -p <profile> --catalog <catalog> --warehouse-id <id> [options]
#
# Steps (each one fails loudly and stops the script):
#   1. prerequisites: apx 0.3.8 (with its bun), uv, Databricks CLI >= 0.281.0, jq, python3, the
#      required variables (catalog and schema: letters, digits and underscores), a profile that
#      authenticates, and (unless --skip-embed-allowlist) that it can read the workspace's
#      dashboard-embedding settings step 8 changes: a caller who cannot learns it here, before
#      anything is built or deployed
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
Usage: ./deploy.sh -p PROFILE --catalog CATALOG --warehouse-id ID [options]

Required:
  -p, --profile PROFILE       Databricks CLI profile of the target workspace (~/.databrickscfg)
  --catalog CATALOG           Unity Catalog catalog (the SETUP job creates the schema in it)
  --warehouse-id ID           SQL warehouse for the AI/BI dashboard

Options:
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
  --run-daily                 also run the DAILY job once after SETUP
  --skip-embed-allowlist      do not change the dashboard-embedding allowlist (not a workspace
                              admin); the script prints the domain an admin must add
  -h, --help                  show this help
USAGE
}

PROFILE="" CATALOG="" WAREHOUSE_ID="" SCHEMA="" ENDPOINT_PREFIX="" MLFLOW_EXPERIMENT="" APP_NAME="" VIEWER_GROUP=""
EMBED_CREDENTIALS="" REMOVE_AFTER="" RUN_DAILY=0 SKIP_ALLOWLIST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--profile) PROFILE="${2:-}"; shift 2 ;;
    --catalog) CATALOG="${2:-}"; shift 2 ;;
    --warehouse-id) WAREHOUSE_ID="${2:-}"; shift 2 ;;
    --schema) SCHEMA="${2:-}"; shift 2 ;;
    --endpoint-prefix) ENDPOINT_PREFIX="${2:-}"; shift 2 ;;
    --mlflow-experiment) MLFLOW_EXPERIMENT="${2:-}"; shift 2 ;;
    --app-name) APP_NAME="${2:-}"; shift 2 ;;
    --viewer-group) VIEWER_GROUP="${2:-}"; shift 2 ;;
    --embed-credentials) EMBED_CREDENTIALS="${2:-}"; shift 2 ;;
    --remove-after) REMOVE_AFTER="${2:-}"; shift 2 ;;
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
case "$EMBED_CREDENTIALS" in
  ""|true|false) ;;
  *) echo "ERROR: --embed-credentials takes true or false, not '$EMBED_CREDENTIALS'" >&2; exit 2 ;;
esac
if [[ -n "$REMOVE_AFTER" && ! "$REMOVE_AFTER" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "ERROR: --remove-after takes a date YYYY-MM-DD, not '$REMOVE_AFTER'" >&2; exit 2
fi

# Every bundle command gets the same target, profile and variables.
BUNDLE=(-t full -p "$PROFILE" --var "catalog=$CATALOG" --var "warehouse_id=$WAREHOUSE_ID")
[[ -z "$SCHEMA" ]] || BUNDLE+=(--var "schema=$SCHEMA")
[[ -z "$ENDPOINT_PREFIX" ]] || BUNDLE+=(--var "endpoint_prefix=$ENDPOINT_PREFIX")
[[ -z "$MLFLOW_EXPERIMENT" ]] || BUNDLE+=(--var "mlflow_experiment=$MLFLOW_EXPERIMENT")
[[ -z "$APP_NAME" ]] || BUNDLE+=(--var "app_name=$APP_NAME")
[[ -z "$VIEWER_GROUP" ]] || BUNDLE+=(--var "viewer_group=$VIEWER_GROUP")
[[ -z "$EMBED_CREDENTIALS" ]] || BUNDLE+=(--var "embed_credentials=$EMBED_CREDENTIALS")
[[ -z "$REMOVE_AFTER" ]] || BUNDLE+=(--var "remove_after=$REMOVE_AFTER")
APP_NAME="${APP_NAME:-ecg12-demo}"

# ---------------------------------------------------------------------------------------------
step "1/8 prerequisites"
for cmd in apx uv databricks jq python3; do
  command -v "$cmd" >/dev/null || fail "'$cmd' is not on PATH (see docs/deploy.md, Path 2 prerequisites)."
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
# Endpoint names from the single source of truth, common/config.py (it imports only the standard
# library): <prefix>_arr / <prefix>_afib, empty prefix = the bundle's default. This also rejects a
# prefix an endpoint name cannot hold, before anything is built or deployed.
endpoints=$(python3 -c 'import sys; sys.path.insert(0, "."); from common import config
print(*config.endpoint_names(sys.argv[1]))' "$ENDPOINT_PREFIX" 2>&1) \
  || fail "invalid --endpoint-prefix: ${endpoints##*: }"
read -r ARR_ENDPOINT AFIB_ENDPOINT <<<"$endpoints"
echo "serving endpoints: $ARR_ENDPOINT, $AFIB_ENDPOINT (workspace-wide names; the SETUP job stops if another deployment owns them)"
# Catalog and schema: plain identifiers, the same check the notebooks make before any SQL
# (config.uc_location), so a name the jobs would refuse stops here, before anything is deployed.
uc_error=$(python3 -c 'import sys; sys.path.insert(0, "."); from common import config
try:
    config.uc_location(sys.argv[1], sys.argv[2])
except ValueError as error:
    sys.exit(str(error))' "$CATALOG" "$SCHEMA" 2>&1) || fail "invalid --catalog / --schema: $uc_error"
# MLflow experiment, if given: the notebooks' own check (config.resolve_mlflow_experiment).
if [[ -n "$MLFLOW_EXPERIMENT" ]]; then
  exp_error=$(python3 -c 'import sys; sys.path.insert(0, "."); from common import config
try:
    config.resolve_mlflow_experiment(sys.argv[1])
except ValueError as error:
    sys.exit(str(error))' "$MLFLOW_EXPERIMENT" 2>&1) || fail "invalid --mlflow-experiment: $exp_error"
fi
me=$(databricks current-user me -p "$PROFILE" -o json | jq -r '.userName') \
  || fail "profile '$PROFILE' does not authenticate: run 'databricks auth login -p $PROFILE'."
echo "apx 0.3.8, uv $(uv --version | cut -d ' ' -f 2), Databricks CLI $cli_version; deploying as $me"

# The dashboard-embedding settings step 8 changes are workspace admin settings. Read them now, so a
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
  [[ -n "$EMBED_POLICY" ]] || fail "cannot read the dashboard-embedding policy that step 8 needs (a workspace admin setting): ${policy_message:-empty reply}. Rerun with --skip-embed-allowlist; the script then prints the app domain a workspace admin must add."
  case "$EMBED_POLICY" in
    ALLOW_ALL_DOMAINS) ;;
    ALLOW_APPROVED_DOMAINS) read_approved_domains >/dev/null ;;     # the list step 8 updates is readable too
    DENY_ALL_DOMAINS)
      fail "dashboard embedding is disabled in this workspace (DENY_ALL_DOMAINS), so the app could not show the dashboard. A workspace admin must allow approved domains (Settings > Security > Embed dashboards), or rerun with --skip-embed-allowlist." ;;
    *)
      fail "unexpected dashboard-embedding policy '$EMBED_POLICY'." ;;
  esac
  echo "dashboard-embedding policy: $EMBED_POLICY (readable; step 8 adds the app's domain if needed)"
fi

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
  [[ -n "$result" ]] || fail "the approved embedding domains kept changing during 5 attempts; rerun step 8 later. $admin_help"
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
