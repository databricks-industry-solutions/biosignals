# ECG/AFib 12-lead demo app (APX: React + FastAPI)

> **Not for clinical use as provided.** Synthetic data only; the app's triage results and
> recommended actions are not a diagnosis or medical advice, and the app is not a medical device.
> It says so in its header and next to every result. See the
> [disclaimer](../../../README.md#disclaimer).

Built with [`apx`](https://github.com/databricks-solutions/apx) 0.3.8. `pyproject.toml` points its
`readme` here, so the wheel build needs this file.

- **Backend** (`src/ecg_demo_react/backend/`): `/api/generate` returns a synthetic 12-lead ECG,
  its lead names and a synthetic patient record; `/api/analyze` sends all 12 leads to the triage
  endpoint `<endpoint_prefix>_arr` and, when flagged, to the classify endpoint
  `<endpoint_prefix>_afib` (app service-principal identity; `common/config.endpoint_names` on the
  app's `ECG_DEMO_REACT_ENDPOINT_PREFIX`, default `bio_ecg12`);
  `/api/warmup` sends one single-ECG request to each endpoint (tagged `client_request_id =
  app-warmup`) and returns each one's status (`ready` / `cold` / `error`); `/api/app-config` gives
  the UI its deep-link names (endpoint and model pages, the DAILY job, the `@champion` label) and its
  time limits.
- **Scale-to-zero.** Both endpoints scale to zero when idle, and a cold one can take a minute or
  more to answer. Every endpoint call is bounded by `SERVING_TIMEOUT_S` (`common/config.py`, set as
  the databricks-sdk `http_timeout_seconds` and `retry_timeout_seconds` of a dedicated serving
  client); a call that hits it gets a 503 with a "waking up, try again" message instead of hanging
  (429 / 502 for rate limits and other endpoint errors). The Live tab calls `/api/warmup` on load
  and every `WARMUP_INTERVAL_S` while the page is visible (stopping while it is hidden), shows a
  status chip per endpoint, and gives Triage → Classify its own time limit, a Cancel button and
  Retry.
- **Rate limits** (`backend/router.py`, `ANALYZE_LIMIT` / `WARMUP_LIMIT`). `/api/analyze` allows
  10 requests per user and 60 from all users together in any 60 s; `/api/warmup` 12 and 60 in any
  `WARMUP_INTERVAL_S`. A refused call gets a 429 with a `Retry-After` header and a message, before
  any endpoint is queried. A user is the `X-Forwarded-User` header the Databricks Apps proxy sets
  (locally: the client address). When both endpoints answered a warm-up ping less than half of
  `WARMUP_INTERVAL_S` ago, `/api/warmup` answers from that ping without a new query and without
  counting it, so many open tabs still query each endpoint about once per interval. The counts and
  that last answer live in one SQLite file in the system temp folder (`backend/shared_state.py`),
  so both uvicorn workers (`app.yml`) share them.
- **Errors.** An ECG that is not 12 leads × 2500 samples of finite numbers is a 422. An endpoint
  error, an endpoint answer the app cannot read or a failed connection is a 502 whose message
  points to the app's logs; the endpoint's own error text goes to the log, not to the user.
- **Configuration** comes from the app's environment, not the code (`backend/core/_config.py`,
  prefix `ECG_DEMO_REACT_`): `ECG_DEMO_REACT_CATALOG` (required; the app does not start without
  it), `ECG_DEMO_REACT_UC_SCHEMA` (default `biosignals`), `ECG_DEMO_REACT_ENDPOINT_PREFIX` (default
  `bio_ecg12`: the serving endpoints `<prefix>_arr` / `<prefix>_afib`; a value an endpoint name
  cannot hold stops the app at startup), `ECG_DEMO_REACT_DASHBOARD_ID` (optional; without it the
  Batch tab shows a "Dashboard not configured" note) and `ECG_DEMO_REACT_DAILY_JOB_ID` (optional;
  the header's Lakeflow link opens that job, else the jobs list). The bundle's `full` target sets
  all five (`../databricks.yml`,
  `resources.apps.ecg12_afib_app.config.env`), the two ids from the dashboard and DAILY job it
  deploys. The workspace host and id come from the app's own
  credentials, which the Databricks Apps runtime provides.
- **`backend/common/`** is a byte-identical copy of `../common/` (`__init__`, `config`,
  `ecg_signal_gen`, `patient_info`), the source of truth. After changing `../common/`, copy the
  file over (`cp ../common/<module>.py src/ecg_demo_react/backend/common/`); the build stops if
  the copies differ.

## Build

```bash
./build_from_lockfiles.sh   # -> .build/ (wheel + app.yml + requirements.txt)
```

The script runs the same steps as `apx build`, but takes every dependency from the committed
`bun.lock` / `uv.lock`, so every build uses the same versions. It fails unless `apx --version` is
exactly 0.3.8, checks the `backend/common/` copy, checks that both lockfiles point only at the
public registries (PyPI and npm), fetches apx's UI entrypoint by full commit SHA and checks its
SHA-256, and fails if `package.json`, `bun.lock` or `uv.lock` changed during the build.

**Deployed versions = locked versions.** `apx build` writes a `.build/requirements.txt` that lists
only the app wheel, whose dependencies are version ranges, so Databricks Apps would pick the
versions at deploy time. The script rewrites it from `uv.lock` (`uv export`): every runtime
dependency with its exact version and file hashes, then the wheel with its hash. Databricks Apps
installs `requirements.txt` with pip, which in hash-checking mode installs exactly those files or
fails.

**Registries.** The lockfiles record PyPI (`pypi.org`, `files.pythonhosted.org`) and the default
npm registry. bun fetches default-registry packages from whatever npm registry you have
configured; uv downloads the `files.pythonhosted.org` URLs in `uv.lock` as they are. To change a
dependency, edit `pyproject.toml` / `package.json`, re-lock against the public registries
(`uv lock --default-index https://pypi.org/simple`; `apx bun install` with the default npm
registry configured), and commit the lockfiles. A lock written through a mirror records the
mirror's URLs, and the build refuses it.

**Security floors.** The `overrides` in `package.json` keep six packages the UI pulls in
indirectly at or above the first release that fixes a published advisory (`seroval-plugins`
moves with `seroval`); the lower bounds of `starlette` and `pydantic-settings` in
`pyproject.toml` do the same. After a re-lock, check both lockfiles for known advisories:
`pip-audit` (with `--vulnerability-service osv`) on the output of `uv export --frozen --no-dev`,
and an [OSV](https://osv.dev) query for each package version in `bun.lock`.

**Why not plain `apx build`?** apx 0.3.8 starts every UI build with an unversioned
`bun add --dev` of nine build tools (vite, the TanStack router plugin, Tailwind's vite plugin,
four OpenTelemetry packages). That moves them to their newest release and rewrites
`package.json` + `bun.lock`. With today's newest OpenTelemetry (`@opentelemetry/sdk-logs` 3.x,
which needs `@opentelemetry/api` >= 1.10 while the lock has 1.9.1) the UI build then fails with
`Export named 'SeverityNumber' not found`. apx's `uv sync` also rewrites `uv.lock` when your
package index URL differs from the one recorded in the lock; the script sets `UV_FROZEN=1`.

## Local development

```bash
cat > .env <<'ENV'          # read by backend/core/_config.py; git-ignored
ECG_DEMO_REACT_CATALOG=<catalog>
DATABRICKS_CONFIG_PROFILE=<profile>
ENV
apx dev start   # backend, frontend and OpenAPI watcher
```

`apx dev check`, `apx frontend build` and `apx frontend dev` run the same unversioned
`bun add --dev` step (`apx dev start` not checked), so they can also rewrite `package.json` /
`bun.lock`. Restore both with `git checkout -- package.json bun.lock` and re-run
`apx bun install --frozen-lockfile` before building or committing.

The TypeScript API client `src/ecg_demo_react/ui/lib/api.ts` is generated from the backend's
OpenAPI schema by `apx build` (the build script's first pass); commit it with backend changes.

Type checks without apx's dependency step:

```bash
UV_FROZEN=1 uv run ty check src                 # Python
./node_modules/.bin/tsc --noEmit -p tsconfig.json   # TypeScript
```

Deploy steps are in [`../docs/deploy.md`](../docs/deploy.md).
