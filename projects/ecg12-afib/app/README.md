# ECG/AFib 12-lead demo app (APX: React + FastAPI)

Built with [`apx`](https://github.com/databricks-solutions/apx) 0.3.8. `pyproject.toml` points its
`readme` here, so the wheel build needs this file.

- **Backend** (`src/ecg_demo_react/backend/`): `/api/generate` returns a synthetic 12-lead ECG,
  its lead names and a synthetic patient record; `/api/analyze` sends all 12 leads to the
  `bio_ecg12_arr` endpoint and, when flagged, to `bio_ecg12_afib` (app service-principal identity,
  `WorkspaceClient()`; names from `common/config.py`, each endpoint named after its model);
  `/api/app-config` gives the UI its deep-link names (endpoint and model pages, the `@champion` label).
- **`backend/common/`** is a byte-identical copy of `../common/` (`__init__`, `config`,
  `ecg_signal_gen`, `patient_info`), the source of truth. After changing `../common/`, copy the
  file over (`cp ../common/<module>.py src/ecg_demo_react/backend/common/`); the build stops if
  the copies differ.

## Build

```bash
./build_from_lockfiles.sh   # -> .build/ (wheel + app.yml + requirements.txt)
```

The script runs the same steps as `apx build`, but takes every dependency from the committed
`bun.lock` / `uv.lock`, so the build reproduces the live `ekg-demo` app's versions. It fails
unless `apx --version` is exactly 0.3.8, fetches apx's UI entrypoint by full commit SHA and checks
its SHA-256, checks the `backend/common/` copy, and fails if `package.json`, `bun.lock` or
`uv.lock` changed during the build.

**Why not plain `apx build`?** apx 0.3.8 starts every UI build with an unversioned
`bun add --dev` of nine build tools (vite, the TanStack router plugin, Tailwind's vite plugin,
four OpenTelemetry packages). That moves them to their newest release and rewrites
`package.json` + `bun.lock`. With today's newest OpenTelemetry (`@opentelemetry/sdk-logs` 3.x,
which needs `@opentelemetry/api` >= 1.10 while the lock has 1.9.1) the UI build then fails with
`Export named 'SeverityNumber' not found`. apx's `uv sync` also rewrites `uv.lock` when your
package index URL differs from the one recorded in the lock; the script sets `UV_FROZEN=1`.

## Local development

```bash
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

Deploy steps are in [`../README.md`](../README.md).
