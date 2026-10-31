#!/usr/bin/env bash
# Reproducible `apx build`: the same steps apx runs, with every dependency taken from the
# committed lockfiles (bun.lock, uv.lock). Output: .build/ (wheel + app.yml + requirements.txt).
#
# Why not plain `apx build`? apx 0.3.8 (latest release, commit 50073e5) starts every UI build
# with `bun add --dev vite @tailwindcss/vite @vitejs/plugin-react @tanstack/router-plugin
# @tanstack/router-generator @opentelemetry/{sdk-logs,exporter-logs-otlp-http,api-logs,resources}`
# with no versions (crates/core/src/common.rs `ensure_entrypoint_deps`), so bun moves them to
# @latest and rewrites package.json + bun.lock. With the 2026-10 latest, @opentelemetry/sdk-logs
# 3.x needs @opentelemetry/api >=1.10 but the lock has 1.9.1, and the UI build fails with
# "Export named 'SeverityNumber' not found". apx's `uv sync` also re-locks uv.lock whenever the
# configured package index URL differs from the one recorded in the lock.
#
# Checks (each one fails the build):
#   1. apx is exactly 0.3.8, the version whose build steps this script copies.
#   2. The app's copy of the shared modules (src/ecg_demo_react/backend/common/) is
#      byte-identical to ../common/, the source of truth.
#   3. The apx UI entrypoint is fetched by full commit SHA and matches its SHA-256.
#   4. package.json, bun.lock and uv.lock are byte-identical before and after the build.
set -euo pipefail
cd "$(dirname "$0")"

APX_REQUIRED_VERSION=0.3.8
APX_COMMIT=50073e561a689ae5292f82e69ae0505417d75d3e   # tag v0.3.8 (git ls-remote ... refs/tags/v0.3.8)
ENTRYPOINT_SHA256=b043e6c1df9684f518e34b8a6a7ee69336140458c09ef1f65fc6985875e6c6ee
ENTRYPOINT=node_modules/.apx/entrypoint.ts
UI_ROOT="$PWD/src/ecg_demo_react/ui"
DIST_DIR="$PWD/src/ecg_demo_react/__dist__"
SHARED_SRC=../common
SHARED_COPY=src/ecg_demo_react/backend/common
SHARED_MODULES=(__init__.py config.py ecg_signal_gen.py patient_info.py)
LOCKFILES=(package.json bun.lock uv.lock)
APP_NAME=$(sed -n 's/^app-name = "\(.*\)"$/\1/p' pyproject.toml)  # [tool.apx.metadata] app-name

# 1. apx version ("apx 0.3.8"; the grep also drops the terminal escape codes apx prints).
apx_version=$(apx --version 2>/dev/null | grep -Eo 'apx [0-9]+\.[0-9]+\.[0-9]+[^[:space:]]*' | head -n 1 || true)
if [[ "$apx_version" != "apx $APX_REQUIRED_VERSION" ]]; then
  echo "ERROR: need apx $APX_REQUIRED_VERSION exactly, found '${apx_version:-no apx on PATH}'." >&2
  exit 1
fi

# 2. Shared-module copy.
for module in "${SHARED_MODULES[@]}"; do
  if ! cmp -s "$SHARED_SRC/$module" "$SHARED_COPY/$module"; then
    echo "ERROR: $SHARED_COPY/$module differs from $SHARED_SRC/$module." >&2
    echo "       Refresh the copy: cp $SHARED_SRC/$module $SHARED_COPY/$module" >&2
    exit 1
  fi
done

lock_digest() { shasum -a 256 "${LOCKFILES[@]}"; }
locks_before=$(lock_digest)

export UV_FROZEN=1                  # every `uv sync` apx runs uses uv.lock as-is

apx bun install --frozen-lockfile   # JS deps exactly as in bun.lock

# 3. apx's vite build script; `apx build` writes this same file (embedded in the apx binary).
mkdir -p "$(dirname "$ENTRYPOINT")"
curl -fsSL -o "$ENTRYPOINT" \
  "https://raw.githubusercontent.com/databricks-solutions/apx/$APX_COMMIT/src/apx/assets/entrypoint.ts"
echo "$ENTRYPOINT_SHA256  $ENTRYPOINT" | shasum -a 256 -c -

# Pass 1: preflight (_metadata.py, _version.py, __dist__/, uv sync) + OpenAPI client, no UI.
apx build --skip-ui-build

# UI build: what `apx build` runs right after its `bun add --dev` step.
NODE_PATH="$PWD/node_modules" APX_APP_NAME="$APP_NAME" \
  apx bun run "$ENTRYPOINT" build "$UI_ROOT" "$DIST_DIR" "$UI_ROOT/public"

# Pass 2: rebuild the wheel so it ships the compiled UI in __dist__/.
apx build --skip-ui-build

# 4. Lockfiles untouched.
if [[ "$(lock_digest)" != "$locks_before" ]]; then
  echo "ERROR: the build changed a lockfile:" >&2
  diff <(echo "$locks_before") <(lock_digest) >&2 || true
  exit 1
fi
echo "Build OK: apx $APX_REQUIRED_VERSION, shared modules in sync, lockfiles unchanged."
