# Third-party packages and licenses

[← ecg12-afib README](../README.md)

## Third-party packages and licenses

The source in this project is provided subject to the Databricks License (see
[LICENSE.md](../../../LICENSE.md)). The packages below keep their own licenses. Versions come from the
lockfiles (`app/uv.lock`, `app/bun.lock`), the notebooks' `%pip install` lines and
`dev/tests/run_tests_python311.sh`; "preinstalled" means the package comes with the Databricks
serverless environment the job runs in (AI v5 for training, ML v5 for batch scoring) and the
project does not pin it. Licenses were read from each package's own metadata or license file.

### Shipped: packages the project's code uses directly

| Package | Version | Purpose (where used) | License | Copyright | Source |
|---|---|---|---|---|---|
| `@tanstack/react-query` | 5.90.16 | API request caching (`ui/lib/api.ts`, `ui/main.tsx`, `ui/routes/`) | MIT | Copyright (c) 2021-present Tanner Linsley | [repo](https://github.com/TanStack/query) |
| `@tanstack/react-router` | 1.145.7 | Client-side routing (`ui/main.tsx`, `ui/routes/`) | MIT | Copyright (c) 2021-present Tanner Linsley | [repo](https://github.com/TanStack/router) |
| `class-variance-authority` | 0.7.1 | Button variant classes (`ui/components/ui/button.tsx`) | Apache-2.0 | Copyright 2022 Joe Bell | [repo](https://github.com/joe-bell/cva) |
| `clsx` | 2.1.1 | Joining CSS class names (`ui/lib/utils.ts`) | MIT | Copyright (c) Luke Edwards | [repo](https://github.com/lukeed/clsx) |
| `databricks-sdk` | 0.117.0 (app); 0.67.0 (`05_serve_endpoints`) | Workspace and serving-endpoint APIs (`backend/router.py`, `backend/core/_defaults.py`, `workflow/05_serve_endpoints.py`) | Apache-2.0 | Copyright 2023 Databricks, Inc. | [repo](https://github.com/databricks/databricks-sdk-py) |
| `fastapi` | 0.136.3 | App HTTP API (`app/src/ecg_demo_react/backend/`) | MIT | Copyright (c) 2018 Sebastián Ramírez | [repo](https://github.com/fastapi/fastapi) |
| `matplotlib` | preinstalled | ECG and evaluation plots (`common/ecg_plots.py`, `common/eval_plots.py`, notebooks 01, 04, 06) | Matplotlib License (PSF-based) | Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved | [repo](https://github.com/matplotlib/matplotlib) |
| `mlflow-skinny` | 3.8.1 (`05_serve_endpoints`); otherwise preinstalled | Experiment tracking, model registry, model loading (notebooks 02-05, 07, 08) | Apache-2.0 | Copyright 2018 Databricks, Inc. | [repo](https://github.com/mlflow/mlflow) |
| `neurokit2` | 0.2.12 | Recording noise for the synthetic ECGs (`common/ecg_signal_gen.py` and its app copy) | MIT | Copyright (c) 2020, Dominique Makowski | [repo](https://github.com/neuropsychology/NeuroKit) |
| `numpy` | 1.26.4 (app); preinstalled (notebooks) | ECG arrays, training, scoring, patient records (`common/`, app backend, notebooks) | BSD-3-Clause | Copyright (c) 2005-2023, NumPy Developers | [repo](https://github.com/numpy/numpy) |
| `pandas` | preinstalled | Training and scoring tables (`common/scoring.py`, `common/training_data.py`, notebooks 02, 03, 04, 06) | BSD-3-Clause | Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team; Copyright (c) 2011-2023, Open source contributors | [repo](https://github.com/pandas-dev/pandas) |
| `pydantic` | 2.13.4 | API request/response models (`backend/models.py`, `backend/core/`) | MIT | Copyright (c) 2017 to present Pydantic Services Inc. and individual contributors | [repo](https://github.com/pydantic/pydantic) |
| `pydantic-settings` | 2.14.1 | App configuration from environment variables (`backend/core/_config.py`) | MIT | Copyright (c) 2022 Samuel Colvin and other contributors | [repo](https://github.com/pydantic/pydantic-settings) |
| `pyspark` | preinstalled | Spark tables and transforms (`common/scoring.py`, `common/training_data.py`, notebooks 04, 06, 08) | Apache-2.0 | Copyright 2014 and onwards The Apache Software Foundation | [repo](https://github.com/apache/spark) |
| `python-dotenv` | 1.2.2 | Loading a local `.env` file (`backend/core/_config.py`) | BSD-3-Clause | Copyright (c) 2014, Saurabh Kumar (python-dotenv), 2013, Ted Tieken (django-dotenv-rw), 2013, Jacob Kaplan-Moss (django-dotenv) | [repo](https://github.com/theskumar/python-dotenv) |
| `PyWavelets` | 1.4.1 | Wavelet support needed by neurokit2 (`app/pyproject.toml`, notebooks 01, 05, 06) | MIT | Copyright (c) 2006-2012 Filip Wasilewski; Copyright (c) 2012-2020 The PyWavelets Developers | [repo](https://github.com/PyWavelets/pywt) |
| `react`, `react-dom` | 19.2.3 | App user interface (`app/src/ecg_demo_react/ui/`) | MIT | Copyright (c) Meta Platforms, Inc. and affiliates | [repo](https://github.com/facebook/react) |
| `recharts` | 3.8.1 | The 12-lead ECG chart (`ui/components/ecg-chart.tsx`); its dependency victory-vendor bundles d3 modules under ISC | MIT | Copyright (c) 2015-present recharts | [repo](https://github.com/recharts/recharts) |
| `scipy` | 1.13.1 (app); preinstalled (notebooks) | Signal processing needed by neurokit2 (`app/pyproject.toml`) | BSD-3-Clause | Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers | [repo](https://github.com/scipy/scipy) |
| `starlette` | 1.2.1 | Serving the built UI files (`backend/core/_static.py`) | BSD-3-Clause | Copyright © 2018, Encode OSS Ltd | [repo](https://github.com/Kludex/starlette) |
| `tailwind-merge` | 3.4.0 | Merging CSS classes (`ui/lib/utils.ts`) | MIT | Copyright (c) 2021 Dany Castillo | [repo](https://github.com/dcastil/tailwind-merge) |
| `tailwindcss` | 4.3.3 | UI styling (`ui/styles/globals.css`; declared directly in `package.json`) | MIT | Copyright (c) Tailwind Labs, Inc. | [repo](https://github.com/tailwindlabs/tailwindcss) |
| `torch` (PyTorch) | preinstalled | The two neural-network models and their training (`common/training.py`, notebooks 02, 03) | BSD-3-Clause | Copyright (c) 2016- Facebook, Inc. (Adam Paszke) and the other contributors listed in its LICENSE | [repo](https://github.com/pytorch/pytorch) |
| `uvicorn` | 0.49.0 | Runs the app's web server (`app/app.yml`) | BSD-3-Clause | Copyright © 2017-present, Encode OSS Ltd | [repo](https://github.com/Kludex/uvicorn) |

### Build and development tools (not shipped)

| Package | Version | Purpose (where used) | License | Source |
|---|---|---|---|---|
| `apx` | 0.3.8 (UI build entrypoint from commit `50073e5`) | Builds the app's UI and Python wheel (`app/build_from_lockfiles.sh`, `deploy.sh`) | Databricks License | [repo](https://github.com/databricks-solutions/apx) |
| `bun` | managed by apx, not pinned | Installs the locked UI packages and runs the UI build | MIT; the binary statically links LGPL-2 JavaScriptCore and LGPL-2.1 tinycc | [repo](https://github.com/oven-sh/bun) |
| `uv` | not pinned | Locked Python installs and the test environment (build and test scripts) | MIT OR Apache-2.0 | [repo](https://github.com/astral-sh/uv) |
| `hatchling` | not pinned | Python wheel build backend (`app/pyproject.toml`) | MIT | [repo](https://github.com/pypa/hatch) |
| `uv-dynamic-versioning` | >=0.7.0, not locked | Wheel version from git (`app/pyproject.toml`) | MIT | [repo](https://github.com/ninoseki/uv-dynamic-versioning) |
| `vite` | 8.0.16 | Bundles the React UI | MIT | [repo](https://github.com/vitejs/vite) |
| `@vitejs/plugin-react` | 6.0.2 | React support for the Vite build | MIT | [repo](https://github.com/vitejs/vite-plugin-react) |
| `@tailwindcss/vite` | 4.3.3 | Compiles Tailwind CSS during the build | MIT | [repo](https://github.com/tailwindlabs/tailwindcss) |
| `@tanstack/router-plugin`, `@tanstack/router-generator` | 1.168.18, 1.167.17 | Generates the typed route definitions | MIT | [repo](https://github.com/TanStack/router) |
| `@opentelemetry/api-logs`, `exporter-logs-otlp-http`, `sdk-logs`, `resources` | 0.218.0, 0.218.0, 0.218.0, 2.7.1 | Build logging in the apx UI build entrypoint | Apache-2.0 | [repo](https://github.com/open-telemetry/opentelemetry-js) |
| `typescript` | 5.9.3 | UI compilation and type checking (`app/tsconfig.json`) | Apache-2.0 | [repo](https://github.com/microsoft/TypeScript) |
| `@types/node`, `@types/react`, `@types/react-dom` | 25.0.3, 19.2.7, 19.2.3 | TypeScript type definitions | MIT | [repo](https://github.com/DefinitelyTyped/DefinitelyTyped) |
| `ty` | 0.0.47 | Python type checking (dev dependency) | MIT | [repo](https://github.com/astral-sh/ty) |
| `pytest` | 9.1.1 | Optional local checks (`dev/tests/`) | MIT | [repo](https://github.com/pytest-dev/pytest) |
| `cloudpickle` | 3.0.0 | Optional local checks: model serialization (`dev/tests/`) | BSD-3-Clause | [repo](https://github.com/cloudpipe/cloudpickle) |
| `PyYAML` | 6.0.3 | Optional local checks: reading the bundle files (`dev/tests/`) | MIT | [repo](https://github.com/yaml/pyyaml) |

### UI packages pulled in indirectly

`app/bun.lock` holds 204 UI packages: the 22 listed above and 182 they pull in. Of those 182, 50
are reachable from the UI's runtime packages (the `dependencies` of `app/package.json`, following
dependencies, optional dependencies and peer dependencies), so their code may end up in the compiled
UI, and 132 only from build tools.
All of them are MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause or 0BSD except the ones below.
Licenses were read from each package's `package.json`, or from the npm registry for platform
builds that aren't installed on the machine that checked them.

| Package | Version | Used | License | Source |
|---|---|---|---|---|
| `victory-vendor` | 37.3.6 | runtime: bundles d3 for `recharts` | MIT AND ISC | [repo](https://github.com/FormidableLabs/victory) |
| `d3-array`, `d3-color`, `d3-format`, `d3-interpolate`, `d3-path`, `d3-scale`, `d3-shape`, `d3-time`, `d3-time-format`, `d3-timer` | 3.2.4, 3.1.0, 3.1.2, 3.0.1, 3.1.0, 4.0.2, 3.2.0, 3.1.0, 4.1.0, 3.0.1 | runtime: chart scales and shapes for `recharts` | ISC | [repo](https://github.com/d3) |
| `internmap` | 2.0.3 | runtime: via `d3-array` | ISC | [repo](https://github.com/mbostock/internmap) |
| `isbot` | 5.1.42 | runtime: via `@tanstack/react-router` | Unlicense | [repo](https://github.com/omrilotan/isbot) |
| `lightningcss` (and its platform builds) | 1.32.0 | build: CSS processing in `vite` and Tailwind | MPL-2.0 | [repo](https://github.com/parcel-bundler/lightningcss) |
| `caniuse-lite` | 1.0.30001797 | build: browser-support data for `browserslist` | CC-BY-4.0 | [repo](https://github.com/browserslist/caniuse-lite) |
| `electron-to-chromium` | 1.5.371 | build: via `browserslist` | ISC | [repo](https://github.com/Kilian/electron-to-chromium) |
| `ansis` | 4.3.1 | build: via `@tanstack/router-utils` | ISC | [repo](https://github.com/webdiscus/ansis) |
| `graceful-fs`, `lru-cache`, `picocolors`, `semver`, `yallist` | 4.2.11, 5.1.1, 1.1.1, 6.3.1, 3.1.1 | build: via the bundler and Babel | ISC | [graceful-fs](https://github.com/isaacs/node-graceful-fs), [lru-cache](https://github.com/isaacs/node-lru-cache), [picocolors](https://github.com/alexeyraspopov/picocolors), [semver](https://github.com/npm/node-semver), [yallist](https://github.com/isaacs/yallist) |

### Every Python package the deployed app installs

The app's build pins every runtime Python package from `app/uv.lock` (the output of
`uv export --frozen --no-dev --no-emit-project`: 50 lines for 49 packages, because `pandas` has
one version per Python range). Packages that pull a dependency in are named in the Purpose column.

| Package | Version | Purpose | License | Source |
|---|---|---|---|---|
| `annotated-doc` | 0.0.4 | via fastapi | MIT | [repo](https://github.com/fastapi/annotated-doc) |
| `annotated-types` | 0.7.0 | via pydantic | MIT | [repo](https://github.com/annotated-types/annotated-types) |
| `anyio` | 4.13.0 | via starlette | MIT | [repo](https://github.com/agronholm/anyio) |
| `certifi` | 2026.5.20 | via requests (TLS certificates) | MPL-2.0 | [repo](https://github.com/certifi/python-certifi) |
| `cffi` | 2.0.0 | via cryptography | MIT | [repo](https://github.com/python-cffi/cffi) |
| `charset-normalizer` | 3.4.7 | via requests | MIT | [repo](https://github.com/jawah/charset_normalizer) |
| `click` | 8.4.1 | via uvicorn | BSD-3-Clause | [repo](https://github.com/pallets/click) |
| `colorama` | 0.4.6 (Windows only) | via click | BSD-3-Clause | [repo](https://github.com/tartley/colorama) |
| `contourpy` | 1.3.3 | via matplotlib | BSD-3-Clause | [repo](https://github.com/contourpy/contourpy) |
| `cryptography` | 48.0.1 | via google-auth | Apache-2.0 OR BSD-3-Clause | [repo](https://github.com/pyca/cryptography) |
| `cycler` | 0.12.1 | via matplotlib | BSD-3-Clause | [repo](https://github.com/matplotlib/cycler) |
| `databricks-sdk` | 0.117.0 | direct | Apache-2.0 | [repo](https://github.com/databricks/databricks-sdk-py) |
| `fastapi` | 0.136.3 | direct | MIT | [repo](https://github.com/fastapi/fastapi) |
| `fonttools` | 4.63.0 | via matplotlib | MIT | [repo](https://github.com/fonttools/fonttools) |
| `google-auth` | 2.53.0 | via databricks-sdk | Apache-2.0 | [repo](https://github.com/googleapis/google-auth-library-python) |
| `h11` | 0.16.0 | via uvicorn | MIT | [repo](https://github.com/python-hyper/h11) |
| `idna` | 3.18 | via anyio, requests | BSD-3-Clause | [repo](https://github.com/kjd/idna) |
| `joblib` | 1.5.3 | via scikit-learn | BSD-3-Clause | [repo](https://github.com/joblib/joblib) |
| `kiwisolver` | 1.5.0 | via matplotlib | BSD-3-Clause | [repo](https://github.com/nucleic/kiwi) |
| `matplotlib` | 3.10.9 | via neurokit2 | Matplotlib License (PSF-based) | [repo](https://github.com/matplotlib/matplotlib) |
| `narwhals` | 2.22.1 | via scikit-learn | MIT | [repo](https://github.com/narwhals-dev/narwhals) |
| `neurokit2` | 0.2.12 | direct | MIT | [repo](https://github.com/neuropsychology/NeuroKit) |
| `numpy` | 1.26.4 | direct | BSD-3-Clause | [repo](https://github.com/numpy/numpy) |
| `packaging` | 26.2 | via matplotlib | Apache-2.0 OR BSD-2-Clause | [repo](https://github.com/pypa/packaging) |
| `pandas` | 3.0.3 (Python <3.14); 2.3.3 (Python >=3.14) | via neurokit2 | BSD-3-Clause | [repo](https://github.com/pandas-dev/pandas) |
| `pillow` | 12.2.0 | via matplotlib | MIT-CMU | [repo](https://github.com/python-pillow/Pillow) |
| `protobuf` | 6.33.6 | via databricks-sdk | BSD-3-Clause | [repo](https://github.com/protocolbuffers/protobuf) |
| `pyasn1` | 0.6.3 | via pyasn1-modules | BSD-2-Clause | [repo](https://github.com/pyasn1/pyasn1) |
| `pyasn1-modules` | 0.4.2 | via google-auth | BSD-2-Clause | [repo](https://github.com/pyasn1/pyasn1-modules) |
| `pycparser` | 3.0 | via cffi | BSD-3-Clause | [repo](https://github.com/eliben/pycparser) |
| `pydantic` | 2.13.4 | direct | MIT | [repo](https://github.com/pydantic/pydantic) |
| `pydantic-core` | 2.46.4 | via pydantic | MIT | [repo](https://github.com/pydantic/pydantic) |
| `pydantic-settings` | 2.14.1 | direct | MIT | [repo](https://github.com/pydantic/pydantic-settings) |
| `pyparsing` | 3.3.2 | via matplotlib | MIT | [repo](https://github.com/pyparsing/pyparsing) |
| `python-dateutil` | 2.9.0.post0 | via matplotlib, pandas | BSD-3-Clause OR Apache-2.0 | [repo](https://github.com/dateutil/dateutil) |
| `python-dotenv` | 1.2.2 | direct | BSD-3-Clause | [repo](https://github.com/theskumar/python-dotenv) |
| `pytz` | 2026.2 (Python >=3.14) | via pandas | MIT | [repo](https://github.com/stub42/pytz) |
| `PyWavelets` | 1.4.1 | direct | MIT | [repo](https://github.com/PyWavelets/pywt) |
| `requests` | 2.34.2 | via databricks-sdk, neurokit2 | Apache-2.0 | [repo](https://github.com/psf/requests) |
| `scikit-learn` | 1.9.0 | via neurokit2 | BSD-3-Clause | [repo](https://github.com/scikit-learn/scikit-learn) |
| `scipy` | 1.13.1 | direct | BSD-3-Clause | [repo](https://github.com/scipy/scipy) |
| `six` | 1.17.0 | via python-dateutil | MIT | [repo](https://github.com/benjaminp/six) |
| `starlette` | 1.2.1 | direct (also via fastapi) | BSD-3-Clause | [repo](https://github.com/Kludex/starlette) |
| `threadpoolctl` | 3.6.0 | via scikit-learn | BSD-3-Clause | [repo](https://github.com/joblib/threadpoolctl) |
| `typing-extensions` | 4.15.0 | via pydantic and others | PSF-2.0 | [repo](https://github.com/python/typing_extensions) |
| `typing-inspection` | 0.4.2 | via pydantic | MIT | [repo](https://github.com/pydantic/typing-inspection) |
| `tzdata` | 2026.2 (Python >=3.14, Windows or Emscripten) | via pandas | Apache-2.0 | [repo](https://github.com/python/tzdata) |
| `urllib3` | 2.7.0 | via databricks-sdk, requests | MIT | [repo](https://github.com/urllib3/urllib3) |
| `uvicorn` | 0.49.0 | direct | BSD-3-Clause | [repo](https://github.com/Kludex/uvicorn) |
