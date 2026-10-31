# Biosignals - Databricks Solution Accelerator

Reference projects for physiological time-series data (ECG and similar biosignals) on Databricks.

[![Databricks](https://img.shields.io/badge/Databricks-Solution_Accelerator-FF3621?style=for-the-badge&logo=databricks)](https://databricks.com)
[![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Enabled-00A1C9?style=for-the-badge)](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
[![Serverless](https://img.shields.io/badge/Serverless-Compute-00C851?style=for-the-badge)](https://docs.databricks.com/en/compute/serverless.html)

> **Synthetic data, not for clinical use.** Every project here runs on synthetic data generated
> by code in the project. The models are demonstrations: they are not clinically validated, are
> not medical devices, and must not be used for diagnosis, treatment or any clinical decision.
> See [Disclaimer](#disclaimer).

## What this is

Biosignals are continuous physiological signals such as electrocardiograms (ECGs). Turning them
into something a care team can act on usually means stitching together many tools: storage for
large waveform arrays, model training, model approval, real-time scoring, batch scoring,
monitoring and an application for clinicians.

Each project in this repo shows that whole path on one governed platform:

- **Data and models in Unity Catalog:** waveform tables, training data, registered models and
  their version aliases, and the request logs of the serving endpoints all live in one catalog,
  with lineage and access control in one place.
- **A model lifecycle with a quality check:** new model versions are registered as candidates
  and only promoted to production after they pass a validation step on held-out data.
- **One model, two ways to score:** the same promoted model version serves real-time requests
  from an app (Model Serving) and scores data in scheduled batch jobs (Lakeflow Jobs).
- **An app and a dashboard on the same data:** a Databricks App for live use and an AI/BI
  dashboard for batch and serving monitoring.
- **Deployable as code:** each project is a Declarative Automation Bundle (formerly Databricks
  Asset Bundles), so the jobs, app and dashboard are created from the repo.

The projects are **reference patterns, not fixed pipelines.** Models, signal types, rhythm
classes, lead counts and catalog/schema names are settings in each project, so the same cycle
adapts to other signals and models.

## Projects

| Project | What it shows |
|---|---|
| [`ecg12-afib`](projects/ecg12-afib/README.md) | Synthetic 12-lead ECGs with synthetic patient records. Two PyTorch models in a cascade: an arrhythmia model screens every ECG and an atrial fibrillation (AFib) model classifies the flagged ones. A setup job generates training data, trains both models, checks them on held-out data, promotes them and serves them; a daily job scores a simulated incoming batch. A Databricks App (React + FastAPI) runs live triage on the endpoints, and an AI/BI dashboard shows batch results and live-serving logs. |

## Repository layout

```
README.md            this file
LICENSE.md           Databricks License
NOTICE.md            copyright, usage notice, how the synthetic data is generated
SECURITY.md          how to report a vulnerability
CONTRIBUTING.md      contributor license agreement
.github/workflows/   CI: validates the bundle of each project changed in a pull request
projects/
  <project>/         one self-contained project
    README.md        what it does, configuration, deploy steps, costs, teardown
    databricks.yml   the project's own bundle (jobs, app, dashboard, variables)
    ...              notebooks, shared code, app source and dashboard definition
```

Each project has its own bundle and is deployed from its own folder. There is no bundle at the
repository root.

## Installation Guidelines

Each project's README lists its prerequisites, the bundle variables to set and the exact
commands. In outline there are two ways to install a project:

**1. From the Databricks workspace (jobs and dashboard).** These steps follow the standard
solution accelerator flow.

1. Clone this repository into your Databricks workspace as a Git folder.

<img width="1726" height="677" alt="Clone the repository into the workspace as a Git folder" src="https://github.com/user-attachments/assets/55b1729f-ad07-420e-a271-843266abfb71" />

2. Open the project's `databricks.yml` (for example `projects/ecg12-afib/databricks.yml`) in the
   Asset Bundle Editor and set the required variables listed in the project README (for
   `ecg12-afib`: `catalog` and `warehouse_id`). For `ecg12-afib` the default target, `no_app`,
   deploys the jobs and the dashboard and needs no local build.

<img width="1120" height="665" alt="Open the Asset Bundle Editor" src="https://github.com/user-attachments/assets/d1f91256-eb8f-4456-8d88-c0a37b1bd4c5" />

3. Click "Deploy".

<img width="1523" height="902" alt="Deploy the bundle" src="https://github.com/user-attachments/assets/9564cbdd-c5c5-4210-bf27-2b19e6efc85b" />

4. In the Deployments tab (🚀 icon), run the project's jobs in the order its README gives (for
   `ecg12-afib`: the setup job first, then the daily job).

<img width="1527" height="880" alt="Run the jobs from the Deployments tab" src="https://github.com/user-attachments/assets/0f612882-7123-449b-8349-1835bc59523c" />

**2. From your laptop (everything, including the app).** A project with a Databricks App needs
the app built locally before it is deployed. For `ecg12-afib`:

```bash
cd projects/ecg12-afib
./deploy.sh -p <your-profile> --catalog <catalog> --warehouse-id <sql-warehouse-id>
```

The script checks the prerequisites, builds the app, deploys the bundle's `full` target, runs
the setup job (about 30 to 60 minutes: GPU training, then endpoint creation), applies the
permissions, starts the app and adds it to the dashboard-embedding allowlist (a workspace admin
setting). Pass your own Databricks CLI profile; nothing in this repo points at a specific
workspace.

The notebooks can also be opened and run one at a time from the Git folder, without deploying
the jobs.

**Costs.** Projects create billable resources: serverless job runs (including GPU for training),
Model Serving endpoints (`ecg12-afib` uses small CPU endpoints with scale-to-zero), a SQL
warehouse for the dashboard and a running Databricks App. Each project README explains how to
remove what it created.

## Contributing

1. **git clone** this project locally.
2. Use the Databricks CLI to test your changes against a Databricks workspace of your choice.
3. Contribute with pull requests (PRs), always with a second-party review from a capable
   teammate.

To add a new project:

- Create `projects/<name>/` (lowercase, hyphens) with its own `databricks.yml` and `README.md`.
- Keep workspace-specific values (hosts, profiles, catalog names, warehouse, dashboard or user
  ids) out of committed files; make them bundle variables.
- Use synthetic or public-domain data only, and explain in `NOTICE.md` how any data in the repo
  was generated.
- List every third-party package your project uses in the license tables below.
- CI validates the bundle of every project a pull request changes. It never deploys.

By contributing you agree to the terms in [CONTRIBUTING.md](CONTRIBUTING.md).

## Disclaimer

This repository is for demonstration and education only.

- **All data is synthetic.** ECG waveforms and patient records are generated by code in each
  project. No real patient data is used, stored or included. Identifiers such as patient IDs
  and record numbers are random and clearly marked as synthetic.
- **Not for clinical use.** The models are trained and evaluated only on that synthetic data.
  They are not clinically validated, are not medical devices, and must not be used for
  diagnosis, treatment or any clinical decision.
- **Metrics are not real-world performance.** The synthetic rhythm classes are separable by
  design, so near-perfect validation metrics show how the data was built, not how a model
  would perform on real recordings.
- **Your compliance stays with you.** If you adapt a project to real data, you are responsible
  for meeting the privacy, security and regulatory requirements that apply (for example HIPAA).

## Project support

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE.md). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support.

## 📄 Third-Party Package Licenses

&copy; 2025 Databricks, Inc. All rights reserved. The source in this project is provided subject to the Databricks License [https://databricks.com/db-license-source]. All included or referenced third party libraries are subject to the licenses set forth below.

### `ecg12-afib`: packages used by the shipped code

Exact versions are pinned in the lockfiles (`projects/ecg12-afib/app/uv.lock` and `bun.lock`),
which the deployed app installs from, and in the notebooks' `%pip install` lines.

| Package | License | Copyright |
|---|---|---|
| apx | Databricks License | Databricks, Inc. |
| fastapi | MIT | Copyright (c) 2018 Sebastián Ramírez |
| uvicorn | BSD-3-Clause | Copyright © 2017-present, Encode OSS Ltd |
| starlette | BSD-3-Clause | Copyright © 2018, Encode OSS Ltd |
| pydantic | MIT | Copyright (c) 2017 to present Pydantic Services Inc. and individual contributors |
| pydantic-settings | MIT | Copyright (c) 2022 Samuel Colvin and other contributors |
| python-dotenv | BSD-3-Clause | Copyright (c) 2014, Saurabh Kumar (python-dotenv), 2013, Ted Tieken (django-dotenv-rw), 2013, Jacob Kaplan-Moss (django-dotenv) |
| databricks-sdk | Apache-2.0 | Copyright 2023 Databricks, Inc. |
| mlflow-skinny | Apache-2.0 | Copyright 2018 Databricks, Inc. |
| numpy | BSD-3-Clause | Copyright (c) 2005-2023, NumPy Developers |
| scipy | BSD-3-Clause | Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers |
| pandas | BSD-3-Clause | Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team; Copyright (c) 2011-2023, Open source contributors |
| matplotlib | Matplotlib License (PSF-based) | Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved |
| neurokit2 | MIT | Copyright (c) 2020, Dominique Makowski |
| PyWavelets | MIT | Copyright (c) 2006-2012 Filip Wasilewski; Copyright (c) 2012-2020 The PyWavelets Developers |
| torch (PyTorch) | BSD-3-Clause | Copyright (c) 2016- Facebook, Inc. (Adam Paszke) and the other contributors listed in its LICENSE |
| pyspark | Apache-2.0 | Copyright 2014 and onwards The Apache Software Foundation |
| react, react-dom | MIT | Copyright (c) Meta Platforms, Inc. and affiliates |
| @tanstack/react-query, @tanstack/react-router | MIT | Copyright (c) 2021-present Tanner Linsley |
| recharts | MIT (its dependency victory-vendor bundles d3 modules under ISC) | Copyright (c) 2015-present recharts |
| sonner | MIT | Copyright (c) 2023 Emil Kowalski |
| clsx | MIT | Copyright (c) Luke Edwards |
| class-variance-authority | Apache-2.0 | Copyright 2022 Joe Bell |
| tailwind-merge | MIT | Copyright (c) 2021 Dany Castillo |
| @radix-ui/react-slot | MIT | Copyright (c) 2022 WorkOS |
| tailwindcss | MIT | Copyright (c) Tailwind Labs, Inc. |
| tw-animate-css | MIT | Copyright (c) 2025 Wombosvideo |

### `ecg12-afib`: build and development tools (not shipped)

| Package | License | Copyright |
|---|---|---|
| bun (installed by apx; builds the UI) | MIT; the binary statically links LGPL-2 JavaScriptCore and LGPL-2.1 tinycc | no copyright line in its LICENSE.md |
| uv | MIT OR Apache-2.0 | Copyright (c) 2025 Astral Software Inc. |
| hatchling | MIT | Copyright (c) 2021-present Ofek Lev |
| uv-dynamic-versioning | MIT | Copyright (c) 2024 Manabu Niseki |
| vite | MIT | Copyright (c) 2019-present, VoidZero Inc. and Vite contributors |
| @vitejs/plugin-react | MIT | Copyright (c) 2019-present, Yuxi (Evan) You and Vite contributors |
| @tailwindcss/vite | MIT | Copyright (c) Tailwind Labs, Inc. |
| @tanstack/router-plugin, @tanstack/router-generator | MIT | Copyright (c) 2021-present Tanner Linsley |
| @opentelemetry/api-logs, exporter-logs-otlp-http, sdk-logs, resources | Apache-2.0 | Copyright The OpenTelemetry Authors |
| typescript | Apache-2.0 | Copyright (c) Microsoft Corporation |
| @types/node, @types/react, @types/react-dom | MIT | Copyright (c) Microsoft Corporation |
| smol-toml | BSD-3-Clause | Copyright (c) Squirrel Chat et al. |
| pytest | MIT | Copyright (c) 2004 Holger Krekel and others |
| cloudpickle | BSD-3-Clause | Copyright (c) 2012-now, CloudPickle developers and contributors; Copyright (c) 2012, Regents of the University of California; Copyright (c) 2009 PiCloud, Inc. |
| pyarrow | Apache-2.0 | Copyright 2016-2024 The Apache Software Foundation |
| PyYAML | MIT | Copyright (c) 2017-2021 Ingy döt Net; Copyright (c) 2006-2016 Kirill Simonov |
| ty | MIT | Copyright (c) 2025 Astral Software Inc. |
