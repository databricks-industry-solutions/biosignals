# Changelog

Notable changes to the ECG/AFib 12-lead project. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- Training and the validation gate run on serverless AI Runtime environment 6; serving and the DAILY
  job run on serverless environment 6.
- Models are logged with explicit serving requirements (exact versions) that avoid what blocks
  Model Serving's [express deployment](https://docs.databricks.com/aws/en/machine-learning/model-serving/express-deployments);
  on AWS and Azure the endpoints used it in our tests.
- `deploy.sh` runs `bundle plan` before deploying, and takes its values from the bundle (flags,
  `BUNDLE_VAR_<name>`, `variable-overrides.json` or defaults), so its checks match what is deployed.
- SETUP's serving step skips an endpoint that already serves the champion version, and reports
  whether each endpoint used express deployment (endpoint tag `deployment`).

### Added
- [Quickstart](docs/quickstart.md) for macOS and Windows, and [express deployment](docs/express-deployment.md) notes.
- `variable-overrides.example.json` template for bundle variables.
- The app's title links to this project's source.

### Fixed
- Markdown that rendered wrong on GitHub (strikethrough from `~`, hidden `ARRAY<FLOAT>`).
- SETUP's serving step no longer waits out its 50-minute timeout when an endpoint update that
  reuses the config version fails before the wait sees it start: it reads the endpoint's events and
  stops with Model Serving's message.
- The dashboard's Ask Genie said "Genie is unavailable due to incompatible SQL expressions in the
  dashboard" after a first deployment (the dashboard was published before DAILY created its
  tables). `deploy.sh --run-daily` now publishes it again at the end; the docs give the command for
  the other paths ([Ask Genie](docs/dashboard.md)).

### Tested
End to end on AWS and on Azure: SETUP and DAILY jobs, both serving endpoints (express deployment),
the AI/BI dashboard and the Databricks App. Not yet tested on Google Cloud.

### Upgrading
1. Run SETUP before DAILY: DAILY loads the models with the new package versions.
2. SETUP moves each endpoint to the new `champion` version; "Until the new configuration is ready,
   the old configuration keeps serving prediction traffic"
   ([Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints)).
3. Every SETUP run sets the endpoint tag `deployment` again, overwriting the previous value.
4. To roll back, set each model's `champion` alias back to its previous version (both models: the
   gate promotes both or neither), then run `workflow/05_serve_endpoints` on its own with the same
   parameters; it serves whatever `champion` points to. DAILY scores `champion` too. Running the whole
   SETUP job instead would retrain and re-gate. A rollback changes only later scoring (live and
   DAILY): it does not rewrite tables already written or the inference tables' history.

## 2026-11-19 — hardening ([#5](https://github.com/databricks-industry-solutions/biosignals/pull/5))
Schema ownership checks, not-for-clinical-use notice, CI hardening, rate limits, dependency updates.

## 2026-11-04 — first release
The ecg12-afib project: SETUP and DAILY jobs, two serving endpoints, AI/BI dashboard, Databricks App.
