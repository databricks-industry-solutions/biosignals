# Installing a project from the Databricks workspace

[← Biosignals README](../README.md)

These steps deploy a project's jobs and dashboard from the workspace, with no local tools. Each
project's README lists the variables to set and any extra steps (for example a local build for a
Databricks App).

To deploy from your own computer with the Databricks CLI instead (macOS or Windows, including
the app), see the ECG/AFib [step-by-step quickstart](../projects/ecg12-afib/docs/quickstart.md).

1. Clone this repository into your Databricks workspace as a Git folder.

<img width="1726" height="677" alt="Clone the repository into the workspace as a Git folder" src="https://github.com/user-attachments/assets/55b1729f-ad07-420e-a271-843266abfb71" />

2. Set the variables the project's README lists as required, without editing the committed
   `databricks.yml`: in the project's folder, create `.databricks/bundle/<target>/variable-overrides.json`
   with their values, for example `{"catalog": "<catalog>", "warehouse_id": "<warehouse-id>"}`. If
   the project has a `variable-overrides.example.json`, start from its content and replace the
   placeholders. Put only variable values in the file, never the workspace address or a secret.
   The bundle reads this file for that target, and `.databricks/` is in the repository's
   `.gitignore`, so it never ends up in a commit. Then open the project's `databricks.yml` in the
   Asset Bundle Editor. If the editor still reports a required variable without a value, set the
   values under `targets.<target>.variables` in your Git folder's copy of `databricks.yml` instead,
   and do not commit that change.

<img width="1120" height="665" alt="Open the Asset Bundle Editor" src="https://github.com/user-attachments/assets/d1f91256-eb8f-4456-8d88-c0a37b1bd4c5" />

3. Click "Deploy".

<img width="1523" height="902" alt="Deploy the bundle" src="https://github.com/user-attachments/assets/9564cbdd-c5c5-4210-bf27-2b19e6efc85b" />

4. In the Deployments tab (🚀 icon), run the project's jobs in the order its README gives.

<img width="1527" height="880" alt="Run the jobs from the Deployments tab" src="https://github.com/user-attachments/assets/0f612882-7123-449b-8349-1835bc59523c" />
