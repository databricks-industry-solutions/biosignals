# Installing a project from the Databricks workspace

[← Biosignals README](../README.md)

These steps deploy a project's jobs and dashboard from the workspace, with no local tools. Each
project's README lists the variables to set and any extra steps (for example a local build for a
Databricks App).

1. Clone this repository into your Databricks workspace as a Git folder.

<img width="1726" height="677" alt="Clone the repository into the workspace as a Git folder" src="https://github.com/user-attachments/assets/55b1729f-ad07-420e-a271-843266abfb71" />

2. Open the project's `databricks.yml` in the Asset Bundle Editor and set the variables its
   README lists as required.

<img width="1120" height="665" alt="Open the Asset Bundle Editor" src="https://github.com/user-attachments/assets/d1f91256-eb8f-4456-8d88-c0a37b1bd4c5" />

3. Click "Deploy".

<img width="1523" height="902" alt="Deploy the bundle" src="https://github.com/user-attachments/assets/9564cbdd-c5c5-4210-bf27-2b19e6efc85b" />

4. In the Deployments tab (🚀 icon), run the project's jobs in the order its README gives.

<img width="1527" height="880" alt="Run the jobs from the Deployments tab" src="https://github.com/user-attachments/assets/0f612882-7123-449b-8349-1835bc59523c" />
