# Deploy the ECG/AFib accelerator: step-by-step (macOS and Windows, no CLI experience needed)

[← ecg12-afib README](../README.md)

This guide installs the ECG/AFib 12-lead accelerator into **your own** Databricks workspace.
Commands are copy-paste. Replace anything in `<angle brackets>` with your own value.
Where macOS and Windows differ, the step says so.

The documentation links go to Databricks on AWS. On Azure, the same page is at the same path under
`https://learn.microsoft.com/en-us/azure/databricks/`; on Google Cloud, change `/aws/` to `/gcp/` in the
link.

Tested end to end on AWS and on Azure: both jobs (SETUP and DAILY), the two serving endpoints (through
[express deployment](express-deployment.md)), the AI/BI dashboard, and the Databricks App (Triage → Classify, with the dashboard
embedded). Not yet tested on Google Cloud.

To deploy the jobs and dashboard from the workspace instead, with no local tools, see
[Installing a project from the Databricks workspace](../../../docs/installation.md).

Repository: https://github.com/databricks-industry-solutions/biosignals (project folder `projects/ecg12-afib`)

## Words used in this guide

- **Terminal**: the app where you type commands. macOS: Applications > Utilities > Terminal.
  Windows: **PowerShell** (Start menu > type "PowerShell"), or the **Ubuntu** window if you use WSL.
- **WSL** (Windows Subsystem for Linux): a Linux terminal that runs inside Windows. Windows users
  need it for Option B, because the app's deploy script (`deploy.sh`) is a Linux/macOS script.
- **Databricks CLI** (command-line interface): a small program that lets the terminal talk to your
  Databricks workspace.
- **Profile**: a saved login for one workspace. You create it once, then pass it as `-p <profile>`.
- **Catalog / schema**: where the demo's tables and models are stored in Unity Catalog.
- **SQL warehouse**: the compute the dashboard uses.

## How the pieces fit

- **The Databricks CLI** is a remote control for your workspace: you type a command on your
  computer and it does it in Databricks, as you, with your permissions.
- **The bundle** is a recipe: `databricks.yml` (plus the files in `resources/`) lists the 2 jobs,
  the dashboard and the app this demo needs, and their settings (e.g. your catalog). Databricks
  calls it a Declarative Automation Bundle (DAB; formerly "Databricks Asset Bundle").
- **`databricks bundle deploy`** creates or updates those items; running it again updates them in
  place. **`databricks bundle run <job>`** starts a job.
- The jobs make the rest: SETUP creates the training data, trains and registers the 2 models and
  starts the 2 serving endpoints; DAILY writes each day's scored tables.
- **`deploy.sh`** (Option B) runs the bundle commands in the right order, plus the app build and
  setup steps the bundle can't do on its own.
- **You don't edit `databricks.yml` or any config file.** Your values (catalog, warehouse, ...) are
  passed on the command line (step 5).

> **Reference:** [CLI tutorial](https://docs.databricks.com/aws/en/dev-tools/cli/tutorial) · [What are bundles (DABs)](https://docs.databricks.com/aws/en/dev-tools/bundles) · [`bundle` commands](https://docs.databricks.com/aws/en/dev-tools/cli/bundle-commands)

## What you will get

- 2 jobs: **SETUP** (creates synthetic data, trains 2 models, checks them, serves them) and
  **DAILY** (scores one simulated day of ECGs).
- 2 model-serving endpoints and an AI/BI dashboard.
- Optionally, a Databricks App (a web page where you pick an ECG and see it triaged live).
- All data is synthetic. No patient data is involved.

## Choose an option first

| | **Option A: jobs + dashboard** | **Option B: everything, including the app** |
|---|---|---|
| You get | 2 jobs, 2 serving endpoints, the dashboard | all of Option A, plus the Databricks App |
| Where you type (macOS) | Terminal | Terminal |
| Where you type (Windows) | PowerShell | the Ubuntu (WSL) window |
| Tools to install (step 1) | Databricks CLI, Git | Databricks CLI, Git, jq, Python 3, uv, apx |
| Deploy command (step 5) | 5 short commands | 1 script, `./deploy.sh` |
| Pick it if | you want the pipeline, models and dashboard, or are trying it first | you want to show the live app |

The two are separate deployments. To move from A to B later, either remove A first (step 8) or
give B its own schema and endpoint prefix (`--schema <name> --endpoint-prefix <prefix>`).

## Before you start

You need, in your workspace:

- permission to use a catalog and create a schema in it (`USE CATALOG` + `CREATE SCHEMA`);
- serverless jobs, including **serverless GPU** (used for training);
- permission to create model-serving endpoints, and `CAN_USE` on a SQL warehouse;
- for Option B: permission to create apps. One dashboard setting needs a workspace admin
  (step 5, Option B, explains a way around it).

On your computer: macOS, or Windows 10/11; permission to install software (Windows: WSL needs
administrator rights once).

Time: about 15–20 minutes of setup, then about 30–60 minutes while the SETUP job runs on its own.
The demo creates billable resources (GPU training, two endpoints, the app). Step 8 removes it.

> **Reference:** [Unity Catalog privileges](https://docs.databricks.com/aws/en/data-governance/unity-catalog/manage-privileges/privileges) · [Serverless GPU (AI Runtime)](https://docs.databricks.com/aws/en/compute/serverless/gpu) · [Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/)

## Step 1. Install the tools (once)

**Windows, Option B only:** first install WSL. Open PowerShell **as administrator** (right-click
"Windows PowerShell" > Run as administrator), run `wsl --install -d Ubuntu`, restart if asked,
then open **Ubuntu** from the Start menu and choose a Linux user name and password. Install the
tools below **inside Ubuntu**, following each page's **Linux** instructions. All later Option B
commands are typed in the Ubuntu window. See [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

Install each tool your option needs, using its official page (pick your operating system there):

| Tool | Option | Install page |
|---|---|---|
| Databricks CLI (version 0.281.0 or later) | A, B | [Install or update the Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install) |
| Git (downloads the code) | A, B | [git-scm.com/downloads](https://git-scm.com/downloads) |
| jq (used by the deploy script) | B | [jqlang.org/download](https://jqlang.org/download/) |
| Python 3 (used by the deploy script) | B | [python.org/downloads](https://www.python.org/downloads/) (Ubuntu already has it) |
| uv (builds the app's Python package) | B | [uv installation](https://docs.astral.sh/uv/getting-started/installation/) |
| apx **0.3.8 exactly** (builds the app) | B | the line below |

apx's own installer installs the **latest** version by default, but the deploy script accepts only
0.3.8, so install it with this line (macOS Terminal, or the Ubuntu window on Windows):

```bash
curl -fsSL https://databricks-solutions.github.io/apx/install.sh | sh -s -- --version v0.3.8
```

This runs apx's official installer script. If you'd rather not run a downloaded script, download the
file for your computer from the [apx 0.3.8 release page](https://github.com/databricks-solutions/apx/releases/tag/v0.3.8)
instead (`apx-aarch64-darwin` for Apple-silicon Macs, `apx-x86_64-darwin` for Intel Macs,
`apx-x86_64-linux` inside Ubuntu), rename it to `apx`, make it runnable with `chmod +x apx`, and move it
to a folder on your `PATH`, such as `~/.local/bin`. If macOS blocks it the first time, allow it under
System Settings > Privacy & Security.

Close and reopen the terminal, then check:

```bash
databricks --version     # 0.281.0 or later
git --version
# Option B also:
jq --version
python3 --version
uv --version
apx --version            # must print apx 0.3.8
```

> **Reference:** [apx](https://github.com/databricks-solutions/apx) · [Homebrew](https://brew.sh) (the macOS installer several of the pages above use)

## Step 2. Log in to your workspace (once)

```bash
databricks auth login --host https://<your-workspace-url> --profile <profile>
```

- `<your-workspace-url>`: the address in your browser when you're in Databricks, without the
  `https://` and anything after the next `/`. It ends in `.cloud.databricks.com` (AWS),
  `.azuredatabricks.net` (Azure) or `.gcp.databricks.com` (Google Cloud).
- `<profile>`: any short name you choose, e.g. `myws`.

A browser window opens; log in and approve. **Windows, Ubuntu window:** if no browser opens, copy
the link it prints into your Windows browser. Check it worked (prints your user name):

```bash
databricks current-user me -p <profile>
```

On Windows, PowerShell and Ubuntu keep **separate** logins: log in in the one you'll deploy from.

> **Reference:** [CLI authentication](https://docs.databricks.com/aws/en/dev-tools/cli/authentication) · [CLI profiles](https://docs.databricks.com/aws/en/dev-tools/cli/profiles)

## Step 3. Download the code

macOS, or Windows Ubuntu (Option B):

```bash
cd ~
git clone https://github.com/databricks-industry-solutions/biosignals.git
cd biosignals/projects/ecg12-afib
```

On Windows Ubuntu, clone into the Ubuntu home folder (`~`), **not** a Windows folder such as
`/mnt/c/Users/...`: Windows folders can change line endings and file permissions, which breaks
`deploy.sh`.

Windows PowerShell (Option A):

```powershell
cd $HOME
git clone -c core.autocrlf=false https://github.com/databricks-industry-solutions/biosignals.git
cd biosignals\projects\ecg12-afib
```

(`-c core.autocrlf=false` keeps the files exactly as published; Git for Windows otherwise
converts line endings.) Every later command runs from this `projects/ecg12-afib` folder.

> **Reference:** [project README](../README.md)

## Step 4. Find your catalog and warehouse

```bash
databricks catalogs list -p <profile>        # pick a catalog you can write to
databricks warehouses list -p <profile>      # copy the ID of a SQL warehouse
```

Or in the workspace: **SQL Warehouses** > your warehouse > **Overview**. Write down `<catalog>`
and `<warehouse-id>`. The demo creates a schema called `biosignals` in that catalog. If someone in
your workspace already deployed this demo, use your own schema and endpoint prefix (see
"Already deployed" in step 9).

## Step 5. Deploy

Run the commands for the option you chose. Keep each command on **one line** (in PowerShell the
`\` line breaks used on macOS don't work).

### Option A: jobs + dashboard

These commands work as they are in the macOS Terminal and in Windows PowerShell (one line each).

```bash
databricks bundle validate --strict -t no_app -p <profile> --var catalog=<catalog> --var warehouse_id=<warehouse-id>
databricks bundle plan -t no_app -p <profile> --var catalog=<catalog> --var warehouse_id=<warehouse-id>
databricks bundle deploy -t no_app -p <profile> --var catalog=<catalog> --var warehouse_id=<warehouse-id>
databricks bundle run ecg12_afib_setup -t no_app -p <profile> --var catalog=<catalog> --var warehouse_id=<warehouse-id>
databricks bundle run ecg12_afib_daily -t no_app -p <profile> --var catalog=<catalog> --var warehouse_id=<warehouse-id>
```

1. `validate` checks the configuration before anything is created.
2. `plan` lists what `deploy` will add, change or delete, and changes nothing. A first deployment shows
   only items to add; a repeat with nothing new shows `Plan: 0 to add, 0 to change, 0 to delete`. If
   this folder was deployed to a different workspace before, `plan` can stop with
   `lineage mismatch in state files` (see step 9).
3. `deploy` uploads the code and creates the jobs and the dashboard.
4. `run ecg12_afib_setup` runs SETUP and waits (30–60 minutes). It prints a link to watch progress.
5. `run ecg12_afib_daily` runs one simulated day so the dashboard has data.

Then publish the dashboard once more, so its **Ask Genie** works: `deploy` published it before
SETUP and DAILY created its tables. The first command prints the dashboard's id (it needs `jq`; in
PowerShell without `jq`, leave out `-o json | jq ...`: the id is the part of the dashboard's URL
after `/dashboardsv3/`):

```bash
databricks bundle summary -t no_app -p <profile> --var catalog=<catalog> --var warehouse_id=<warehouse-id> -o json | jq -r '.resources.dashboards.ecg12_afib_dashboard.id'
databricks lakeview publish <dashboard-id> --warehouse-id <warehouse-id> --embed-credentials -p <profile>
```

Use the same `<profile>` you deployed with: viewers' queries run with the credentials of whoever
publishes (leave out `--embed-credentials` if you set `embed_credentials` to `false`). Or open the dashboard in the workspace and click **Publish** (upper right), as the same
person. Why, and what we saw in testing: Ask Genie in [the dashboard notes](dashboard.md).

Other settings, all optional, are passed the same way (`--var schema=<name>`, ...); the full list
is in Configuration in [deploy.md](deploy.md#configuration).

Instead of `--var` on every command, you can copy the template once and fill it in: copy
`variable-overrides.example.json` to `.databricks/bundle/no_app/variable-overrides.json`, then
replace `<catalog>` and `<warehouse-id>` in the copy with your values in a text editor.

```bash
mkdir -p .databricks/bundle/no_app
cp variable-overrides.example.json .databricks/bundle/no_app/variable-overrides.json
```

In PowerShell:

```powershell
New-Item -ItemType Directory -Force .databricks\bundle\no_app
Copy-Item variable-overrides.example.json .databricks\bundle\no_app\variable-overrides.json
```

The bundle commands above then work without the `--var` options (for example
`databricks bundle deploy -t no_app -p <profile>`); a `--var` you still pass wins over the file.
Put only variable values in it, never the workspace address, the profile or a password: the
workspace comes from `-p <profile>`. `deploy.sh` (Option B) deploys the `full` target, so it reads
`.databricks/bundle/full/variable-overrides.json` instead, and its options win over that file. More in [Variables in a file instead of `--var`](deploy.md#variables-in-a-file-instead-of---var).

> **Reference:** [bundle variables](https://docs.databricks.com/aws/en/dev-tools/bundles/variables) · [docs/deploy.md, Configuration](deploy.md#configuration)

### Option B: everything, including the app (macOS Terminal or Windows Ubuntu)

```bash
./deploy.sh -p <profile> --catalog <catalog> --warehouse-id <warehouse-id> --run-daily
```

This one script checks the tools, builds the app, validates and plans (`bundle validate`, then
`bundle plan`, which can stop before anything is deployed if this folder's saved state belongs to
another workspace), deploys, runs SETUP (30–60 minutes) and DAILY, gives the app access to the
endpoints, starts the app, and lets the dashboard show inside the app. With `--run-daily` it then
publishes the dashboard once more, so its Ask Genie works; without it, the script prints the
command to run once DAILY has run. It stops with a clear message at the first problem (a failed
publish only warns) and ends with `Done. App: https://...`.

- Not a workspace admin? Add `--skip-embed-allowlist`; the script then prints the one setting an
  admin has to add for you.
- Optional: `--schema`, `--endpoint-prefix`, `--app-name`, and `--remove-after <YYYY-MM-DD>` (tags
  the jobs and endpoints with an expiry date, for workspaces that clean up tagged resources
  automatically). `./deploy.sh --help` lists every option.

> **Reference:** [docs/deploy.md](deploy.md) · [app/README.md](../app/README.md) · [Deploy a Databricks app](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/deploy) · [Embed a dashboard](https://docs.databricks.com/aws/en/dashboards/embed)

## Step 6. Check it worked

In the workspace, no commands needed:

- **Jobs & Pipelines**: the SETUP and DAILY jobs show a green "Succeeded" run.
- **Serving**: `bio_ecg12_arr` and `bio_ecg12_afib` show **Ready**. Each also has a tag
  `deployment`; the expected value on AWS and Azure is `express` (it was in our tests), meaning
  Model Serving used
  [express deployment](https://docs.databricks.com/aws/en/machine-learning/model-serving/express-deployments)
  (no container image build). The SETUP job's `serve_endpoints` task prints the same per endpoint,
  e.g. `bio_ecg12_arr-1: EXPRESS deployment (environment prepared in 2m42s; no container image build)`.
  `standard` means the endpoint built a container image instead (it works, but updates are slower);
  `unknown` means the task could not read or interpret the evidence, which does not mean express was
  not used. Either way the task prints a warning and does not fail. To check by hand, open the
  endpoint's **Logs > Build logs** and pick the served model: an express build log begins with "This
  environment was prepared ahead of time by express deployment". More in "Express deployment" under
  [Batch path vs live path](architecture.md#batch-path-vs-live-path) in the architecture notes, and
  in [Express deployment](express-deployment.md).
- **Catalog**: `<catalog>` > `biosignals` holds tables such as `training_data_12l` and
  `arr_triage_history_12l`, and two models.
- **Dashboards**: "ECG-AFib 12-lead Demo (batch + live)": the batch tiles show data once DAILY has
  run. With Option A its name ends in `[no_app]`. **Ask Genie** on the published dashboard opens
  with suggested questions once the dashboard was published again after DAILY (step 5).
- **Apps** (Option B): open the app, pick a rhythm, click **Triage → Classify**: you should see a
  result and the "Not for clinical use" notice. A brand-new app can show "App Not Available" for a
  few minutes while its compute starts and its code is built; wait and reload the page.
- **Live tiles**: the dashboard's live tiles fill in after the endpoints have had requests:
  Databricks writes the request logs to the inference tables with a delay, and the docs say to expect
  them within an hour of a request
  ([inference tables for serving endpoints](https://docs.databricks.com/aws/en/ai-gateway/inference-tables-serving-endpoints)). Refresh the
  dashboard after that; until the first logs arrive the live tiles show an error, while the batch
  tiles work straight away.

Job and dashboard names start with `[dev <your name>]`. That's expected: the demo deploys in
development mode.

> **Reference:** [bundle deployment modes](https://docs.databricks.com/aws/en/dev-tools/bundles/deployment-modes) · [Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/)

### More than one day on the dashboard

Each DAILY run adds **one** simulated day, and its daily schedule (06:00 UTC) starts **paused**, so
right after setup the dashboard shows a single day. To fill in a trend, run DAILY for past dates.
Find the DAILY job's ID: **Jobs & Pipelines** > the DAILY job > the number in the page address.

macOS Terminal or Windows Ubuntu:

```bash
databricks jobs run-now --json '{"job_id": <daily-job-id>, "job_parameters": {"run_date": "2026-11-17"}}' -p <profile>
```

Windows PowerShell: save the request in a small file first (JSON typed straight into PowerShell
gets its quotes stripped):

```powershell
Set-Content run.json '{"job_id": <daily-job-id>, "job_parameters": {"run_date": "2026-11-17"}}'
databricks jobs run-now --json "@run.json" -p <profile>
```

(the quotes around `"@run.json"` matter in PowerShell). Repeat with each date; wait for one run to
finish before starting the next.

To add a day automatically every morning instead: **Jobs & Pipelines** > the DAILY job >
**Schedules & Triggers** > resume the schedule. It then runs (and bills) once a day; a later deploy
sets it back to paused.

> **Reference:** [Lakeflow Jobs](https://docs.databricks.com/aws/en/jobs/)

## Step 7. Look around

- The notebooks run in order: SETUP = `workflow/01` to `05`, DAILY = `workflow/06` to `08`. Open
  any job run and click a task to see its output.
- `docs/` in the repository explains the architecture, data, model lifecycle and dashboard.

> **Reference:** [architecture](architecture.md) · [data](data.md) · [model lifecycle](model-lifecycle.md) · [dashboard](dashboard.md)

## Step 8. Remove the demo (when you are done)

Follow **Teardown** in `docs/deploy.md`. Its commands are Linux shell commands: run them in the
macOS Terminal, or on Windows in the Ubuntu (WSL) window (for Option A too: install WSL and the
CLI as in step 1, log in as in step 2). It first checks the schema belongs to your deployment, then
removes the jobs, dashboard and app (`databricks bundle destroy`), and then the endpoints, models
and tables.

To stop costs without removing anything: stop the app in **Apps**. The endpoints scale to zero on
their own when not used.

> **Reference:** [docs/deploy.md, Teardown](deploy.md#teardown)

## Step 9. If something goes wrong

| Message | What to do |
|---|---|
| `command not found` / "is not recognized" for a tool | Close and reopen the terminal; if still missing, install it again (step 1) |
| `refresh token is invalid` / not logged in | Run the `databricks auth login ...` line again (step 2) |
| `need apx 0.3.8 exactly` | Re-run the apx line in step 1 (it pins 0.3.8) |
| `deploy.sh`: `bad interpreter` or `\r` errors (Windows) | The code was cloned into a Windows folder; clone again in the Ubuntu home (`cd ~`, step 3) |
| A PowerShell command fails with a JSON or "unexpected token" error | Use the `run.json` file pattern (step 6) |
| App name already taken | Add `--app-name <your-name>` (lowercase, digits, hyphens, max 30 characters) |
| **Already deployed** by someone else: SETUP stops saying the schema belongs to another deployment | Use your own schema **and** endpoint prefix: `--schema <your_schema> --endpoint-prefix <your_prefix>` (Option B), or `--var schema=<your_schema> --var endpoint_prefix=<your_prefix>` on every command (Option A) |
| `Error: lineage mismatch in state files` from `bundle plan` (or `deploy.sh` stopping at step 3) | This folder was deployed to another workspace before, and the deployment state it keeps in `.databricks/bundle/<target>/` (`no_app` or `full`) belongs to that workspace. Use a fresh clone for this workspace (step 3), or delete `.databricks/bundle/<target>/` (move out a `variable-overrides.json` first if you keep one there) and run the commands again |
| `Unable to find dashboard [...]` from `bundle deploy`, in a folder deployed to another workspace before | Same cause and fix as the row above |
| `This dashboard has been modified remotely since the last bundle deployment.` when redeploying | Only when redeploying a deployment whose dashboard was edited in the workspace UI. Option A: run the same `databricks bundle deploy ...` command with `--force` at the end. Option B: run `databricks bundle deploy -t full -p <profile> --var catalog=<catalog> --var warehouse_id=<warehouse-id> --force` (plus any other `--var` you deploy with), then `./deploy.sh` again. `--force` replaces the UI edits with the committed dashboard; to keep them, see [the dashboard notes](dashboard.md) |
| Ask Genie says `Genie is unavailable due to incompatible SQL expressions in the dashboard` | The dashboard was published before the DAILY job created its tables. Run DAILY once if it has not run, then publish the dashboard again (step 5, Option A; [Ask Genie](dashboard.md)) |
| A job task fails with a brief file or network error | Open the failed run in **Jobs & Pipelines** and re-run the failed task (repair the run) |
| Dashboard doesn't show inside the app | A workspace admin adds the app's address under Settings > Security > Embed dashboards |

Re-running any step is safe: everything updates in place.

> **Reference:** [Troubleshoot and repair job failures](https://docs.databricks.com/aws/en/jobs/repair-job-failures) · [Embed a dashboard](https://docs.databricks.com/aws/en/dashboards/embed)
