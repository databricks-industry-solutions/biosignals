import { useState, useEffect, type ReactNode } from "react";
import { createFileRoute } from "@tanstack/react-router";
import {
  useGenerate,
  useAnalyze,
  useAppConfig,
  type GenerateResponse,
  type AnalyzeResponse,
  type AppConfigOut,
  type PatientRecord,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { EcgChart } from "@/components/ecg-chart";

export const Route = createFileRoute("/")({ component: () => <App /> });

const RHYTHMS = ["Random", "NSR (normal)", "AFib", "Other (tachycardia)"];

// Platform-backend deep links (Unity Catalog–governed). Names come from the backend's
// /api/app-config (workspace, catalog/schema and dashboard id from the app's environment, model
// and endpoint names from common/config.py), so nothing is hard-coded here.
// The dashboard links stay undefined when no dashboard id is configured.
function buildLinks(c: AppConfigOut) {
  const host = c.workspace_host;
  const dash = c.dashboard_id;
  return {
    arrEndpoint: `${host}/ml/endpoints/${c.arr_endpoint}`,
    afibEndpoint: `${host}/ml/endpoints/${c.afib_endpoint}`,
    arrModel: `${host}/explore/data/models/${c.catalog}/${c.uc_schema}/${c.arr_model}`,
    afibModel: `${host}/explore/data/models/${c.catalog}/${c.uc_schema}/${c.afib_model}`,
    infTable: `${host}/explore/data/${c.catalog}/${c.uc_schema}/${c.inference_table}`,
    schema: `${host}/explore/data/${c.catalog}/${c.uc_schema}`,
    endpoints: `${host}/ml/endpoints`,
    jobs: `${host}/jobs?o=${c.workspace_id}`,
    apps: `${host}/apps`,
    dashboard: dash ? `${host}/sql/dashboardsv3/${dash}` : undefined,
    dashboardEmbed: dash ? `${host}/embed/dashboardsv3/${dash}?o=${c.workspace_id}` : undefined,
  };
}
type Links = ReturnType<typeof buildLinks>;

const LINK_CLS = "font-medium text-slate-700 underline underline-offset-2 hover:text-slate-900";

/** External link; plain text until the app config (and so the URL) has loaded. */
function Ext({ href, className = LINK_CLS, children }: { href?: string; className?: string; children: ReactNode }) {
  return href ? (
    <a href={href} target="_blank" rel="noreferrer" className={className}>{children}</a>
  ) : (
    <span>{children}</span>
  );
}

function App() {
  const [tab, setTab] = useState<"live" | "batch">("live");
  const config = useAppConfig();
  const links = config.data ? buildLinks(config.data.data) : undefined;
  const alias = config.data?.data.champion_alias;
  return (
    <div className="mx-auto min-h-screen w-full max-w-7xl bg-background px-6 pb-6 pt-10 text-slate-800">
      <header className="mb-6">
        <h1 className="text-2xl font-bold">🫀 ECG/AFib | Biosignals on Databricks — triage → classify</h1>
        <p className="text-sm text-muted-foreground">
          Synthetic 12-lead ECG: the 12 leads shown are the same 12 the batch job stores in{" "}
          <Ext href={links?.schema}>Unity Catalog</Ext> and both models score · an{" "}
          <Ext href={links?.arrModel}><em>arr</em> model</Ext> triages abnormal rhythms · an{" "}
          <Ext href={links?.afibModel}><em>AFib</em> model</Ext> classifies the flagged. Live +
          batch, governed end-to-end.
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          Powered by: <Ext href={links?.schema}>Unity Catalog</Ext> ·{" "}
          <Ext href={links?.endpoints}>MLflow Model Serving</Ext> ·{" "}
          <Ext href={links?.dashboard}>AI/BI</Ext> · <Ext href={links?.jobs}>Lakeflow</Ext> ·{" "}
          <Ext href={links?.apps}>Databricks Apps</Ext>
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          A reference pattern, not a fixed pipeline: the catalog/schema are bundle variables and the
          models, rhythm classes and lead count are set in <code>common/config.py</code>, so the same
          generate → train → validate → promote → serve cycle adapts to other signals and models.
        </p>
      </header>

      <div className="mb-4 flex gap-1 border-b">
        {(["live", "batch"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-base font-semibold ${
              tab === t
                ? "border-b-2 border-slate-700 text-slate-900"
                : "text-muted-foreground hover:text-slate-700"
            }`}
          >
            {t === "live" ? "Live inference" : "Batch & monitoring"}
          </button>
        ))}
      </div>

      {tab === "live" ? <LiveTab links={links} alias={alias} /> : <BatchTab links={links} />}
    </div>
  );
}

function LiveTab({ links, alias }: { links?: Links; alias?: string }) {
  // The served model alias from common/config.py ("@champion"); empty until the config loads.
  const at = alias ? `@${alias}` : "";
  const [rhythm, setRhythm] = useState("Random");
  const [signal, setSignal] = useState<GenerateResponse | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [zoom, setZoom] = useState(false);

  const gen = useGenerate({
    mutation: { onSuccess: (r) => { setSignal(r.data); setAnalysis(null); } },
  });
  const analyze = useAnalyze({ mutation: { onSuccess: (r) => setAnalysis(r.data) } });

  const generate = (rh: string) => gen.mutate({ rhythm: rh });
  // All leads go to both models (the (1, 12, 2500) input contract).
  const runAnalyze = () => signal && analyze.mutate({ ecg: signal.ecg });

  // Auto-populate a random signal on landing so the demo is never empty.
  useEffect(() => {
    generate("Random");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
    {/* Left column at least 20rem, so each patient value fits on one line; the ECG takes the rest. */}
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(20rem,1.3fr)_minmax(0,3fr)_1.6fr]">
      <div className="flex flex-col gap-6">
        {/* 1 - Select */}
        <section className="rounded-lg border p-5">
          <h2 className="mb-2 font-semibold">1 · Select rhythm</h2>
          <select
            className="w-full rounded-md border bg-card px-3 py-2 text-sm"
            value={rhythm}
            onChange={(e) => { setRhythm(e.target.value); generate(e.target.value); }}
          >
            {RHYTHMS.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
          <Button variant="outline" className="mt-2 w-full" onClick={() => generate(rhythm)} disabled={gen.isPending}>
            New sample
          </Button>
          <p className="mt-3 text-xs text-muted-foreground">
            Live samples are generated here and not stored; each model call is logged to its AI
            Gateway inference table in Unity Catalog. Batch ECGs land in Unity Catalog (Delta).
          </p>
          <Ext href={links?.infTable} className="text-xs font-medium text-slate-700 underline underline-offset-2">
            Inspect data lineage in Unity Catalog ↗
          </Ext>
        </section>

        {signal && <PatientCard patient={signal.patient} />}
      </div>

      {/* 2 - Signal */}
      <section className="rounded-lg border p-5">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="font-semibold">2 · ECG signal (12-lead)</h2>
          {signal && (
            <button onClick={() => setZoom(true)} title="Zoom to full 12-lead view"
                    className="rounded border border-slate-300 px-2 py-1 text-sm text-slate-600 hover:bg-slate-50">
              ⤢
            </button>
          )}
        </div>
        {gen.isPending && <p className="text-sm text-muted-foreground">Generating…</p>}
        {signal ? (
          <EcgChart leads={signal.ecg} leadNames={signal.lead_names} />
        ) : (
          !gen.isPending && <p className="text-sm text-muted-foreground">Select a rhythm to generate a signal.</p>
        )}
        <LeadLegend />
      </section>

      {/* 3 - Triage -> Classify + 4 - Action */}
      <section className="rounded-lg border p-5">
        <h2 className="mb-2 font-semibold">3 · Triage → Classify</h2>
        <Button
          variant="secondary"
          className="w-full bg-slate-700 font-semibold text-white hover:bg-slate-800"
          onClick={runAnalyze}
          disabled={!signal || analyze.isPending}
        >
          Triage → Classify
        </Button>
        <p className="mt-1 text-[11px] text-muted-foreground">Model Serving · MLflow {at} · all 12 leads</p>

        {analyze.isPending && (
          <p className="mt-3 text-sm text-muted-foreground">Querying live endpoints (cold start ~1 min)…</p>
        )}
        {analysis && (
          <div className="mt-3 space-y-3">
            <Metric
              label="ARR triage"
              value={analysis.flagged ? "FLAGGED (abnormal)" : "normal"}
              sub={`p=${analysis.arr_prob.toFixed(3)}`}
              link={links?.arrEndpoint}
              linkText={`${analysis.arr_model}${at} ↗`}
            />
            {analysis.flagged ? (
              <Metric
                label="AFib classifier"
                value={analysis.afib_label ?? "-"}
                sub={`p(AFib)=${(analysis.p_afib ?? 0).toFixed(3)}`}
                link={links?.afibEndpoint}
                linkText={`${analysis.afib_model}${at} ↗`}
              />
            ) : (
              <div className="rounded-md bg-muted p-2 text-sm">
                <strong>Triage passed</strong> → AFib model not run.
                <br />Cheap screen first; costly classify runs only when flagged.
              </div>
            )}
            {signal && (
              <p className="text-xs text-muted-foreground">
                Ground truth (simulated): <strong>{signal.label_name}</strong>
              </p>
            )}
            <ActionCard analysis={analysis} />
          </div>
        )}
      </section>
    </div>

    {zoom && signal && (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-6"
           onClick={() => setZoom(false)}>
        <div className="max-h-[92vh] w-full max-w-5xl overflow-auto rounded-lg bg-white p-6"
             onClick={(e) => e.stopPropagation()}>
          <div className="mb-3 flex items-center justify-between">
            <h3 className="font-semibold">12-lead ECG — full view (all 12 leads go to both models)</h3>
            <button onClick={() => setZoom(false)}
                    className="rounded border border-slate-300 px-3 py-1 text-sm hover:bg-slate-50">Close ✕</button>
          </div>
          <EcgChart leads={signal.ecg} leadNames={signal.lead_names} step={2} leadHeight={64} />
        </div>
      </div>
    )}
    </>
  );
}

/**
 * What the 12 leads are, collapsed under the chart. Planes and "12 views" follow the Merck Manual
 * Professional, "Electrocardiography"; recorded vs derived follows common/ecg_signal_gen.py.
 */
function LeadLegend() {
  return (
    <details className="mt-3 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700">
      <summary className="cursor-pointer font-semibold">What are the 12 leads?</summary>
      <ul className="mt-2 list-disc space-y-1 pl-4">
        <li>
          <strong>Limb leads I, II, III</strong> and <strong>augmented limb leads aVR, aVL, aVF</strong>:
          six views in the frontal (vertical) plane, from electrodes on the arms and left leg.
        </li>
        <li>
          <strong>Chest (precordial) leads V1–V6</strong>: six views in the horizontal plane, from
          electrodes across the chest.
        </li>
        <li>
          <strong>Why 12:</strong> each lead views the same heartbeat's electrical activity from a
          different direction, so the beat has a different shape in every lead.
        </li>
        <li>
          <strong>Recorded vs derived:</strong> eight leads are recorded (I, II, V1–V6); III, aVR, aVL
          and aVF are computed from I and II (<em>italic</em> labels, hover for the formula). Both models
          receive all 12.
        </li>
      </ul>
      <p className="mt-2 text-[10px] text-muted-foreground">
        Source: Merck Manual Professional, <em>Electrocardiography</em>; derivation: Einthoven's law and
        Goldberger's augmented leads, as implemented in the demo's signal generator.
      </p>
    </details>
  );
}

function PatientCard({ patient: p }: { patient: PatientRecord }) {
  const acquired = `${new Date(p.acquisition_datetime_utc).toISOString().slice(0, 16).replace("T", " ")} UTC`;
  const rows: [string, string][] = [
    ["MRN", p.mrn],
    ["Age · sex", `${p.age_years} y · ${p.sex}`],
    ["Height · weight", `${p.height_cm} cm · ${p.weight_kg} kg`],
    ["BMI", `${p.bmi} kg/m²`],
    ["Heart rate", `${p.heart_rate_bpm} bpm`],
    ["Blood pressure", `${p.bp_systolic_mmhg}/${p.bp_diastolic_mmhg} mmHg`],
    ["Acquired", acquired],
    ["Recording", `${p.n_leads} leads · ${p.sampling_rate_hz} Hz · ${p.duration_s} s`],
  ];
  return (
    <section className="rounded-lg border p-5">
      <h2 className="font-semibold">Patient context</h2>
      <p className="mb-2 text-[11px] text-muted-foreground">Synthetic, non-PII · {p.pseudo_patient_id}</p>
      <dl className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-1 text-xs">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="whitespace-nowrap text-muted-foreground">{k}</dt>
            <dd className="whitespace-nowrap font-medium">{v}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function ActionCard({ analysis }: { analysis: AnalyzeResponse }) {
  const isAfib = analysis.flagged && analysis.afib_label === "AFib";
  const action = !analysis.flagged
    ? "Routine — no action"
    : isAfib
      ? "Route to cardiology review + anticoagulation assessment"
      : "Route to general review queue";
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-3 text-sm">
      <div className="text-xs font-semibold text-muted-foreground">4 · Recommended action</div>
      <div className="mt-1 font-medium">{action}</div>
    </div>
  );
}

function BatchTab({ links }: { links?: Links }) {
  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <p className="text-sm text-muted-foreground">
        Synthetic 12-lead ECGs and their synthetic patient records <strong>land in Unity Catalog</strong>{" "}
        → <strong>arr triage</strong> → <strong>AFib classify</strong>, all governed. The 12-lead
        AI/BI dashboard (embedded below once configured) has two separate sections.{" "}
        <strong>Batch monitoring:</strong> the daily job scores with the champion models loaded from
        Unity Catalog (daily funnel, arrivals per hour, triage by rhythm, AFib-probability
        distribution, multi-day trend, patient context). <strong>Live serving:</strong> this app's
        endpoint calls, from the endpoints' AI Gateway inference tables.
      </p>
      {!links ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : links.dashboardEmbed ? (
        <>
          <div className="flex justify-end">
            <Ext href={links.dashboard} className="text-xs font-medium text-slate-600 hover:text-slate-900">
              Open in Databricks ↗
            </Ext>
          </div>
          <iframe title="ECG AI/BI dashboard" src={links.dashboardEmbed} className="h-[1100px] w-full rounded-lg border" />
        </>
      ) : (
        <DashboardNotConfigured />
      )}
    </div>
  );
}

/** Shown instead of an embed when the app has no dashboard id (ECG_DEMO_REACT_DASHBOARD_ID). */
function DashboardNotConfigured() {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-6 text-sm">
      <div className="font-semibold">Dashboard not configured</div>
      <p className="mt-2 text-muted-foreground">
        This app embeds the AI/BI dashboard whose id is in its <code>ECG_DEMO_REACT_DASHBOARD_ID</code>{" "}
        environment variable. The bundle's <code>full</code> target deploys the dashboard and sets that
        variable (<code>databricks.yml</code>); redeploy the app with it to show the dashboard here.
      </p>
    </div>
  );
}

function Metric({ label, value, sub, link, linkText }: {
  label: string; value: string; sub: string; link?: string; linkText?: string;
}) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-xl font-bold">{value}</div>
      <div className="text-xs text-slate-500">{sub}</div>
      {linkText && (
        <Ext href={link} className="text-xs font-medium text-slate-700 underline underline-offset-2">
          {linkText}
        </Ext>
      )}
    </div>
  );
}
