import { LineChart, Line, ResponsiveContainer, YAxis } from "recharts";

// Limb leads computed from I and II (as ECG machines do), shown in italics with their formula.
const DERIVED: Record<string, string> = {
  III: "derived: III = II − I",
  aVR: "derived: aVR = −(I + II) / 2",
  aVL: "derived: aVL = I − II / 2",
  aVF: "derived: aVF = II − I / 2",
};

/**
 * Clinical 12-lead ECG strip: one row per lead (I, II, III, aVR, aVL, aVF, V1–V6). All rows
 * share one millivolt scale, so differences between leads (e.g. the R wave growing from V1 to
 * V5) stay visible. Each lead is downsampled (every `step`-th sample) so recharts stays
 * responsive for 2500-sample leads.
 */
export function EcgChart({
  leads,
  leadNames,
  step = 4,
  leadHeight = 46,
}: {
  leads: number[][];
  leadNames: string[];
  step?: number;
  leadHeight?: number;
}) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const lead of leads) {
    for (const v of lead) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const domain: [number, number] = [lo, hi];

  return (
    <div className="w-full">
      {leads.map((lead, i) => {
        const name = leadNames[i] ?? `ch${i + 1}`;
        const derived = DERIVED[name];
        const data = [];
        for (let j = 0; j < lead.length; j += step) data.push({ v: lead[j] });
        return (
          <div key={name} className="flex items-center gap-2">
            <span
              title={derived ?? "recorded lead"}
              className={`w-8 shrink-0 text-right text-[10px] text-muted-foreground ${derived ? "italic" : ""}`}
            >
              {name}
            </span>
            <div style={{ height: leadHeight }} className="flex-1 rounded border border-border/60">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} margin={{ top: 4, bottom: 4, left: 2, right: 2 }}>
                  <YAxis hide domain={domain} />
                  <Line
                    type="monotone"
                    dataKey="v"
                    stroke="#2272B4"
                    strokeWidth={1}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      })}
      <p className="mt-1 text-[10px] text-muted-foreground">
        All leads on one mV scale. <em>Italic</em> leads are derived from I and II (hover for the
        formula); the other eight are recorded.
      </p>
    </div>
  );
}
