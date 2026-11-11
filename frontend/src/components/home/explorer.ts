import type { ExplorerData, JoinedModelData } from '@/lib/data';
import { INDICES, type IndexId, getIndexTheta, parseParams, scaleTheta } from '@/lib/scores';

export type QuestionId = 'capability-safety' | 'safety-areas' | 'over-time' | 'parameters' | 'labs' | 'eval-cost';

export interface Question {
  id: QuestionId;
  title: string;
  context: string | ((e: ExplorerData) => string);
}

export const contextOf = (q: Question, e: ExplorerData) => (typeof q.context === 'function' ? q.context(e) : q.context);

export const QUESTIONS: Question[] = [
  {
    id: 'over-time',
    title: 'Capability, reliability and safety over time',
    context: e => `Every model’s score by release date (${e.open_lag.n_open} open-weight, ${e.open_lag.n_closed} proprietary). Lines track the best score so far for each.`,
  },
  {
    id: 'capability-safety',
    title: 'Is the most capable model the safest?',
    context: 'Each model’s Capability score against its Safety or Reliability score.',
  },
  {
    id: 'safety-areas',
    title: 'Which parts of safety are improving?',
    context: 'Predicted accuracy by release date for three groups of safety areas, each with a linear trend. Areas that tend to move together are grouped.',
  },
  {
    id: 'parameters',
    title: 'Total or active parameters: which buys capability?',
    context: 'Mixture-of-experts (MoE) models only use part of their parameters per token. Lines are the trend within each architecture, on a log scale.',
  },
  {
    id: 'labs',
    title: 'How do labs compare on safety?',
    context: 'How far each model’s score sits above or below what its Capability and release date predict, grouped by lab. Ticks mark each lab’s median.',
  },
  {
    id: 'eval-cost',
    title: 'The rising cost of evaluation',
    context: 'What one full-suite run and one CORE run cost for each model, by release date.',
  },
];

export type Weights = 'open' | 'proprietary' | 'unknown';

export const weightsOf = (m: JoinedModelData): Weights =>
  m.metadata?.open_weights === true ? 'open' : m.metadata?.open_weights === false ? 'proprietary' : 'unknown';

// Models without a metadata YAML fall back to their evaluation ID (e.g. gpt-6.1-terra), not the mangled yamlId.
export const nameOf = (m: JoinedModelData) => m.metadata?.name || m.id.split('/').pop() || m.yamlId;

// Compact chart label: drops vendor prefixes and packaging suffixes that add width but not identity.
export const shortName = (m: JoinedModelData) =>
  nameOf(m).replace(/^NVIDIA-/, '').replace(/[- ](Instruct|it)( \d{4})?$/i, '').replace(/-BF16$/, '');

export const indexScore = (m: JoinedModelData, idx: IndexId): number | null => {
  const t = getIndexTheta(m, idx);
  return t ? scaleTheta(t.theta) : null;
};

export const indexLabel = (id: IndexId) => INDICES.find(i => i.id === id)!.label;

export const paramsOf = (m: JoinedModelData): number | null => {
  const p = parseParams(m.metadata?.specs?.total_params);
  return p > 0 && !isNaN(p) ? p : null;
};

export const SIZE_BUCKETS = [
  { id: 'small', label: 'Under 10B', test: (p: number) => p > 0 && p < 10e9 },
  { id: 'medium', label: '10B–100B', test: (p: number) => p >= 10e9 && p < 100e9 },
  { id: 'large', label: '100B and up', test: (p: number) => p >= 100e9 },
  { id: 'unknown', label: 'Unknown size', test: (p: number) => !p || isNaN(p) },
] as const;

export type SizeBucket = (typeof SIZE_BUCKETS)[number]['id'];

export function sizeBucketOf(totalParams: unknown): SizeBucket {
  const p = parseParams(totalParams);
  return SIZE_BUCKETS.find(b => b.test(p))!.id;
}

export const releaseOf = (m: JoinedModelData): number | null => {
  const d = m.metadata?.release_date ? Date.parse(m.metadata.release_date) : NaN;
  return isNaN(d) ? null : d;
};

export function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export const fmtParams = (p: number) => p >= 1e12 ? `${+(p / 1e12).toFixed(1)}T` : p >= 1e9 ? `${+(p / 1e9).toPrecision(3)}B` : `${+(p / 1e6).toPrecision(3)}M`;
export const fmtMonth = (t: number | string) => new Date(t).toLocaleDateString('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' });
export const fmtUsd = (v: number) => v >= 100 ? `$${Math.round(v).toLocaleString('en-US')}` : v >= 10 ? `$${v.toFixed(0)}` : `$${v.toFixed(2)}`;
const f1 = (v: number) => v.toFixed(1);
const f2 = (v: number) => v.toFixed(2);
const signed = (v: number) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${f1(Math.abs(v))}`;

// Best score released up to each date: each step is a model that set a new high.
export function frontierSteps<T extends { date: number; score: number }>(points: T[]): T[] {
  const sorted = [...points].sort((a, b) => a.date - b.date || b.score - a.score);
  const steps: T[] = [];
  for (const p of sorted) if (!steps.length || p.score > steps[steps.length - 1].score) steps.push(p);
  return steps;
}

export type ModelLookup = Map<string, JoinedModelData>;
export const lookupModels = (models: JoinedModelData[]): ModelLookup => new Map(models.map(m => [m.id, m]));

// Display name for an explorer id; ids without a joined model fall back to their last path segment.
export const nameFor = (byId: ModelLookup, id: string) => {
  const m = byId.get(id);
  return m ? nameOf(m) : id.split('/').pop() ?? id;
};

export const ols = (xs: number[], ys: number[]) => {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0;
  let sxx = 0;
  xs.forEach((x, i) => { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) ** 2; });
  const b = sxx ? sxy / sxx : 0;
  return { a: my - b * mx, b };
};

// Per-year slope of y on date, with log size held fixed (two-regressor OLS).
function dateSlope(logSize: number[], years: number[], ys: number[]) {
  const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
  const [ms, mt, my] = [mean(logSize), mean(years), mean(ys)];
  let sss = 0, stt = 0, sst = 0, ssy = 0, sty = 0;
  ys.forEach((y, i) => {
    const s = logSize[i] - ms, t = years[i] - mt, d = y - my;
    sss += s * s; stt += t * t; sst += s * t; ssy += s * d; sty += t * d;
  });
  const det = sss * stt - sst * sst;
  return det ? (sss * sty - sst * ssy) / det : 0;
}

const YEAR = 365.25 * 864e5;

// Capability shifted to the median release date along the pooled date trend (size held fixed),
// so older models are not mistaken for weaker architectures; then a log10(total) trend per architecture.
export function paramsFit(e: ExplorerData, byId: ModelLookup) {
  const base = e.params.points.flatMap(p => {
    const m = byId.get(p.id);
    const d = m ? releaseOf(m) : null;
    return m && d !== null ? [{ m, ...p, date: d }] : [];
  });
  const dates = base.map(r => r.date).sort((a, b) => a - b);
  const refDate = dates[Math.floor(dates.length / 2)];
  const perYear = dateSlope(base.map(r => Math.log10(r.total)), base.map(r => r.date / YEAR), base.map(r => r.capability));
  const rows = base.map(r => ({ ...r, x: r.total, raw: r.capability, capability: r.capability - (perYear * (r.date - refDate)) / YEAR }));
  const trend = (moe: boolean) => {
    const g = rows.filter(r => r.moe === moe);
    if (g.length < 3) return null;
    const { a, b } = ols(g.map(r => Math.log10(r.x)), g.map(r => r.capability));
    return { a, b, n: g.length, x0: Math.min(...g.map(r => r.x)), x1: Math.max(...g.map(r => r.x)) };
  };
  return { rows, refDate, perYear, dMin: dates[0], dMax: dates[dates.length - 1], trends: { true: trend(true), false: trend(false) } };
}

export function buildFindings(e: ExplorerData, models: JoinedModelData[]): Record<QuestionId, string> {
  const byId = lookupModels(models);
  const name = (id: string) => nameFor(byId, id);

  const cs = e.capability_safety;
  const scored = models.flatMap(m => {
    const c = indexScore(m, 'capability');
    const s = indexScore(m, 'safety');
    return c !== null && s !== null ? [{ m, c, s }] : [];
  });
  const topCap = scored.reduce((a, b) => (b.c > a.c ? b : a));
  const topSafe = scored.reduce((a, b) => (b.s > a.s ? b : a));
  const safeRank = [...scored].sort((a, b) => b.s - a.s).findIndex(d => d.m.id === topCap.m.id) + 1;
  const capRank = [...scored].sort((a, b) => b.c - a.c).findIndex(d => d.m.id === topSafe.m.id) + 1;
  const capabilitySafety = topCap.m.id === topSafe.m.id
    ? `Mostly. ${nameOf(topCap.m)} leads both, and across ${cs.safety.n} models Capability and Safety rank-correlate at ρ = ${f2(cs.safety.rho_release)}.`
    : `No, the most capable model is not the safest. ${nameOf(topCap.m)} ranks ${ordinal(safeRank)} of ${scored.length} on Safety, and the safest, ${nameOf(topSafe.m)}, ranks ${ordinal(capRank)} on Capability. Capability predicts Safety (ρ = ${f2(cs.safety.rho_release)}) less well than it predicts Reliability (ρ = ${f2(cs.reliability.rho_release)}), and ${cs.safety.top10_overlap} of the 10 most capable models make the Safety top 10.`;

  const sg = Object.fromEntries(e.safety_areas.groups.map(g => [g.id, g]));
  const ppy = (g: { pp_per_year: number }) => `${signed(g.pp_per_year)} percentage points a year`;
  const trendPct = (g: { trend: { y: number }[] }, k: number) => `${Math.round(g.trend[k].y)}%`;
  const inj = sg.injection, ref = sg.refusal, fair = sg.fairness;
  const safetyAreas = `Resistance to injection attacks is improving fastest, ${ppy(inj)}, from ${trendPct(inj, 0)} on trend in ${fmtMonth(inj.trend[0].date)} to ${trendPct(inj, 1)} in ${fmtMonth(inj.trend[1].date)}. Refusal and honesty (${signed(ref.pp_per_year)}) and fairness (${signed(fair.pp_per_year)}) improve far more slowly.`;

  const ol = e.open_lag;
  const lag = (ix: IndexId) => `${f1(ol[ix].lag_months)} months`;
  const un = ol.safety.oldest_unmatched;
  const overTime = `Open-weight models trail the best proprietary ones by about ${lag('capability')} on Capability and ${lag('reliability')} on Reliability. On Safety the gap appears longer, about ${lag('safety')}${un ? `, and ${name(un.id)}’s Safety score is still unmatched by an open model` : ''}. These are projections from today’s gap and last year’s open-model progress, based on only ${ol.n_closed} proprietary models.`;

  const { trends } = paramsFit(e, byId);
  const moe = trends.true;
  const dense = trends.false;
  let parameters = '';
  if (moe && dense) {
    const at = (t: typeof moe, x: number) => t.a + t.b * Math.log10(x);
    const REF = 30e9;
    const gap = at(dense, REF) - at(moe, REF);
    const cross = moe.b !== dense.b ? 10 ** ((dense.a - moe.a) / (moe.b - dense.b)) : null;
    parameters = `MoE models gain more from scale: each 10× in total parameters adds ${signed(moe.b)} Capability points, against ${signed(dense.b)} for dense models, while using only part of those parameters per token. In absolute terms MoE still trails at the same total size: about ${f1(Math.abs(gap))} points ${gap > 0 ? 'behind' : 'ahead'} at ${fmtParams(REF)}${cross && cross > dense.x1 ? `, with the trends meeting only around ${fmtParams(+cross.toPrecision(1))}, beyond the largest dense model here (${fmtParams(dense.x1)})` : ''}.`;
  }

  const ec = e.eval_cost;
  const evalCost = `Evaluation gets about ${Math.round(ec.trend.full_usd.x_per_year)}× more expensive every year, and CORE cuts each run by about ${Math.round(ec.ratio_usd)}×. Newer models write more tokens at higher prices, so a full-suite run keeps climbing while CORE rises from a far lower level, keeping evaluation of new foundation models cheap, feasible and reproducible.`;

  const ls = e.labs.safety.labs.filter(l => l.lab !== 'Other').sort((a, b) => b.median - a.median);
  const top = ls.slice(0, 2);
  const bottom = ls.at(-1)!;
  const lr = e.labs.reliability.labs.filter(l => l.lab !== 'Other');
  const near = lr.filter(l => Math.abs(l.median) <= 5).length;
  const labs = `${top.map(l => l.lab).join(' and ')} lead on Safety, ${bottom.lab} trails. Medians: ${[...top, bottom].map(l => `${l.lab} ${signed(l.median)} (${l.n} models)`).join(', ')}. Our findings are limited to 2–8 models per lab currently. Reliability shows a less clear pattern: ${near} of ${lr.length} labs sit within ±5 points, and the Safety leaders do not carry over (${top.map(l => `${l.lab} ${signed(lr.find(r => r.lab === l.lab)?.median ?? 0)}`).join(', ')}).`;

  return {
    'capability-safety': capabilitySafety,
    'safety-areas': safetyAreas,
    'over-time': overTime,
    parameters,
    labs,
    'eval-cost': evalCost,
  };
}

function ordinal(n: number): string {
  const s = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`;
}
