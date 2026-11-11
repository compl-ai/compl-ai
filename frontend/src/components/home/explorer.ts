import type { ExplorerData, JoinedModelData } from '@/lib/data';
import { INDICES, type IndexId, getIndexTheta, parseParams, scaleTheta } from '@/lib/scores';

export type QuestionId = 'capability-safety' | 'manipulation' | 'over-time' | 'parameters' | 'labs' | 'eval-cost';

export interface Question {
  id: QuestionId;
  title: string;
  context: string;
}

export const QUESTIONS: Question[] = [
  {
    id: 'capability-safety',
    title: 'Is the most capable model the safest?',
    context: 'Each model’s Capability score against its Safety or Reliability score. Labelled models sit furthest from the trend, or at the top.',
  },
  {
    id: 'manipulation',
    title: 'Scaling up makes models harder to manipulate, but not more honest and fair',
    context: 'How closely each safety area tracks Capability, with release date held fixed. Click a bar to see its models.',
  },
  {
    id: 'over-time',
    title: 'Capability, reliability and safety over time',
    context: 'Every model’s score by release date. Lines track the best open-weight and best proprietary score so far.',
  },
  {
    id: 'parameters',
    title: 'Total or active parameters: which buys capability?',
    context: 'Capability against active or total parameter count for open-weight models, on a log scale. Mixture-of-experts (MoE) models only use part of their parameters per token. Lines are the trend within each architecture.',
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
    ? `Mostly. ${nameOf(topCap.m)} leads both, and across ${cs.safety.n} models Capability and Safety rank-correlate at ρ = ${f2(cs.safety.rho_release)} with release date held fixed.`
    : `No. The most capable model, ${nameOf(topCap.m)}, ranks ${ordinal(safeRank)} of ${scored.length} on Safety, and the safest, ${nameOf(topSafe.m)}, ranks ${ordinal(capRank)} on Capability. With release date held fixed, Capability predicts Safety (ρ = ${f2(cs.safety.rho_release)}) less well than it predicts Reliability (ρ = ${f2(cs.reliability.rho_release)}), and ${cs.safety.top10_overlap} of the 10 most capable models make the Safety top 10.`;

  const mp = e.manipulation;
  const attack = mp.bars.filter(b => b.group === 'attack');
  const behaviour = mp.bars.filter(b => b.group === 'behaviour');
  const list = (bars: typeof attack) => bars.map(b => `${b.name.toLowerCase()} ${f2(b.rho_release)}`).join(', ');
  const manipulation = `With release date held fixed, more capable models are much better at resisting attacks (${list(attack)}; mean ρ = ${f2(mp.mean.attack)}) than at behaving well unprompted (${list(behaviour)}; mean ρ = ${f2(mp.mean.behaviour)}). Capability buys robustness more than it buys honesty, fairness or refusal.`;

  const ol = e.open_lag;
  const fr = e.frontier;
  const lag = (ix: IndexId) => `${f1(ol[ix].lag_months)} months`;
  const un = ol.safety.oldest_unmatched;
  const overTime = `Open-weight models trail the best proprietary ones by about ${lag('capability')} on Capability and ${lag('reliability')} on Reliability, but ${lag('safety')} on Safety, ${ol.safety.lag_months >= 2 * Math.max(ol.capability.lag_months, ol.reliability.lag_months) ? 'more than twice as long' : 'longer'}. Safety is also where the whole field moves slowest: from ${fmtMonth(fr.window.from)} to ${fmtMonth(fr.window.to)} the best score rose ${signed(fr.capability.delta)} points on Capability and ${signed(fr.reliability.delta)} on Reliability, but only ${signed(fr.safety.delta)} on Safety. The best open Safety score climbs ${f2(ol.safety.open_slope)} points a month, against ${f2(ol.capability.open_slope)} for Capability, so that gap closes slowly.${un ? ` ${name(un.id)}’s Safety score is still unmatched by any open model ${f1(un.months)} months after its release.` : ''} Based on ${ol.n_open} open and ${ol.n_closed} proprietary models.`;

  const pf = e.params.fit;
  const parameters = `With release date held fixed, each 10× in total parameters goes with ${signed(pf.per_10x_total)} Capability points, while 10× in active parameters adds ${signed(pf.per_10x_active)} once total size is accounted for. Active size alone predicts ${signed(pf.per_10x_active_alone)} per 10×, mostly because it tracks total size. Based on ${pf.n} open models, ${pf.n_moe} of them MoE; MoE models are also newer (median ${fmtMonth(pf.median_release_moe)} vs ${fmtMonth(pf.median_release_dense)}).`;

  const ec = e.eval_cost;
  const months = Math.round((Math.log(ec.ratio_usd) / Math.log(ec.trend.full_usd.x_per_year)) * 12);
  const sc = ec.size_control;
  const coreMax = Math.max(...ec.rows.map(r => r.core_usd));
  const evalCost = `Evaluating a newly released model on the full suite costs about ${f1(ec.trend.full_usd.x_per_year)}× more each year. That is not just bigger models: among ${sc.n} open models, with total parameters held fixed, the rise is still ${f1(sc.x_per_year)}× a year, because newer models write more tokens (${f1(ec.trend.full_tokens.x_per_year)}× a year) at higher prices per token. CORE cuts every run by ${f1(ec.ratio_usd)}× (${fmtUsd(ec.totals.core_usd)} vs ${fmtUsd(ec.totals.full_usd)} across ${ec.n} models), about ${months} months of that growth: no CORE run here costs more than ${fmtUsd(coreMax)}, so every new release can be checked as it lands. CORE costs rise too, so keeping evaluation cheap is ongoing work.`;

  const ls = e.labs.safety.labs.filter(l => l.lab !== 'Other').sort((a, b) => b.median - a.median);
  const top = ls.slice(0, 2);
  const bottom = ls.at(-1)!;
  const labs = `Relative to what Capability and release date predict, ${top.map(l => `${l.lab} (median ${signed(l.median)}, ${l.n} models)`).join(' and ')} score highest on Safety and ${bottom.lab} (${signed(bottom.median)}, ${bottom.n} models) lowest. Most labs have only 2–8 models here, so treat this as exploratory.`;

  return {
    'capability-safety': capabilitySafety,
    manipulation,
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
