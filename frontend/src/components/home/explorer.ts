import type { JoinedModelData } from '@/lib/data';
import { INDICES, type IndexId, getIndexTheta, getRunType, scaleTheta, formatTokens, formatPercent, parseParams } from '@/lib/scores';

export type QuestionId = 'capability-safety' | 'frontier' | 'small-vs-large' | 'open-vs-proprietary' | 'reliability-safety' | 'cost';

export interface Question {
  id: QuestionId;
  title: string;
  context: string;
}

export const QUESTIONS: Question[] = [
  {
    id: 'capability-safety',
    title: 'Are the most capable models also the safest?',
    context: 'Each model’s Capability score against its Safety score. Models in the top right are above the panel average on both.',
  },
  {
    id: 'frontier',
    title: 'How are frontier models progressing?',
    context: 'Capability by release date. The orange line tracks the best score released so far; dot area shows total parameters where they are published.',
  },
  {
    id: 'small-vs-large',
    title: 'Which smaller models compete with larger ones?',
    context: 'Capability against total parameter count, on a log scale. Only models with a published parameter count are shown, so proprietary models are excluded.',
  },
  {
    id: 'open-vs-proprietary',
    title: 'How do open-weight and proprietary models compare?',
    context: 'Every model’s score on each index, split into open-weight and proprietary. Short bars mark each group’s median.',
  },
  {
    id: 'reliability-safety',
    title: 'Where do models differ in reliability and safety?',
    context: 'Reliability against Safety. Models can be factual and consistent without avoiding harmful behaviour, and the reverse.',
  },
  {
    id: 'cost',
    title: 'How much does it cost to evaluate a model with CORE?',
    context: 'Tokens used on the full suite against tokens used on CORE, per model, on log scales. Dollar costs need per-model prices, which are not published yet.',
  },
];

export type Weights = 'open' | 'proprietary' | 'unknown';

export const weightsOf = (m: JoinedModelData): Weights =>
  m.metadata?.open_weights === true ? 'open' : m.metadata?.open_weights === false ? 'proprietary' : 'unknown';

// Models without a metadata YAML fall back to their evaluation ID (e.g. gpt-6.1-terra), not the mangled yamlId.
export const nameOf = (m: JoinedModelData) => m.metadata?.name || m.id.split('/').pop() || m.yamlId;

export const indexScore = (m: JoinedModelData, idx: IndexId): number | null => {
  const t = getIndexTheta(m, idx);
  return t ? scaleTheta(t.theta) : null;
};

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

// Tokens are measured for the run the model actually did; the other side is estimated
// from per-item medians, so the source is reported alongside the numbers.
export const tokensOf = (m: JoinedModelData) => {
  const t = m.prediction?.usage?.tokens;
  if (!t || !(t.core > 0) || !(t.full > 0)) return null;
  const run = getRunType(m);
  return { core: t.core, full: t.full, reduction: t.reduction ?? 1 - t.core / t.full, measured: run === 'core' ? 'core' as const : run === 'full' ? 'full' as const : null };
};

export function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function ranks(values: number[]): number[] {
  const order = values.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
  const r = new Array<number>(values.length);
  for (let i = 0; i < order.length;) {
    let j = i;
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++;
    for (let k = i; k <= j; k++) r[order[k][1]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return r;
}

function spearman(xs: number[], ys: number[]): number {
  const rx = ranks(xs);
  const ry = ranks(ys);
  const mx = rx.reduce((a, b) => a + b, 0) / rx.length;
  const my = ry.reduce((a, b) => a + b, 0) / ry.length;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < rx.length; i++) {
    num += (rx[i] - mx) * (ry[i] - my);
    dx += (rx[i] - mx) ** 2;
    dy += (ry[i] - my) ** 2;
  }
  return num / Math.sqrt(dx * dy);
}

function strength(rho: number): string {
  const a = Math.abs(rho);
  if (a < 0.2) return 'barely related';
  const word = a < 0.4 ? 'weakly' : a < 0.6 ? 'moderately' : 'strongly';
  return `${word} ${rho > 0 ? 'positively' : 'negatively'} related`;
}

const fmtDate = (t: number) => new Date(t).toLocaleDateString('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' });
export const fmtParams = (p: number) => p >= 1e12 ? `${(p / 1e12).toFixed(1)}T` : `${Math.round(p / 1e9)}B`;
const r0 = (v: number) => Math.round(v).toString();

function pairFinding(models: JoinedModelData[], a: IndexId, b: IndexId): string | null {
  const pairs = models.flatMap(m => {
    const x = indexScore(m, a);
    const y = indexScore(m, b);
    return x !== null && y !== null ? [{ m, x, y }] : [];
  });
  if (pairs.length < 8) return null;
  const rho = spearman(pairs.map(p => p.x), pairs.map(p => p.y));
  const n = Math.min(10, Math.floor(pairs.length / 4));
  const topA = new Set([...pairs].sort((p, q) => q.x - p.x).slice(0, n).map(p => p.m.id));
  const overlap = [...pairs].sort((p, q) => q.y - p.y).slice(0, n).filter(p => topA.has(p.m.id)).length;
  const [la, lb] = [a, b].map(id => INDICES.find(i => i.id === id)!.label);
  return `Across ${pairs.length} models, ${la} and ${lb} are ${strength(rho)} (Spearman ρ = ${rho.toFixed(2)}). ${overlap} of the ${n} highest-${la} models are also in the top ${n} for ${lb}.`;
}

interface FrontierStep { id: string; name: string; date: number; score: number }

export function capabilityFrontier(models: JoinedModelData[]): FrontierStep[] {
  const dated = models.flatMap(m => {
    const date = releaseOf(m);
    const score = indexScore(m, 'capability');
    return date !== null && score !== null ? [{ id: m.yamlId, name: nameOf(m), date, score }] : [];
  }).sort((p, q) => p.date - q.date || q.score - p.score);
  const steps: FrontierStep[] = [];
  for (const p of dated) if (!steps.length || p.score > steps[steps.length - 1].score) steps.push(p);
  return steps;
}

function frontierFinding(models: JoinedModelData[]): string | null {
  const steps = capabilityFrontier(models);
  if (steps.length < 2) return null;
  const first = steps[0];
  const last = steps[steps.length - 1];
  return `Among the models evaluated, the best Capability score rose from ${r0(first.score)} (${first.name}, ${fmtDate(first.date)}) to ${r0(last.score)} (${last.name}, ${fmtDate(last.date)}). ${steps.length} models set a new high along the way.`;
}

export const LARGE_PARAMS = 100e9;

function sizeFinding(models: JoinedModelData[]): string | null {
  const sized = models.flatMap(m => {
    const p = paramsOf(m);
    const c = indexScore(m, 'capability');
    return p !== null && c !== null ? [{ m, p, c }] : [];
  });
  const large = sized.filter(d => d.p >= LARGE_PARAMS);
  const small = sized.filter(d => d.p < LARGE_PARAMS);
  if (large.length < 3 || small.length < 3) return null;
  const bar = median(large.map(d => d.c));
  const competitive = small.filter(d => d.c >= bar).sort((a, b) => a.p - b.p);
  const best = [...small].sort((a, b) => b.c - a.c)[0];
  if (competitive.length === 0) {
    return `None of the ${small.length} models under 100B parameters reaches the median Capability of the ${large.length} larger models (${r0(bar)}). The closest is ${nameOf(best.m)} (${fmtParams(best.p)}) at ${r0(best.c)}.`;
  }
  const smallest = competitive[0];
  return `${competitive.length} of ${small.length} models under 100B parameters score at or above the median Capability of the ${large.length} larger models (${r0(bar)}). The smallest is ${nameOf(smallest.m)} at ${fmtParams(smallest.p)} (${r0(smallest.c)}).`;
}

function groupFinding(models: JoinedModelData[]): string | null {
  const parts: string[] = [];
  let nProp = 0;
  let nOpen = 0;
  for (const idx of INDICES) {
    const open = models.filter(m => weightsOf(m) === 'open').map(m => indexScore(m, idx.id)).filter(v => v !== null);
    const prop = models.filter(m => weightsOf(m) === 'proprietary').map(m => indexScore(m, idx.id)).filter(v => v !== null);
    if (!open.length || !prop.length) continue;
    nOpen = Math.max(nOpen, open.length);
    nProp = Math.max(nProp, prop.length);
    parts.push(`${idx.label} ${r0(median(prop))} vs ${r0(median(open))}`);
  }
  if (!parts.length) return null;
  return `Median scores, proprietary vs open-weight: ${parts.join('; ')}. Only ${nProp} proprietary models are evaluated against ${nOpen} open-weight models, so treat the gaps as indicative.`;
}

function costFinding(models: JoinedModelData[]): string | null {
  const t = models.map(tokensOf).filter(v => v !== null);
  if (t.length < 3) return null;
  const red = t.map(v => v.reduction);
  const coreMeasured = t.filter(v => v.measured === 'core').length;
  return `Across ${t.length} models, CORE uses a median of ${formatPercent(median(red))} fewer tokens than the full suite (${formatPercent(Math.min(...red))}–${formatPercent(Math.max(...red))}), typically ${formatTokens(median(t.map(v => v.core)))} instead of ${formatTokens(median(t.map(v => v.full)))}. CORE tokens are measured for the ${coreMeasured} models that ran CORE and estimated for the rest.`;
}

export function computeFindings(models: JoinedModelData[]): Record<QuestionId, string | null> {
  return {
    'capability-safety': pairFinding(models, 'capability', 'safety'),
    'frontier': frontierFinding(models),
    'small-vs-large': sizeFinding(models),
    'open-vs-proprietary': groupFinding(models),
    'reliability-safety': pairFinding(models, 'reliability', 'safety'),
    'cost': costFinding(models),
  };
}
