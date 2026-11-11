import type { IndexTheta, JoinedModelData, LabelScore, RunType, TaskStatus } from '@/lib/data';

// θ is standardized on the model panel (mean 0, SD 1). It is published on an
// IQ-style scale: 100 = panel average, 15 points = one standard deviation.
export const THETA_MEAN = 100;
const THETA_SD = 15;

export const INDICES = [
  { id: 'capability', label: 'Capability', description: 'Measures the model’s ability to successfully perform challenging tasks across reasoning, knowledge, mathematics, coding, languages and modalities.' },
  { id: 'reliability', label: 'Reliability', description: 'Measures whether the model produces factual, consistent and instruction-following outputs that remain stable under perturbation.' },
  { id: 'safety', label: 'Safety', description: 'Measures the model’s propensity to avoid harmful, deceptive, adversarial and discriminatory behavior.' },
] as const;

export type IndexId = (typeof INDICES)[number]['id'];

// Static class names keyed by index, so Tailwind can see them.
export const INDEX_TONE: Record<IndexId, { bar: string; soft: string; line: string }> = {
  capability: { bar: 'bg-capability', soft: 'bg-brand-soft', line: 'border-capability' },
  reliability: { bar: 'bg-reliability', soft: 'bg-teal-soft', line: 'border-reliability' },
  safety: { bar: 'bg-safety', soft: 'bg-lavender-soft', line: 'border-safety' },
};

// Interval band and point colours for score bars (leaderboard and model page).
export const INDEX_BAR: Record<IndexId, { band: string; dot: string }> = {
  capability: { band: 'bg-capability/45', dot: 'bg-capability' },
  safety: { band: 'bg-lavender-strong/45', dot: 'bg-lavender-strong' },
  reliability: { band: 'bg-teal-strong/45', dot: 'bg-teal-strong' },
};

export function getIndexTheta(m: JoinedModelData, index: string): IndexTheta | null {
  return m.prediction?.indices?.[index]?.index_theta ?? null;
}

export function scaleTheta(theta: number): number {
  return THETA_MEAN + THETA_SD * theta;
}

// Bar position (0–100%) for a scaled score. 50–100 is squeezed into the left quarter so the crowded range above the average gets more room.
const BAR_LOW = 50;
const BAR_HIGH = 150;
const BAR_SPLIT = 25;
export function scoreBarPct(score: number): number {
  const v = Math.min(Math.max(score, BAR_LOW), BAR_HIGH);
  return v <= THETA_MEAN
    ? (v - BAR_LOW) / (THETA_MEAN - BAR_LOW) * BAR_SPLIT
    : BAR_SPLIT + (v - THETA_MEAN) / (BAR_HIGH - THETA_MEAN) * (100 - BAR_SPLIT);
}

export function formatScore(theta: number): string {
  return Math.round(scaleTheta(theta)).toString();
}

export function formatInterval(interval: readonly [number, number] | number[]): string {
  return `${formatScore(interval[0])}–${formatScore(interval[1])}`;
}

export const NO_SCORE_REASON = "Too few of this index's benchmarks were run to estimate a score.";

export const RUN_TYPES: Record<RunType, { label: string; short: string; description: string; className: string }> = {
  core: {
    label: 'CORE subset',
    short: 'CORE',
    description: 'Evaluated on the compact CORE subset only; scores are estimated from those items.',
    className: 'bg-teal-soft text-teal-strong border-teal/60',
  },
  full: {
    label: 'Calibration',
    short: 'Calibration',
    description: 'Ran every item of the benchmarks it was evaluated on, though usually not every benchmark. These runs form the calibration set the CORE subset is fitted on.',
    className: 'bg-lavender-soft text-lavender-strong border-lavender/50',
  },
  mixed: {
    label: 'Mixed',
    short: 'Mixed',
    description: 'Some benchmarks were run on all items and others on the CORE subset only.',
    className: 'bg-muted text-muted-foreground border-border',
  },
};

// Falls back to coverage for exports without usage data: a CORE-only run observed no
// population items beyond the CORE items it answered.
export function getRunType(m: JoinedModelData): RunType | null {
  const usage = m.prediction?.usage;
  if (usage) return usage.run_type;
  const cov = m.prediction?.coverage;
  if (!cov) return null;
  return cov.population_samples_observed > cov.samples_completed ? 'full' : 'core';
}

export function formatTokens(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}k`;
  return n.toString();
}

// Parameter counts may arrive as numbers or as digit-grouped strings like "70_000_000_000".
export const parseParams = (v: unknown) => typeof v === 'number' ? v : Number(String(v ?? '0').replace(/_/g, ''));

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
}

// The span of a set of ISO dates, e.g. "Nov 5–6, 2026" or "Mar 20 – Apr 13, 2026".
export function formatDateRange(isos: string[]): string {
  if (!isos.length) return '—';
  const days = isos.map(d => d.slice(0, 10)).sort();
  const [first, last] = [days[0], days[days.length - 1]];
  if (first === last) return formatDate(first);
  const part = (iso: string, opts: Intl.DateTimeFormatOptions) => new Date(iso).toLocaleDateString('en-US', { ...opts, timeZone: 'UTC' });
  if (first.slice(0, 7) === last.slice(0, 7)) return `${part(first, { month: 'short', day: 'numeric' })}–${part(last, { day: 'numeric' })}, ${first.slice(0, 4)}`;
  if (first.slice(0, 4) === last.slice(0, 4)) return `${part(first, { month: 'short', day: 'numeric' })} – ${formatDate(last)}`;
  return `${formatDate(first)} – ${formatDate(last)}`;
}

// How many times fewer tokens CORE uses than the full suite.
export function tokenMultiplier(t: { core: number; full: number }): number | null {
  return t.core > 0 && t.full > 0 ? t.full / t.core : null;
}

export function formatMultiplier(m: number): string {
  return m >= 10 ? Math.round(m).toString() : m.toFixed(1);
}

export function formatPercent(share: number): string {
  return `${Math.round(share * 100)}%`;
}

export function formatAccuracyInterval(interval: readonly [number, number]): string {
  return `${Math.round(interval[0] * 100)}–${Math.round(interval[1] * 100)}%`;
}

// Above this share of imputed items, an evaluation-area score mostly reflects benchmarks the
// model did not run, and its interval (which excludes imputation error) understates the uncertainty.
export const IMPUTED_SHARE_LIMIT = 0.5;

// An evaluation-area score is only shown when the model answered at least one of its CORE items.
export function hasMeasuredScore(score: LabelScore | undefined): score is LabelScore & { predicted_score: number } {
  return !!score && score.predicted_score !== null && score.observations > 0;
}

export function imputedShare(score: LabelScore): number {
  if (score.imputed_weight !== undefined) return score.imputed_weight;
  return score.population_items > 0 ? score.imputed_items / score.population_items : 0;
}

// Plausible rank range from 90% intervals: models whose interval lies entirely above this
// one's are clearly ahead; models whose interval lies entirely below are clearly behind.
export function rankRange(models: JoinedModelData[], model: JoinedModelData, index: string): { best: number; worst: number; of: number } | null {
  const own = getIndexTheta(model, index);
  if (!own) return null;
  const others = models.filter(m => m.id !== model.id).map(m => getIndexTheta(m, index)).filter(t => t !== null);
  const ahead = others.filter(t => t.interval[0] > own.interval[1]).length;
  const behind = others.filter(t => t.interval[1] < own.interval[0]).length;
  const of = others.length + 1;
  return { best: ahead + 1, worst: of - behind, of };
}

export function formatRankRange({ best, worst }: { best: number; worst: number }): string {
  return best === worst ? `#${best}` : `#${best}–${worst}`;
}

export const TASK_STATUS: Record<TaskStatus, { label: string; description: string; className: string }> = {
  ok: {
    label: 'Run',
    description: 'All of this benchmark’s CORE items were answered.',
    className: 'bg-teal-soft text-teal-strong border-teal/60',
  },
  partial: {
    label: 'Partly run',
    description: 'Some of this benchmark’s CORE items were answered; the rest are predicted.',
    className: 'bg-brand-soft text-brand-strong border-brand/50',
  },
  imputed: {
    label: 'Not run',
    description: 'The model was not run on this benchmark. It does not count towards the index score.',
    className: 'bg-muted text-muted-foreground border-border',
  },
  missing: {
    label: 'Not run',
    description: 'The model was not run on this benchmark.',
    className: 'bg-muted text-muted-foreground border-border',
  },
};

// A YYYY-MM-DD date as "Nov '23", as on the model cards.
export function formatRelease(date: string, long = false): string {
  const parsed = new Date(`${date}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return date;
  if (long) return `Released ${parsed.toLocaleDateString('en-US', { month: 'long', year: 'numeric', timeZone: 'UTC' })}`;
  const month = parsed.toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' });
  return `${month} '${String(parsed.getUTCFullYear()).slice(-2)}`;
}
