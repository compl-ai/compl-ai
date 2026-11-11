import type { Metadata } from 'next';
import Link from 'next/link';
import { README_QUICKSTART_URL, pageMetadata } from '@/lib/site';
import { Fragment } from 'react';
import { ArrowDownRight, ArrowRight, ArrowUpRight } from 'lucide-react';
import { notFound } from 'next/navigation';
import { IntervalBar } from '@/components/IntervalBar';
import { ScoreDistribution } from '@/components/ScoreDistribution';
import { OutcomeBar, formatShare } from '@/components/OutcomeBar';
import { RunTabs } from '@/components/RunTabs';
import { ReportToc } from '@/components/ReportToc';
import { cn } from '@/lib/utils';
import { ModelMeta } from '@/components/ModelCard';
import { CopyCommand } from '@/components/home/CopyCommand';
import { fmtUsd } from '@/components/home/explorer';
import { areaSlug, benchmarkAnchor, getBenchmarkCatalog, type CatalogBenchmark } from '@/lib/catalog';
import {
  getAreaDescriptions,
  getBenchmarkMeta,
  getCoreRelease,
  getExplorer,
  getGrounding,
  getJoinedModels,
  getPredictionRecord,
  getSampleAnalysis,
  getTaxonomy,
  type JoinedModelData,
  type LabelScore,
  type ModelConfig,
  type SampleAnalysis,
  type ExplorerCostRow,
  type TaskUsage,
} from '@/lib/data';
import { citeLabel } from '@/lib/frameworks';
import { OUTCOMES, TAG_LABELS } from '@/lib/outcomes';
import {
  hasMeasuredScore,
  IMPUTED_SHARE_LIMIT,
  INDEX_BAR,
  INDICES,
  NO_SCORE_REASON,
  RUN_TYPES,
  THETA_MEAN,
  formatDate,
  formatDateRange,
  formatInterval,
  formatMultiplier,
  formatAccuracyInterval,
  formatPercent,
  formatRankRange,
  formatScore,
  formatTokens,
  getIndexTheta,
  getRunType,
  imputedShare,
  rankRange,
  scaleTheta,
  scoreBarPct,
  tokenMultiplier,
} from '@/lib/scores';

export async function generateMetadata(props: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const { id } = await props.params;
  const model = getJoinedModels().find(m => m.yamlId === id);
  const name = model?.metadata?.name || id;
  return pageMetadata(name, `${name} on ComplAI CORE: Capability, Reliability and Safety scores with 90% intervals, by benchmark and evaluation area.`);
}

export function generateStaticParams() {
  return getJoinedModels().map((model) => ({ id: model.yamlId }));
}

function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Numbers hang in a gutter, so each level's text starts further in than the one above.
const INDENT = 'sm:pl-12';
const GUTTER_NUMBER = 'mr-3 shrink-0 sm:mr-0 sm:w-12';

// Report chapters are numbered (1, 2, 3) and their parts (1.1, 1.2, 1.2.1), matching the contents on the left.
function SectionHeading({ id, number, title, children }: { id?: string; number?: string; title: string; children?: React.ReactNode }) {
  return (
    <div className="mb-4">
      <h2 id={id} className="flex items-baseline scroll-mt-28 text-2xl font-extrabold tracking-tight text-ink">
        {number && <span className={`${GUTTER_NUMBER} tabular text-muted-foreground/60`}>{number}</span>}
        <span>{title}</span>
      </h2>
      {children && <p className={`mt-1 text-base text-muted-foreground leading-relaxed ${number ? INDENT : ''}`}>{children}</p>}
    </div>
  );
}

function SubHeading({ id, number, title, dot, level = 3, children }: { id: string; number: string; title: string; dot?: string; level?: 3 | 4; children?: React.ReactNode }) {
  const H = level === 3 ? 'h3' : 'h4';
  return (
    <div className="mb-3">
      <H id={id} className={`flex items-baseline scroll-mt-28 font-extrabold tracking-tight text-ink ${level === 3 ? 'text-lg' : 'text-base'}`}>
        <span className={`${GUTTER_NUMBER} tabular text-muted-foreground/60`}>{number}</span>
        <span className="inline-flex items-center gap-2.5">
          {dot && <span className={`size-2.5 shrink-0 rounded-full ${dot}`} aria-hidden />}
          {title}
        </span>
      </H>
      {children && <p className={`mt-1 text-base text-muted-foreground leading-relaxed ${INDENT}`}>{children}</p>}
    </div>
  );
}

function BarKey({ tick, point }: { tick: string; point: string }) {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-xs text-muted-foreground">
      <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-px bg-ink/40" />{tick}</span>
      <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full bg-ink/60" />{point}</span>
      <span className="inline-flex items-center gap-1.5"><span className="h-1.5 w-4 rounded-full bg-ink/20" />90% interval</span>
    </div>
  );
}

function IndexCard({ models, model, index }: { models: JoinedModelData[]; model: JoinedModelData; index: (typeof INDICES)[number] }) {
  const theta = getIndexTheta(model, index.id);
  const rank = rankRange(models, model, index.id);
  return (
    <div data-print-block className="rounded-2xl border border-border bg-card p-5" title={index.description}>
      <div className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">{index.label}</div>
      {theta ? (
        <>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="font-serif text-4xl font-extrabold tracking-tight tabular text-ink">{formatScore(theta.theta)}</span>
            <span className="text-sm text-muted-foreground tabular">{formatInterval(theta.interval)}</span>
          </div>
          <IntervalBar
            className="mt-3"
            interval={[scaleTheta(theta.interval[0]), scaleTheta(theta.interval[1])]}
            point={scaleTheta(theta.theta)}
            position={scoreBarPct}
            reference={THETA_MEAN}
            tone={INDEX_BAR[index.id]}
          />
          <div className="mt-3 flex justify-between gap-2 text-xs text-muted-foreground">
            <span>From {theta.tasks} of {theta.tasks_total} benchmarks</span>
            {rank && (
              <span
                className="font-semibold text-ink tabular"
                title="Ranks this model could hold given the 90% intervals: models whose interval lies wholly above or below this one are clearly ahead or behind."
              >
                Rank {formatRankRange(rank)} of {rank.of}
              </span>
            )}
          </div>
        </>
      ) : (
        <>
          <div className="mt-2 font-serif text-4xl font-extrabold tracking-tight text-muted-foreground/50">—</div>
          <p className="mt-3 text-xs text-muted-foreground leading-relaxed">{NO_SCORE_REASON}</p>
        </>
      )}
    </div>
  );
}

// The route is the prefix of the eval model id, e.g. "openai-api/vllm/Qwen/..." or "openrouter/anthropic/...".
function formatRoute(id: string, config: ModelConfig | undefined): React.ReactNode {
  const route = id.startsWith('openai-api/vllm/') || id.startsWith('vllm/') ? 'Self-hosted vLLM'
    : id.startsWith('openrouter/') ? 'OpenRouter'
    : id.startsWith('openai/') ? 'OpenAI API'
    : id.split('/')[0];
  const label = config?.batch ? `${route} (batch)` : route;
  if (!id.startsWith('openrouter/')) return label;
  // The OpenRouter model slug from the logs maps directly to its model page.
  return (
    <a href={`https://openrouter.ai/${id.slice('openrouter/'.length)}`} target="_blank" rel="noopener noreferrer" className="text-link hover:underline">
      {label}
    </a>
  );
}

function formatReasoning(config: ModelConfig | undefined): string {
  const kwargs = config?.extra_body?.chat_template_kwargs ?? {};
  const parts: string[] = [];
  if (config?.reasoning_effort) parts.push(`${config.reasoning_effort[0].toUpperCase()}${config.reasoning_effort.slice(1)} effort`);
  if (typeof kwargs.thinking_budget === 'number') parts.push(`${formatTokens(kwargs.thinking_budget)} thinking budget`);
  if (kwargs.interleaved_thinking) parts.push('interleaved thinking');
  return parts.length ? parts.join(', ') : 'Not set';
}

type Tone = { band: string; dot: string };
type Prediction = JoinedModelData['prediction'];
type TaskPrediction = Prediction['tasks'][string];
type Tokens = NonNullable<NonNullable<Prediction['usage']>['tokens']>;

const notRun = (t: TaskPrediction | undefined) => !t || t.status === 'imputed' || t.status === 'missing';
const scoreMargin = (t: TaskPrediction) => (t.predicted_score_error ? 1.645 * t.predicted_score_error : 0);

const FACT_ROW = 'flex gap-x-6 gap-y-3';

function Fact({ label, className, children }: { label: string; className?: string; children: React.ReactNode }) {
  return (
    <div className={cn('min-w-0', className)}>
      <dt className="text-[11px] font-semibold uppercase tracking-[0.02em] text-muted-foreground whitespace-nowrap">{label}</dt>
      <dd className="mt-0.5 text-sm font-medium text-ink">{children}</dd>
    </div>
  );
}

function CostSection({ tokens, usd, run, coreAnswered, coreTotal, tasksTotal }: {
  tokens: Tokens;
  usd: ExplorerCostRow | null;
  run: NonNullable<ReturnType<typeof getRunType>>;
  coreAnswered: number | null;
  coreTotal: number | null;
  tasksTotal: number;
}) {
  const multiplier = tokenMultiplier(tokens);
  const coreEstimated = run !== 'core';
  const fullEstimated = run !== 'full';
  const bars = [
    { label: 'CORE subset', value: tokens.core, estimated: coreEstimated, className: 'bg-teal-strong' },
    { label: 'Every question', value: tokens.full, estimated: fullEstimated, className: 'bg-ink/25' },
  ];
  return (
    <section aria-labelledby="cost">
      <SectionHeading id="cost" title="Cost of the evaluation">
        {RUN_TYPES[run].description} Answering only the CORE subset takes a fraction of the tokens needed to run every question of the suite.
      </SectionHeading>
      <div className="rounded-2xl border border-border bg-card p-5 md:p-6 grid grid-cols-1 md:grid-cols-[minmax(0,15rem)_minmax(0,1fr)] gap-6 md:gap-10 items-center">
        {multiplier !== null && (
          <div>
            <div className="font-serif text-5xl font-extrabold tracking-tight tabular text-teal-strong">{formatMultiplier(multiplier)}×</div>
            <div className="mt-1 text-sm text-ink">more efficient than running every question</div>
          </div>
        )}
        <div className="space-y-3">
          {bars.map(b => (
            <div key={b.label}>
              <div className="flex items-baseline justify-between gap-3 text-sm">
                <span className="text-muted-foreground">{b.label}</span>
                <span className="tabular">
                  <span className="font-semibold text-ink">{b.estimated ? '~' : ''}{formatTokens(b.value)} tokens</span>
                  <span className="ml-1.5 text-xs text-muted-foreground">{b.estimated ? 'estimated' : 'measured'}</span>
                </span>
              </div>
              <div className="mt-1.5 h-2 rounded-full bg-track">
                <div className={`h-2 rounded-full ${b.className}`} style={{ width: `${Math.max((b.value / tokens.full) * 100, 1)}%` }} />
              </div>
            </div>
          ))}
          <p className="pt-1 text-sm text-muted-foreground leading-relaxed">
            {coreAnswered !== null && coreTotal !== null && <>{coreAnswered.toLocaleString()} of {coreTotal.toLocaleString()} core questions answered. </>}
            Tokens of the evaluated model only, measured in the logs of {tokens.tasks} of {tasksTotal} benchmarks; judge models are not counted.{' '}
            <Link href="/methodology#efficiency" className="font-semibold text-link hover:underline">How we estimate this</Link>
          </p>
          {usd && (
            <p className="text-sm text-muted-foreground leading-relaxed">
              At {usd.source} list prices this is about <span className="font-semibold text-ink tabular">{fmtUsd(usd.core_usd)}</span> for CORE
              and <span className="font-semibold text-ink tabular">{fmtUsd(usd.full_usd)}</span> for every question. Batch APIs roughly halve both.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

// One benchmark of "What we tested". In the CORE view the score is predicted from the core questions; in the calibration
// view it is the plain accuracy of the full run, with the CORE score beside it for comparison.
function BenchmarkRow({ task, name, tests, t, view, usage, field, tone }: {
  task: string;
  name: string;
  tests: string[];
  t: TaskPrediction;
  view: 'core' | 'full';
  usage: TaskUsage | undefined;
  field: number[];
  tone: Tone;
}) {
  const fullRun = t.full_run;
  const predicted = !(notRun(t) || t.predicted_score === null);
  const shown = view === 'full' ? !!fullRun : predicted;
  const margin = view === 'core' && predicted ? scoreMargin(t) : 0;
  const point = view === 'full' ? fullRun?.score : t.predicted_score;
  // Core view: the core questions the prediction rests on. Tokens show only under the run they were spent in.
  const answered = view === 'full' ? fullRun?.items_answered : predicted ? t.observations : undefined;
  const runTokens = usage?.model_tokens && usage.run === (view === 'full' ? 'full' : 'core') ? usage.model_tokens : 0;
  const meta = [
    ...(shown && answered !== undefined
      ? [view === 'full'
        ? `${answered.toLocaleString()} of ${t.population_items.toLocaleString()} questions`
        : answered === t.subset_items
          ? `All ${t.subset_items.toLocaleString()} core questions`
          : `${answered.toLocaleString()} of ${t.subset_items.toLocaleString()} core questions`]
      : []),
    ...(shown && runTokens ? [`${formatTokens(runTokens)} tokens`] : []),
  ];
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] sm:grid-cols-[minmax(0,1fr)_minmax(0,14rem)_5.5rem] items-center gap-x-6 gap-y-2 py-3 border-b border-border last:border-0">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <Link href={`/benchmarks#${benchmarkAnchor(task)}`} className={cn('text-sm font-semibold hover:text-link hover:underline', shown ? 'text-ink' : 'text-muted-foreground')}>{name}</Link>
        </div>
        {tests.length > 0 && <div className="mt-0.5 text-sm text-muted-foreground">{tests.join(' · ')}</div>}
        {meta.length > 0 && <div className="mt-0.5 text-xs tabular text-muted-foreground/75">{meta.join(' · ')}</div>}
      </div>
      <div className="hidden sm:block">
        {shown && (
          <ScoreDistribution
            field={field}
            point={point!}
            interval={margin ? [Math.max(0, point! - margin), Math.min(1, point! + margin)] : null}
            tone={tone}
          />
        )}
      </div>
      <div className="text-right tabular whitespace-nowrap">
        {!shown ? (
          <span className="text-xs text-muted-foreground">{view === 'full' && predicted ? 'Core only' : 'Not run'}</span>
        ) : (
          <>
            <span className="text-sm font-semibold text-ink">{formatPercent(point!)}</span>
            {margin >= 0.005 && <span className="ml-1 text-xs text-muted-foreground">±{Math.round(margin * 100)}</span>}
            {view === 'full' && predicted && (
              <div className="text-xs text-muted-foreground" title="The CORE score, predicted from the core questions, for comparison.">
                CORE {formatPercent(t.predicted_score!)}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

interface AreaSource { task: string; name: string; items: number; ownShare: number; areaShare: number; weight: number }

export type AreaStandout = 'best' | 'worst' | null;

const SOURCE_SHADES = ['bg-ink/70', 'bg-ink/45', 'bg-ink/30', 'bg-ink/20'];

// What an area's score is made of: each benchmark's weight in it, as one stacked bar with a key underneath.
function SourceStack({ shown, rest, prediction }: {
  shown: AreaSource[];
  rest: number;
  prediction: Prediction;
}) {
  const pct = (v: number) => (v < 0.005 ? '<1%' : `${Math.round(v * 100)}%`);
  const segments = [
    ...shown.map((s, i) => ({ key: s.task, share: s.weight, shade: SOURCE_SHADES[Math.min(i, SOURCE_SHADES.length - 1)], source: s })),
    ...(rest >= 0.005 ? [{ key: 'other', share: rest, shade: 'bg-track', source: null }] : []),
  ];
  return (
    <div className="mt-7 max-w-xl">
      <div className="text-xs text-muted-foreground">What benchmark scores inform this area and how do we weigh them?</div>
      <div className="mt-1 flex h-1.5 w-full gap-[2px] overflow-hidden rounded-full">
        {segments.map(seg => (
          <span
            key={seg.key}
            className={`basis-0 ${seg.shade}`}
            style={{ flexGrow: seg.share }}
            title={seg.source
              ? `${seg.source.name}: ${pct(seg.share)} of the score · ${seg.source.items.toLocaleString()} questions in this area`
              : `Other benchmarks: ${pct(seg.share)} of the score`}
          />
        ))}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {segments.map(seg => (
          <span key={seg.key} className="inline-flex items-center gap-1.5">
            <span className={`size-2 rounded-sm ${seg.shade}`} />
            {seg.source ? (
              <Link href={`/benchmarks#${benchmarkAnchor(seg.source.task)}`} className="text-ink hover:text-link hover:underline">{seg.source.name}</Link>
            ) : 'Other benchmarks'}
            <span className="tabular">{pct(seg.share)}</span>
            {seg.source && notRun(prediction.tasks[seg.source.task]) && <span className="italic">not run</span>}
          </span>
        ))}
      </div>
    </div>
  );
}

// A one-line reading of the area score against the other models, from fixed templates.
function areaReading(score: LabelScore & { predicted_score: number }, field: number[], panelMedian: number | null, standout: AreaStandout, indexLabel: string): { direction: 'up' | 'down' | 'level'; text: React.ReactNode } | null {
  if (panelMedian === null || field.length === 0) return null;
  const interval = score.score_interval;
  const ahead = field.filter(v => v < score.predicted_score).length;
  const share = ahead / field.length;
  const inLine = !!interval && panelMedian >= interval[0] && panelMedian <= interval[1];
  const above = !inLine && score.predicted_score > panelMedian;
  const of = `${ahead} of the ${field.length} other models scored here`;
  const lead =
    inLine ? <>In line with the median model ({formatPercent(panelMedian)}).</>
    : above && ahead === field.length ? <>The highest score of the {field.length + 1} models scored in this area.</>
    : !above && ahead === 0 ? <>The lowest score of the {field.length + 1} models scored in this area.</>
    : above && share >= 0.9 ? <>One of the strongest models in this area, ahead of {of}.</>
    : !above && share <= 0.1 ? <>One of the weakest models in this area, ahead of only {of}.</>
    : above ? <>Above the median model ({formatPercent(panelMedian)}), ahead of {of}.</>
    : <>Below the median model ({formatPercent(panelMedian)}), ahead of {of}.</>;
  const tail =
    standout === 'best' ? <> Compared with other models, this is its strongest {indexLabel.toLowerCase()} area.</>
    : standout === 'worst' ? <> Compared with other models, this is its weakest {indexLabel.toLowerCase()} area.</>
    : null;
  return { direction: inLine ? 'level' : above ? 'up' : 'down', text: <>{lead}{tail}</> };
}

function ReadingIcon({ direction }: { direction: 'up' | 'down' | 'level' }) {
  const Icon = direction === 'up' ? ArrowUpRight : direction === 'down' ? ArrowDownRight : ArrowRight;
  return <Icon className="relative top-0.5 size-3.5 shrink-0 self-start text-muted-foreground" strokeWidth={2.5} aria-hidden />;
}

// Each scored area of an index on one 0–100% scale: the other models' scores as a histogram, the median
// model as a tick and this model as a line with its 90% interval.
function AreaChart({ rows, tone }: {
  rows: { key: string; name: string; field: number[]; score: LabelScore & { predicted_score: number }; median: number | null }[];
  tone: Tone;
}) {
  return (
    <figure className="mt-5 mb-2">
      <div className="grid grid-cols-[minmax(0,8.5rem)_minmax(0,1fr)] gap-x-4 gap-y-2.5 items-center">
        <span />
        <div className="relative h-3 text-xs tabular text-muted-foreground">
          {[0, 25, 50, 75, 100].map(v => (
            <span key={v} className={cn('absolute top-0', v === 0 ? '' : v === 100 ? '-translate-x-full' : '-translate-x-1/2')} style={{ left: `${v}%` }}>{v}%</span>
          ))}
        </div>
        {rows.map(r => {
          const muted = imputedShare(r.score) >= IMPUTED_SHARE_LIMIT;
          return (
            <Fragment key={r.key}>
              <a href={`#result-${areaSlug(r.key)}`} className="truncate text-sm text-ink hover:text-link hover:underline">{r.name}</a>
              <ScoreDistribution
                field={r.field}
                point={r.score.predicted_score}
                interval={r.score.score_interval ?? null}
                reference={r.median}
                tone={tone}
                className={muted ? 'opacity-50' : ''}
              />
            </Fragment>
          );
        })}
      </div>
      <figcaption className="mt-3 text-xs text-muted-foreground">
        Grey bars: the other models’ scores · thin tick: the median model · coloured line and band: this model and its 90% interval.
      </figcaption>
    </figure>
  );
}

function AreaBlock({ areaKey, name, description, refs, sources, prediction, score, panelMedian, field, standout, indexLabel }: {
  areaKey: string;
  name: string;
  description: string;
  refs: { id: string; label: string; title: string; direct: boolean }[];
  sources: AreaSource[];
  prediction: Prediction;
  score: LabelScore & { predicted_score: number };
  panelMedian: number | null;
  field: number[];
  standout: AreaStandout;
  indexLabel: string;
}) {
  // Each benchmark's weight in the score (item share when an older export has no weights). Benchmarks with at
  // least 5% of the weight, or most of their own questions in the area, are shown; the rest are grouped.
  const weighted = sources.map(s => ({ ...s, weight: score.task_weights ? score.task_weights[s.task] ?? 0 : s.areaShare }));
  const shown = weighted.filter(s => s.weight >= 0.05 || s.ownShare >= 0.5).sort((x, y) => y.weight - x.weight);
  const rest = 1 - shown.reduce((sum, s) => sum + s.weight, 0);
  const mostlyImputed = imputedShare(score) >= IMPUTED_SHARE_LIMIT;
  const answered = score.observations === score.subset_items
    ? `from ${score.subset_items.toLocaleString()} CORE answers`
    : `from ${score.observations.toLocaleString()} of ${score.subset_items.toLocaleString()} CORE answers`;
  const reading = areaReading(score, field, panelMedian, standout, indexLabel);

  return (
    <div id={`result-${areaSlug(areaKey)}`} data-print-block className="scroll-mt-28 py-10 border-b border-border last:border-0">
      <div className="flex items-start justify-between gap-6">
        <div className="min-w-0">
          <h5><Link href={`/mapping#${areaSlug(areaKey)}`} className="text-base font-bold text-ink hover:text-link hover:underline">{name}</Link></h5>
          {reading && (
            <p className="mt-1 flex items-baseline gap-2 text-[13px] text-ink/80 leading-snug">
              <ReadingIcon direction={reading.direction} />
              <span>{reading.text}</span>
            </p>
          )}
        </div>
        <div className="shrink-0 text-right tabular" title={score.score_interval ? `90% interval ${formatAccuracyInterval(score.score_interval)}` : undefined}>
          <div className={`text-xl font-bold leading-6 ${mostlyImputed ? 'text-ink/50' : 'text-ink'}`}>{formatPercent(score.predicted_score)}{mostlyImputed && '*'}</div>
          {score.score_interval && <div className="text-xs text-muted-foreground">{formatAccuracyInterval(score.score_interval)}</div>}
          <div className="text-xs text-muted-foreground">{answered}</div>
        </div>
      </div>
      <p className="mt-3 text-base text-muted-foreground leading-relaxed">
        {description}
        {mostlyImputed && (
          <>
            {' '}Most of this score comes from benchmarks the model did not run, so it is{' '}
            <Link href="/methodology#areas" className="text-link hover:underline">largely inferred</Link> from its other results and the
            real uncertainty is wider than the interval shows.
          </>
        )}
      </p>
      <SourceStack shown={shown} rest={rest} prediction={prediction} />
      {refs.length > 0 && (
        <div className="mt-6">
          <div className="flex items-baseline justify-between gap-3 text-xs text-muted-foreground">
            <span>How it maps to regulation</span>
            {refs.some(r => !r.direct) && <span>Dashed: related, not directly tested</span>}
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2">
          {refs.map(r => (
            <Link
              key={r.id}
              href={`/mapping#ref-${r.id}`}
              title={`${r.label}${r.direct ? '' : ' (related)'}`}
              className={`rounded-full border px-2 py-0.5 text-xs leading-4 hover:border-ink ${r.direct ? 'border-ink/40' : 'border-dashed border-ink/30'}`}
            >
              <span className={`font-semibold ${r.direct ? 'text-ink' : 'text-ink/70'}`}>{r.title}</span>
              <span className="ml-1.5 text-muted-foreground">{r.label}</span>
            </Link>
          ))}
          </div>
        </div>
      )}
    </div>
  );
}

const listJoin = (items: React.ReactNode[]) =>
  items.map((item, i) => (
    <Fragment key={i}>
      {i > 0 && (i === items.length - 1 ? ' and ' : ', ')}
      {item}
    </Fragment>
  ));

const TD = 'px-4 py-2.5 align-top';
const TD_NUM = 'px-4 py-2.5 align-top text-right tabular whitespace-nowrap';

function Table({ columns, children }: { columns: { label: string; numeric?: boolean }[]; children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-border bg-card">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
            {columns.map(c => (
              <th key={c.label} scope="col" className={cn('px-4 pt-3 pb-2 font-semibold whitespace-nowrap', c.numeric ? 'text-right' : 'text-left')}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border">{children}</tbody>
      </table>
    </div>
  );
}

function Note({ children }: { children: React.ReactNode }) {
  return <p className="mt-2 text-sm text-muted-foreground leading-relaxed">{children}</p>;
}

const benchmarkLink = (task: string, name: string) => (
  <Link href={`/benchmarks#${benchmarkAnchor(task)}`} className="font-semibold text-ink hover:text-link hover:underline">{name}</Link>
);

// Bars on a straight 50–150 index scale, so lengths compare directly.
const PEER_LOW = 50;
const PEER_HIGH = 150;
const peerPct = (score: number) => (Math.min(Math.max(score, PEER_LOW), PEER_HIGH) - PEER_LOW) / (PEER_HIGH - PEER_LOW) * 100;

function formatParams(n: number): string {
  return n >= 1e12 ? `${+(n / 1e12).toFixed(1)}T` : n >= 1e10 ? `${Math.round(n / 1e9)}B` : `${+(n / 1e9).toFixed(1)}B`;
}

type PeerGroup = { id: string; label: string; note: (n: number) => string; peers: JoinedModelData[]; meta?: (m: JoinedModelData) => string | null };

// This model against a group of others, one bar per index.
function PeerChart({ model, group }: { model: JoinedModelData; group: PeerGroup }) {
  const rows = [model, ...group.peers];
  return (
    <div data-print-block>
      <p className="mb-3 text-base text-muted-foreground">{group.note(group.peers.length)}</p>
      <div className="overflow-x-auto rounded-2xl border border-border bg-card p-5 md:p-6">
        <div className="grid min-w-[36rem] grid-cols-[minmax(0,14rem)_repeat(3,minmax(0,1fr))] items-center gap-x-6 gap-y-3 text-sm">
          <span />
          {INDICES.map(index => (
            <span key={index.id} className="inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
              <span className={`size-2 rounded-full ${INDEX_BAR[index.id].dot}`} aria-hidden />
              {index.label}
            </span>
          ))}
          {rows.map(m => {
            const current = m === model;
            const meta = group.meta?.(m);
            return (
              <Fragment key={m.id}>
                <div className="flex min-w-0 items-baseline gap-2">
                  {current ? (
                    <span className="truncate font-semibold text-ink">{m.metadata?.name || m.yamlId}</span>
                  ) : (
                    <Link href={`/models/${m.yamlId}`} className="truncate text-ink hover:text-link hover:underline">{m.metadata?.name || m.yamlId}</Link>
                  )}
                  {current && <span className="shrink-0 rounded-full bg-muted px-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Current</span>}
                  {meta && <span className="shrink-0 text-sm tabular text-muted-foreground">{meta}</span>}
                </div>
                {INDICES.map(index => {
                  const score = scaleTheta(getIndexTheta(m, index.id)!.theta);
                  return (
                    <div key={index.id} className="flex items-center gap-2">
                      <div className="relative h-2 flex-1 rounded-full bg-track/50">
                        <div className={`absolute inset-y-0 left-0 rounded-full ${current ? INDEX_BAR[index.id].dot : 'bg-ink/30'}`} style={{ width: `${peerPct(score)}%` }} />
                        <div className="absolute -inset-y-1 w-px bg-ink/40" style={{ left: `${peerPct(THETA_MEAN)}%` }} />
                      </div>
                      <span className={`w-7 text-right tabular text-sm ${current ? 'font-bold text-ink' : 'text-ink/80'}`}>{Math.round(score)}</span>
                    </div>
                  );
                })}
              </Fragment>
            );
          })}
        </div>
      </div>
      <Note>Bars run from {PEER_LOW} to {PEER_HIGH} on the index scale; the thin line marks the average model ({THETA_MEAN}).</Note>
    </div>
  );
}

function RunHealth({ analysis, modelId }: { analysis: SampleAnalysis; modelId: string }) {
  const summary = analysis.model_outcomes[modelId];
  const definition = analysis.outcomes.definition;
  const shares = summary.outcomes;
  const medians = Object.fromEntries(
    OUTCOMES.map(o => [o.id, median(Object.values(analysis.model_outcomes).map(m => m.outcomes[o.id] ?? 0)) ?? 0])
  );
  return (
    <div className="rounded-2xl border border-border bg-card p-5 md:p-6">
      <OutcomeBar shares={shares} definition={definition} size="md" />
      <table className="mt-3 w-full text-sm">
        <thead>
          <tr className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
            <th scope="col" className="pb-1 text-left font-semibold"><span className="sr-only">Outcome</span></th>
            <th scope="col" className="pb-1 text-right font-semibold">This model</th>
            <th scope="col" className="pb-1 pl-6 text-right font-semibold">Median model</th>
          </tr>
        </thead>
        <tbody>
          {OUTCOMES.filter(o => shares[o.id] > 0 || medians[o.id] > 0).map(o => (
            <tr key={o.id} title={definition[o.id].tags.map(t => analysis.tags[t]).join(' ')}>
              <td className="py-1">
                <span className="inline-flex items-center gap-2 text-muted-foreground"><span className={`size-2.5 rounded-sm ${o.bar}`} />{definition[o.id].label}</span>
              </td>
              <td className="py-1 text-right font-medium tabular text-ink">{formatShare(shares[o.id])}</td>
              <td className="py-1 pl-6 text-right tabular text-muted-foreground">{formatShare(medians[o.id])}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// The benchmarks with answers that didn't finish or were blocked, and why.
function FailedAnswers({ analysis, modelId, names }: { analysis: SampleAnalysis; modelId: string; names: Record<string, string> }) {
  const definition = analysis.outcomes.definition;
  const perTask = analysis.models[modelId] ?? {};
  const count = (tags: Record<string, number>, ids: string[]) => ids.reduce((sum, id) => sum + (tags[id] ?? 0), 0);
  const failureTags = [...definition.did_not_finish.tags, ...definition.blocked.tags];
  const rows = Object.entries(perTask)
    .map(([task, e]) => ({ task, e, unfinished: count(e.tags, definition.did_not_finish.tags), blocked: count(e.tags, definition.blocked.tags) }))
    .filter(r => r.unfinished + r.blocked > 0)
    .sort((a, b) => (b.unfinished + b.blocked) / b.e.answers - (a.unfinished + a.blocked) / a.e.answers);
  const blockedButCorrect = Object.entries(perTask)
    .filter(([, e]) => e.tags.provider_block_passed)
    .sort((a, b) => b[1].tags.provider_block_passed - a[1].tags.provider_block_passed);
  const repeated = rows.some(r => r.e.answers > r.e.questions);
  const cell = (n: number) => (n ? n.toLocaleString() : <span className="text-muted-foreground/50">–</span>);
  return (
    <>
      {rows.length > 0 ? (
        <>
          <Table columns={[{ label: 'Benchmark' }, { label: 'Answers', numeric: true }, { label: definition.did_not_finish.label, numeric: true }, { label: 'Blocked', numeric: true }, { label: 'Why' }]}>
            {rows.map(({ task, e, unfinished, blocked }) => (
              <tr key={task}>
                <td className={TD}>{benchmarkLink(task, names[task] ?? task)}</td>
                <td className={TD_NUM}>{e.answers.toLocaleString()}</td>
                <td className={TD_NUM}>{cell(unfinished)}</td>
                <td className={TD_NUM}>{cell(blocked)}</td>
                <td className={`${TD} min-w-48 text-muted-foreground`}>
                  {failureTags.filter(t => e.tags[t]).sort((a, b) => e.tags[b] - e.tags[a]).map(t => `${e.tags[t].toLocaleString()} ${TAG_LABELS[t] ?? t}`).join(', ')}
                </td>
              </tr>
            ))}
          </Table>
          {repeated && <Note>Some benchmarks ask each question several times, so there can be more answers than questions.</Note>}
        </>
      ) : (
        <p className="text-base text-ink leading-relaxed">Every answer to the core questions ended correct or wrong: none ended with an error, were cut off or were blocked.</p>
      )}
      {blockedButCorrect.length > 0 && (
        <p className="mt-4 text-base text-muted-foreground leading-relaxed">
          A provider block also fired on answers that were still scored correct, such as refusals the benchmark counts as right:{' '}
          {listJoin(blockedButCorrect.map(([task, e]) => (
            <>{names[task] ?? task} ({e.tags.provider_block_passed.toLocaleString()} of {e.answers.toLocaleString()} answers)</>
          )))}.
        </p>
      )}
    </>
  );
}

// Tokens per benchmark: measured in the run it was answered in, and estimated from the tokens per question otherwise.
function TokenSpend({ taskUsage, names }: { taskUsage: Record<string, TaskUsage>; names: Record<string, string> }) {
  const rows = Object.entries(taskUsage)
    .filter(([, u]) => u.observed_items > 0 && u.model_tokens > 0)
    .map(([task, u]) => {
      const perQuestion = u.model_tokens / u.observed_items;
      return {
        task,
        u,
        perQuestion,
        core: u.run === 'core' ? u.model_tokens : perQuestion * u.subset_items,
        coreEstimated: u.run !== 'core',
        full: u.run === 'full' ? u.model_tokens : perQuestion * u.population_items,
        fullEstimated: u.run !== 'full',
      };
    })
    .sort((a, b) => b.core - a.core);
  if (!rows.length) return null;
  const coreTotal = rows.reduce((sum, r) => sum + r.core, 0);
  const top = rows.slice(0, 5);
  const topShare = top.reduce((sum, r) => sum + r.core, 0) / coreTotal;
  const saving = [...rows].sort((a, b) => b.full - b.core - (a.full - a.core))[0];
  const output = rows.reduce((sum, r) => sum + r.u.output_tokens, 0);
  const reasoning = rows.reduce((sum, r) => sum + r.u.reasoning_tokens, 0);
  const tokens = (value: number, estimated: boolean) => `${estimated ? '~' : ''}${formatTokens(value)}`;
  return (
    <>
      <p className="mb-3 text-base text-ink leading-relaxed">
        The {top.length} benchmarks below take {formatPercent(topShare)} of the model’s CORE tokens. The largest saving is on {names[saving.task] ?? saving.task}:
        its {saving.u.subset_items.toLocaleString()} core questions take {tokens(saving.core, saving.coreEstimated)} tokens, against {tokens(saving.full, saving.fullEstimated)} for
        all {saving.u.population_items.toLocaleString()}.{reasoning > 0 && ` Reasoning made up ${formatPercent(reasoning / output)} of the model’s output tokens.`}
      </p>
      <Table columns={[{ label: 'Benchmark' }, { label: 'Per question', numeric: true }, { label: 'CORE', numeric: true }, { label: 'Every question', numeric: true }]}>
        {top.map(r => (
          <tr key={r.task}>
            <td className={TD}>{benchmarkLink(r.task, names[r.task] ?? r.task)}</td>
            <td className={TD_NUM}>{formatTokens(r.perQuestion)}</td>
            <td className={TD_NUM}>
              <span className="font-semibold text-ink">{tokens(r.core, r.coreEstimated)}</span>
              <span className="ml-1.5 text-xs text-muted-foreground">{formatPercent(r.core / coreTotal)}</span>
            </td>
            <td className={`${TD_NUM} text-muted-foreground`}>{tokens(r.full, r.fullEstimated)}</td>
          </tr>
        ))}
      </Table>
      <Note>Tokens of the evaluated model, with the share of its CORE tokens. ~ marks estimates from the tokens per question in the run that was logged.</Note>
    </>
  );
}

// The CORE prediction against the plain score of the calibration run, which answered every question.
function PredictionCheck({ prediction, names }: { prediction: Prediction; names: Record<string, string> }) {
  const checked = Object.entries(prediction.tasks).filter(([, t]) => t.full_run && !notRun(t) && t.predicted_score !== null);
  const rows = checked
    .filter(([, t]) => t.subset_items < t.population_items)
    .map(([task, t]) => {
      const gap = t.predicted_score! - t.full_run!.score;
      return { task, t, gap, margin: scoreMargin(t), within: Math.abs(gap) <= scoreMargin(t) };
    });
  if (!rows.length) return null;
  const within = rows.filter(r => r.within).length;
  const meanGap = rows.reduce((sum, r) => sum + Math.abs(r.gap), 0) / rows.length;
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const signedPoints = (v: number) => {
    const points = (v * 100).toFixed(1);
    return Number(points) === 0 ? '0.0' : `${v > 0 ? '+' : '−'}${points.replace('-', '')}`;
  };
  return (
    <>
      <p className="mb-3 text-base text-ink leading-relaxed">
        {within} of {rows.length} calibration scores fall within the 90% interval of the CORE prediction, and the prediction is {(meanGap * 100).toFixed(1)} points off on average.
      </p>
      <Table columns={[{ label: 'Benchmark' }, { label: 'CORE prediction', numeric: true }, { label: 'Calibration run', numeric: true }, { label: 'Gap', numeric: true }, { label: 'Within interval', numeric: true }]}>
        {rows.map(({ task, t, gap, margin, within }) => {
          const fullRun = t.full_run!;
          return (
            <tr key={task}>
              <td className={TD}>{benchmarkLink(task, names[task] ?? task)}</td>
              <td className={TD_NUM}>
                <span className="font-semibold text-ink">{pct(t.predicted_score!)}</span>
                <span className="ml-1 text-xs text-muted-foreground">±{(margin * 100).toFixed(1)}</span>
              </td>
              <td className={TD_NUM}>
                {pct(fullRun.score)}
                {fullRun.items_answered < fullRun.population_items && (
                  <div className="text-xs text-muted-foreground">{fullRun.items_answered.toLocaleString()} of {fullRun.population_items.toLocaleString()} questions</div>
                )}
              </td>
              <td className={TD_NUM}>{signedPoints(gap)}</td>
              <td className={cn(TD_NUM, within ? 'text-ink' : 'font-semibold text-brand-strong')}>{within ? 'Yes' : 'No'}</td>
            </tr>
          );
        })}
      </Table>
      <Note>
        Gaps are in points; positive means the prediction was higher.
        {checked.length > rows.length && ' Benchmarks whose core questions are all of their questions are left out, as there is nothing to predict.'}
      </Note>
    </>
  );
}

// The command lines to rerun this evaluation, built from the settings in the model's logs.
function reproduceCommands(modelId: string, config: ModelConfig | undefined): string[] {
  const flags: string[] = [];
  if (config?.max_tokens) flags.push(`--max-tokens ${config.max_tokens}`);
  if (config?.reasoning_effort) flags.push(`--reasoning-effort ${config.reasoning_effort}`);
  if (config?.batch) flags.push(`--batch ${config.batch}`);
  if (config?.extra_body) flags.push(`-G 'extra_body=${JSON.stringify(config.extra_body)}'`);
  const logFolder = `logs/${modelId.split('/').slice(-2).join('_')}_<timestamp>`;
  return [
    flags.length ? `complai eval ${modelId} \\` : `complai eval ${modelId}`,
    ...flags.map((flag, i) => `  ${flag}${i < flags.length - 1 ? ' \\' : ''}`),
    '# predict the scores from the logs',
    `complai predict ${logFolder} --output predicted.json`,
  ];
}

export default async function ModelReportPage(props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const models = getJoinedModels();
  const model = models.find(m => m.yamlId === params.id);
  if (!model) notFound();

  const taxonomy = getTaxonomy();
  const grounding = getGrounding();
  const areaDescriptions = getAreaDescriptions();
  const benchmarkMeta = getBenchmarkMeta();
  const catalog = getBenchmarkCatalog();
  const prediction = model.prediction;
  const run = getRunType(model);
  const tokens = prediction.usage?.tokens ?? null;
  const taskUsage = prediction.usage?.tasks ?? {};
  const coverage = prediction.coverage;
  const modelConfig = prediction.model_config;
  const coreRelease = getCoreRelease();
  const name = model.metadata?.name || model.yamlId;
  const analysis = getSampleAnalysis();
  const answerOutcomes = analysis.models[model.id] ?? {};
  const isCalibration = Object.values(prediction.tasks).some(t => t.full_run);
  const hasSeparateFullRun = Object.entries(prediction.tasks).some(([task, t]) => t.full_run && answerOutcomes[task] && t.full_run.log_path !== answerOutcomes[task].log_path);
  const tasksTotal = Object.keys(prediction.tasks).length;
  const usd = getExplorer().eval_cost.rows.find(r => r.id === model.id) ?? null;

  const benchmarkName = (task: string) => benchmarkMeta[task]?.name ?? task;
  const names = Object.fromEntries(catalog.map(b => [b.name, benchmarkName(b.name)]));
  const labelledItems = (b: CatalogBenchmark) => b.areas.reduce((sum, a) => sum + a.items, 0);

  // What each benchmark tests: the areas holding at least a tenth of its questions.
  const benchmarkTests = Object.fromEntries(
    catalog.map(b => [b.name, b.areas.filter(a => a.items / labelledItems(b) >= 0.1).map(a => a.name)])
  );

  // The benchmarks in each area: their questions in it, and its share of their own questions and of the area's.
  const areaSources: Record<string, AreaSource[]> = {};
  for (const b of catalog) {
    for (const a of b.areas) (areaSources[a.key] ??= []).push({ task: b.name, name: benchmarkName(b.name), items: a.items, ownShare: a.items / labelledItems(b), areaShare: a.items, weight: 0 });
  }
  for (const list of Object.values(areaSources)) {
    const total = list.reduce((sum, s) => sum + s.areaShare, 0);
    for (const s of list) s.areaShare /= total;
    list.sort((x, y) => y.ownShare - x.ownShare);
  }

  // Every other model's score on each benchmark it ran, for the faint dots behind this model's bar.
  const taskField = (task: string) =>
    models
      .filter(m => m !== model)
      .map(m => m.prediction.tasks[task])
      .filter(t => !notRun(t) && t.predicted_score !== null)
      .map(t => t.predicted_score!);

  // Other models' plain scores from their calibration runs, for the calibration view.
  const fullField = (task: string) =>
    models
      .filter(m => m !== model)
      .map(m => m.prediction.tasks[task]?.full_run?.score)
      .filter((v): v is number => typeof v === 'number');
  const calibrationModels = models.filter(m => m !== model && Object.values(m.prediction.tasks).some(t => t.full_run)).length;

  // The runs behind the scores, dated from the eval log names (UTC).
  const logDate = (logPath: string | undefined) => logPath?.match(/(\d{4}-\d{2}-\d{2})T[^/]*$/)?.[1];
  const scoredTasks = Object.keys(prediction.tasks).filter(task => !notRun(prediction.tasks[task]) && prediction.tasks[task].predicted_score !== null);
  // A benchmark small enough to be all core questions is tagged 'core' but may still have been answered in the calibration run.
  const runBasis = (task: string) => {
    const fullRun = prediction.tasks[task]?.full_run;
    if (fullRun && fullRun.log_path === answerOutcomes[task]?.log_path) return taskUsage[task]?.run === 'partial' ? 'partial' : 'full';
    return taskUsage[task]?.run;
  };
  const fromCoreRun = scoredTasks.filter(task => (runBasis(task) ?? 'core') === 'core');
  const fromCalibration = scoredTasks.filter(task => !fromCoreRun.includes(task));
  const answerDates = (tasks: string[]) =>
    tasks.map(task => logDate(answerOutcomes[task]?.log_path) ?? prediction.last_evaluated).filter((d): d is string => !!d);
  const fullRunTasks = Object.values(prediction.tasks).filter(t => t.full_run);
  const fullRunPartial = fullRunTasks.filter(t => t.full_run!.items_answered < t.full_run!.population_items).length;
  const notRunCount = tasksTotal - scoredTasks.length;
  const releaseLabel = coreRelease ? coreRelease.replace(/^core\./, 'CORE ') : 'CORE';
  const benchmarks = (n: number) => `${n} benchmark${n === 1 ? '' : 's'}`;

  const subcategoryMedians = Object.fromEntries(
    Object.keys(taxonomy.subcategories).map(key => [
      key,
      median(models.map(m => m.prediction.subcategories?.[key]).filter(hasMeasuredScore).map(s => s.predicted_score)),
    ])
  );

  // Every other model's score in an evaluation area, where it has one.
  const areaField = (key: string) =>
    models
      .filter(m => m !== model)
      .map(m => m.prediction.subcategories?.[key])
      .filter(sc => hasMeasuredScore(sc) && !sc.gap)
      .map(sc => sc!.predicted_score!);

  // Its standing in each scored area: the share of other models it is ahead of.
  const scoredAreas = Object.keys(taxonomy.subcategories).filter(key => hasMeasuredScore(prediction.subcategories?.[key]) && !prediction.subcategories?.[key]?.gap);
  const standing = Object.fromEntries(scoredAreas.map(key => {
    const field = areaField(key);
    const own = prediction.subcategories![key]!.predicted_score!;
    return [key, field.length ? field.filter(v => v < own).length / field.length : 0.5];
  }));
  // Areas mostly imputed from benchmarks the model did not run are left out of comparisons.
  const rankAreas = (keys: string[]) =>
    keys.filter(key => imputedShare(prediction.subcategories![key]!) < IMPUTED_SHARE_LIMIT).sort((a, b) => standing[b] - standing[a]);

  // Comparison groups for 3.1: models scored on all three indices, nearest first.
  const PEER_COUNT = 5;
  const indexThetas = (m: JoinedModelData) => INDICES.map(index => getIndexTheta(m, index.id)?.theta ?? null);
  const ownThetas = indexThetas(model);
  const scoreDistance = (m: JoinedModelData) => Math.hypot(...indexThetas(m).map((v, i) => v! - ownThetas[i]!));
  const candidates = ownThetas.some(v => v === null) ? [] : models.filter(m => m !== model && indexThetas(m).every(v => v !== null));
  const nearest = (pool: JoinedModelData[]) => [...pool].sort((a, b) => scoreDistance(a) - scoreDistance(b)).slice(0, PEER_COUNT);
  const activeParams = (m: JoinedModelData) => m.metadata?.specs?.active_params ?? null;
  const architecture = (m: JoinedModelData): 'MoE' | 'hybrid' | 'dense' | null => {
    const a = m.metadata?.specs?.architecture;
    return !a ? null : /moe|mixture[- ]of[- ]experts/i.test(a) ? 'MoE' : /hybrid/i.test(a) ? 'hybrid' : /dense/i.test(a) ? 'dense' : null;
  };
  const ARCH_NAMES = { MoE: 'mixture-of-experts', hybrid: 'hybrid-attention', dense: 'dense' } as const;
  const ownActive = activeParams(model);
  const ownArch = architecture(model);
  const ownOpen = model.metadata?.open_weights ?? null;
  const peerGroups: PeerGroup[] = [
    { id: 'closest', label: 'Closest scores', note: (n: number) => `The ${n} models with the closest scores on all three indices.`, peers: nearest(candidates) },
    ...(ownActive ? [{
      id: 'size',
      label: 'Similar size',
      note: (n: number) => `The ${n} models nearest in size, by active parameters (${formatParams(ownActive)} for this model).`,
      peers: candidates
        .filter(m => activeParams(m))
        .sort((a, b) => Math.abs(Math.log(activeParams(a)! / ownActive)) - Math.abs(Math.log(activeParams(b)! / ownActive)) || scoreDistance(a) - scoreDistance(b))
        .slice(0, PEER_COUNT),
      meta: (m: JoinedModelData) => (activeParams(m) ? `${formatParams(activeParams(m)!)} active` : null),
    }] : []),
    ...(ownArch ? [{
      id: 'architecture',
      label: `Also ${ownArch}`,
      note: (n: number) => `The ${n} ${ARCH_NAMES[ownArch]} models with the closest scores.`,
      peers: nearest(candidates.filter(m => architecture(m) === ownArch)),
    }] : []),
    ...(ownOpen !== null ? [{
      id: 'weights',
      label: ownOpen ? 'Also open weights' : 'Also proprietary',
      note: (n: number) => `The ${n} ${ownOpen ? 'open-weight' : 'proprietary'} models with the closest scores.`,
      peers: nearest(candidates.filter(m => m.metadata?.open_weights === ownOpen)),
    }] : []),
  ].filter(g => g.peers.length > 0);
  const peers = peerGroups[0]?.peers ?? [];

  const outcomeSummary = analysis.model_outcomes[model.id];
  const isVllm = /(^|\/)vllm\//.test(model.id);
  const predictionRecord = getPredictionRecord();
  const answersFrom = fromCalibration.length === 0 ? `the ${releaseLabel} run`
    : fromCoreRun.length === 0 ? 'the calibration run, which had already asked the core questions'
    : `the ${releaseLabel} run and, for ${benchmarks(fromCalibration.length)}, from the calibration run, which had already asked those questions`;
  const hasPredictionCheck = Object.values(prediction.tasks).some(t => t.full_run && !notRun(t) && t.predicted_score !== null && t.subset_items < t.population_items);
  const laterChapters = [
    ...(outcomeSummary ? [{ id: 'health', label: 'Run health' }] : []),
    ...(Object.keys(taskUsage).length ? [{ id: 'tokens', label: 'Cost' }] : []),
  ].map((chapter, i) => ({ ...chapter, number: `${i + 3}` }));
  const chapterNumber = (id: string) => laterChapters.find(c => c.id === id)?.number;

  const tasksByIndex = Object.fromEntries(
    INDICES.map(index => [
      index.id,
      catalog.filter(b => b.index === index.id && prediction.tasks[b.name]).map(b => [b.name, prediction.tasks[b.name]] as const),
    ])
  );

  const metadata = model.metadata;
  const specs = metadata?.specs;
  const modelFacts = ([
    ['Architecture', specs?.architecture],
    ['Context', specs?.context ? `${(specs.context / 1000).toFixed(0)}K tokens` : null],
    ['Licence', metadata?.license],
  ] as [string, React.ReactNode][]).filter(([, value]) => value);
  const evaluationFacts = ([
    ['Last evaluated', prediction.last_evaluated ? formatDate(prediction.last_evaluated) : null],
  ] as [string, React.ReactNode][]).filter(([, value]) => value);
  const evaluationSettings: [string, React.ReactNode][] = [
    ['Served via', formatRoute(model.id, modelConfig)],
    ['Reasoning', formatReasoning(modelConfig)],
    ['Max output tokens', modelConfig?.max_tokens ? modelConfig.max_tokens.toLocaleString() : '—'],
  ];

  return (
    <div className="frame pt-20 pb-10 print:pt-6">
    <div className="cols-10 space-y-14">
      <div>
        <Link href="/models" className="inline-block mb-[34px] text-sm font-semibold text-muted-foreground hover:text-link transition-colors print:hidden">
          ← All models
        </Link>

        <header>
          <div className="space-y-3 max-w-3xl">
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-ink">{name}</h1>
            <ModelMeta model={model} className="gap-2 text-sm" logoClassName="size-[18px]" showActive longDate />
            {metadata?.description && <p className="text-base text-muted-foreground leading-relaxed">{metadata.description}</p>}
          </div>
          <dl className="mt-6 flex flex-wrap gap-x-10 gap-y-3">
            {[...modelFacts, ...evaluationFacts].map(([label, value]) => <Fact key={label} label={label}>{value}</Fact>)}
          </dl>
          {metadata?.sources?.length ? (
            <p className="mt-4 text-[13px]">
              <span className="text-muted-foreground">Sources:</span>{' '}
              {metadata.sources.map((source, i) => (
                <span key={source.url} className="inline-block">
                  <a href={source.url} target="_blank" rel="noopener noreferrer" className="text-link font-medium hover:underline">
                    {source.title || source.url}
                  </a>
                  {i < metadata.sources!.length - 1 && <span className="text-muted-foreground">&nbsp;·&nbsp;</span>}
                </span>
              ))}
            </p>
          ) : null}
        </header>
      </div>

      <section>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {INDICES.map(index => <IndexCard key={index.id} models={models} model={model} index={index} />)}
        </div>
        <div className="mt-3 space-y-1.5">
          <BarKey tick={`Average model (${THETA_MEAN})`} point="This model, in points" />
          <p className="text-sm text-muted-foreground leading-relaxed max-w-3xl">
            {THETA_MEAN} is the average of the {models.length} evaluated models and 15 points is one standard deviation. Small numbers show the 90% interval.
          </p>
        </div>
      </section>

      {tokens && run && (
        <CostSection
          tokens={tokens}
          usd={usd}
          run={run}
          coreAnswered={coverage?.samples_completed ?? null}
          coreTotal={coverage?.samples_total ?? null}
          tasksTotal={tasksTotal}
        />
      )}

      <div className="lg:grid lg:grid-cols-[12rem_minmax(0,1fr)] lg:gap-12 print:block">
      <aside className="hidden lg:block print:hidden">
        <ReportToc
          items={[
            { id: 'settings', number: '1', label: 'Settings', depth: 0 },
            { id: 'results', number: '2', label: 'Results', depth: 0 },
            { id: 'benchmarks', number: '2.1', label: 'Benchmark results', depth: 1 },
            { id: 'areas', number: '2.2', label: 'Results by evaluation area', depth: 1 },
            ...INDICES.map((index, i) => ({ id: `area-${index.id}`, number: `2.2.${i + 1}`, label: index.label, depth: 2 as const })),
            ...(peers.length ? [{ id: 'peers', number: '2.3', label: 'Similar models', depth: 1 as const }] : []),
            ...laterChapters.map(chapter => ({ ...chapter, depth: 0 as const })),
          ]}
        />
      </aside>
      <div className="min-w-0 space-y-14">
      <section aria-labelledby="settings">
        <SectionHeading id="settings" number="1" title="Settings" />
        <div className={`mt-6 space-y-5 ${INDENT}`}>
          <dl className={`flex-wrap ${FACT_ROW}`}>
            {evaluationSettings.map(([label, value]) => <Fact key={label} label={label}>{value}</Fact>)}
            {coreRelease && <Fact label="CORE release">{releaseLabel}</Fact>}
            {predictionRecord && (
              <>
                <Fact label="Method"><span className="font-mono">{predictionRecord.method}</span></Fact>
                <Fact label="Parameters ID"><span className="font-mono">{predictionRecord.params_id}</span></Fact>
                <Fact label="Subset ID"><span className="font-mono">{predictionRecord.subset_id}</span></Fact>
                <Fact label="Prediction ID"><span className="font-mono">{predictionRecord.prediction_id}</span></Fact>
              </>
            )}
          </dl>
          <p id="reproduce" className="scroll-mt-28 pt-3 text-base text-muted-foreground leading-relaxed">
            To reproduce this evaluation with the same settings, run:
          </p>
          <CopyCommand lines={reproduceCommands(model.id, modelConfig)} />
          {isCalibration && (
            <p className="text-sm text-muted-foreground leading-relaxed">
              To answer every question, as in the calibration run, add <code className="font-mono">--full</code>.
            </p>
          )}
          <p className="text-sm text-muted-foreground leading-relaxed">
            <a href={README_QUICKSTART_URL} target="_blank" rel="noreferrer" className="font-semibold text-link hover:underline">Set up ComplAI</a> first.{' '}
            {isVllm ? (
              <>The model is served by vLLM: start a server first and pass its address with <code className="font-mono">--model-base-url</code>. Set </>
            ) : (
              <>Set the API key of the model’s provider{model.id.startsWith('openrouter/') && <> (<code className="font-mono">OPENROUTER_API_KEY</code>)</>}, </>
            )}
            <code className="font-mono">OPENAI_API_KEY</code> for the judge models and <code className="font-mono">HF_TOKEN</code> for gated datasets.
            {' '}Your predicted.json records its parameters and subset IDs, which should match these; its prediction ID also depends on the other models predicted with it.
            Answers vary between runs, so scores will differ slightly.
          </p>
        </div>
      </section>

      <section aria-labelledby="results">
        <SectionHeading id="results" number="2" title="Results" />
        <div className={INDENT}>
        <div className="mt-8">
        <SubHeading id="benchmarks" number="2.1" title="Benchmark results">
          The model’s score on each benchmark. Under each benchmark are the evaluation areas most of its
          questions fall in.{isCalibration && ' The model also has a calibration run, which answered every question; switch to it to see those scores and how close the CORE prediction came.'}
        </SubHeading>
        <div className={INDENT}>
        {(() => {
          const panel = (view: 'core' | 'full') => (
            <>
              <div className="space-y-1.5">
                {view === 'core' ? (
                  <p className="text-base text-ink leading-relaxed">
                    <strong className="font-semibold">{releaseLabel} run</strong> on {formatDateRange(answerDates(scoredTasks))}, across {benchmarks(scoredTasks.length)}.
                    {notRunCount > 0 && <> {benchmarks(notRunCount)} {notRunCount === 1 ? 'was' : 'were'} not run.</>}
                  </p>
                ) : (
                  <p className="text-base text-ink leading-relaxed">
                    <strong className="font-semibold">Calibration run</strong> on {formatDateRange(fullRunTasks.map(t => t.full_run!.run_created))}: every question of {benchmarks(fullRunTasks.length)}
                    {fullRunPartial > 0 && <>, {fullRunPartial} of them only in part</>}.
                    {tasksTotal - fullRunTasks.length > 0 && <> The other {benchmarks(tasksTotal - fullRunTasks.length)} {tasksTotal - fullRunTasks.length === 1 ? 'was' : 'were'} not part of it.</>}
                  </p>
                )}
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {view === 'core' ? (
                    <>
                      Each score is the predicted accuracy on all of a benchmark’s questions, from the model’s answers to its core questions, and ± is the
                      90% interval.{fromCalibration.length > 0 && ` ${fromCoreRun.length > 0 ? `For ${benchmarks(fromCalibration.length)}, the` : 'The'} core answers come from the calibration run, which had already asked those questions.`} The
                      grey bars show how the other evaluated models score, and the line marks this model. Under each benchmark are the core questions answered and the evaluated model’s tokens{fromCalibration.length > 0 ? ' in the CORE run; the calibration run’s tokens are under its tab' : ''}.
                    </>
                  ) : (
                    <>
                      Each score is the plain accuracy on the questions answered, with nothing predicted, and the CORE score is beside it for comparison.
                      {hasSeparateFullRun && (fromCalibration.length === 0 ? ` The CORE scores come from the separate ${releaseLabel} run.` : ' For some benchmarks the CORE score comes from a separate run of the core questions.')} The grey bars show the
                      calibration runs of the other {calibrationModels} models that have one, and the line marks this model. Under each benchmark are the questions answered and the evaluated model’s tokens in this run.
                    </>
                  )}
                </p>
              </div>
              <div className="mt-4 grid grid-cols-1 gap-4">
                {INDICES.map(index => (
                  <div key={index.id} className="rounded-2xl border border-border bg-card px-5 py-2">
                    <div className="pt-3 pb-1 text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">{index.label}</div>
                    {tasksByIndex[index.id].map(([task, t]) => (
                      <BenchmarkRow
                        key={task}
                        task={task}
                        name={benchmarkName(task)}
                        tests={benchmarkTests[task] ?? []}
                        t={t}
                        view={view}
                        usage={taskUsage[task]}
                        field={view === 'full' ? fullField(task) : taskField(task)}
                        tone={INDEX_BAR[index.id]}
                      />
                    ))}
                  </div>
                ))}
              </div>
              {view === 'full' && hasPredictionCheck && (
                <div className="mt-10">
                  <h4 className="text-base font-bold text-ink">Prediction check</h4>
                  <p className="mt-1 mb-4 text-base text-muted-foreground leading-relaxed">
                    The calibration run answered every question, so the CORE prediction can be checked against the plain score.
                  </p>
                  <PredictionCheck prediction={prediction} names={names} />
                </div>
              )}
            </>
          );
          return isCalibration ? (
            <RunTabs
              label="Results"
              tabs={[
                { id: 'core', label: releaseLabel, content: panel('core') },
                { id: 'full', label: 'Calibration run', content: panel('full') },
              ]}
            />
          ) : panel('core');
        })()}
        </div>
        </div>

        <div className="mt-14">
        <SubHeading id="areas" number="2.2" title="Results by evaluation area">
          Evaluation areas group the questions of the benchmarks by what they test, and link to the provisions of the EU AI Act and the
          GPAI Code of Practice that they inform, as set out in the <Link href="/mapping" className="text-link hover:underline">AI Act mapping</Link>.
          Each score is the predicted accuracy on the area’s questions, with its 90% interval. Areas mix questions of different
          difficulty, so their scores are not comparable with each other.
        </SubHeading>
        <div className="mt-8 space-y-10">
        {INDICES.map((index, i) => {
          const areas = Object.entries(taxonomy.subcategories).filter(([, s]) => s.index === index.id);
          const scored = areas.filter(([key]) => hasMeasuredScore(prediction.subcategories?.[key]) && !prediction.subcategories?.[key]?.gap);
          // The extremes are named when an index has 3+ areas.
          const ranked = rankAreas(scored.map(([key]) => key));
          const standout = (key: string): AreaStandout =>
            ranked.length < 3 ? null
            : key === ranked[0] && standing[key] > standing[ranked[1]] && standing[key] > 0.1 ? 'best'
            : key === ranked[ranked.length - 1] && standing[key] < standing[ranked[ranked.length - 2]] && standing[key] < 0.9 ? 'worst'
            : null;
          return (
            <div key={index.id}>
              <SubHeading id={`area-${index.id}`} number={`2.2.${i + 1}`} level={4} title={index.label} dot={INDEX_BAR[index.id].dot}>{index.description}</SubHeading>
              <div className={INDENT}>
                {scored.length > 0 && (
                  <AreaChart
                    tone={INDEX_BAR[index.id]}
                    rows={scored.map(([key, info]) => ({
                      key,
                      name: info.name,
                      field: areaField(key),
                      score: prediction.subcategories![key] as LabelScore & { predicted_score: number },
                      median: subcategoryMedians[key],
                    }))}
                  />
                )}
                {scored.map(([key, info]) => (
                  <AreaBlock
                    key={key}
                    areaKey={key}
                    name={info.name}
                    description={areaDescriptions[key] ?? info.description}
                    refs={(grounding.subcategories[key]?.refs ?? [])
                      .filter(r => r.coverage !== 'none' && grounding.refs[r.id])
                      .map(r => ({ id: r.id, label: citeLabel(grounding.refs[r.id]), title: grounding.refs[r.id].title, direct: r.strength === 'direct' }))
                      .sort((a, b) => Number(b.direct) - Number(a.direct))}
                    sources={areaSources[key] ?? []}
                    prediction={prediction}
                    score={prediction.subcategories![key] as LabelScore & { predicted_score: number }}
                    panelMedian={subcategoryMedians[key]}
                    field={areaField(key)}
                    standout={standout(key)}
                    indexLabel={index.label}
                  />
                ))}
              </div>
            </div>
          );
        })}
        </div>
        </div>
        {peers.length > 0 && (
          <div className="mt-14">
            <SubHeading id="peers" number="2.3" title="Comparison with similar models">
              How the model compares with others on the three indices. Choose which models to compare it with.
            </SubHeading>
            <div className={INDENT}>
              <RunTabs
                label="Models to compare with"
                tabs={peerGroups.map(g => ({ id: g.id, label: g.label, content: <PeerChart model={model} group={g} /> }))}
              />
            </div>
          </div>
        )}
        </div>
      </section>

      {outcomeSummary && (
        <section aria-labelledby="health">
          <SectionHeading id="health" number={chapterNumber('health')} title="Run health">
            How the model’s {Object.values(analysis.models[model.id] ?? {}).reduce((sum, e) => sum + e.answers, 0).toLocaleString()} answers to the core questions ended, against
            the median evaluated model, and why answers didn’t finish or were blocked. The answers come from {answersFrom}. Answers that didn’t finish were not scored as correct, so they lower scores like wrong answers.
          </SectionHeading>
          <div className={`mt-8 ${INDENT} space-y-6`}>
            <RunHealth analysis={analysis} modelId={model.id} />
            <div><FailedAnswers analysis={analysis} modelId={model.id} names={names} /></div>
          </div>
        </section>
      )}

      {Object.keys(taskUsage).length > 0 && (
        <section aria-labelledby="tokens">
          <SectionHeading id="tokens" number={chapterNumber('tokens')} title="Cost">
            The benchmarks that took the most of the model’s tokens in CORE, against answering every question.
          </SectionHeading>
          <div className={`mt-8 ${INDENT}`}><TokenSpend taskUsage={taskUsage} names={names} /></div>
        </section>
      )}
      </div>
      </div>

    </div>
    </div>
  );
}
