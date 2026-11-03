import Link from 'next/link';
import { notFound } from 'next/navigation';
import { IntervalBar } from '@/components/IntervalBar';
import { RunBadge } from '@/components/RunBadge';
import { getBenchmarkLabels, getJoinedModels, getTaxonomy, type JoinedModelData, type LabelScore } from '@/lib/data';
import { EYEBROW_CLASS } from '@/components/layout/Eyebrow';
import {
  hasMeasuredScore,
  IMPUTED_SHARE_LIMIT,
  INDICES,
  NO_SCORE_REASON,
  RUN_TYPES,
  TASK_STATUS,
  THETA_MEAN,
  formatAccuracyInterval,
  formatInterval,
  formatPercent,
  formatRankRange,
  formatScore,
  formatTokens,
  getIndexTheta,
  getRunType,
  imputedShare,
  rankRange,
  scaleTheta,
} from '@/lib/scores';

// Same fixed range as the leaderboard bars, in scaled points.
const SCORE_MIN = 40;
const SCORE_MAX = 150;

export function generateStaticParams() {
  return getJoinedModels().map((model) => ({ id: model.yamlId }));
}

function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function formatParams(model: JoinedModelData): string {
  const specs = model.metadata?.specs;
  const total = specs?.total_params;
  if (!total) return '—';
  const fmt = (n: number) => `${(n / 1e9).toFixed(0)}B`;
  const active = specs?.active_params;
  return active && active !== total ? `${fmt(total)} (${fmt(active)} active)` : fmt(total);
}

function SectionHeading({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="mb-4">
      <h2 className="text-xl font-extrabold tracking-tight text-ink">{title}</h2>
      {children && <p className="mt-1 text-sm text-muted-foreground leading-relaxed max-w-3xl">{children}</p>}
    </div>
  );
}

function IndexCard({ models, model, index }: { models: JoinedModelData[]; model: JoinedModelData; index: (typeof INDICES)[number] }) {
  const theta = getIndexTheta(model, index.id);
  const rank = rankRange(models, model, index.id);
  return (
    <div className="rounded-2xl border border-border bg-white p-5" title={index.description}>
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
            min={SCORE_MIN}
            max={SCORE_MAX}
            reference={THETA_MEAN}
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

function AreaRow({ name, description, score, panelMedian }: {
  name: string;
  description: string;
  score: LabelScore | undefined;
  panelMedian: number | null;
}) {
  let body: React.ReactNode;
  if (!score || score.gap) {
    body = (
      <span className="text-xs text-muted-foreground">
        Not scored: too few CORE items cover this evaluation area{score?.gap ? ` (${score.gap})` : ''}.
      </span>
    );
  } else if (!hasMeasuredScore(score)) {
    body = <span className="text-xs text-muted-foreground">Not measured: the model answered none of this area’s CORE items.</span>;
  } else {
    const mostlyImputed = imputedShare(score) >= IMPUTED_SHARE_LIMIT;
    const interval = score.score_interval;
    body = (
      <div className="flex flex-col gap-1.5">
        <div className="flex items-baseline justify-between gap-3">
          <span className={`text-lg font-bold tabular ${mostlyImputed ? 'text-ink/50' : 'text-ink'}`}>{formatPercent(score.predicted_score)}</span>
          {interval && <span className="text-xs text-muted-foreground tabular">{formatAccuracyInterval(interval)}</span>}
        </div>
        <IntervalBar interval={interval} point={score.predicted_score} min={0} max={1} reference={panelMedian ?? undefined} muted={mostlyImputed} />
        <div className="text-[11px] text-muted-foreground tabular">
          {score.observations.toLocaleString()} of {score.subset_items.toLocaleString()} core items answered
        </div>
        {mostlyImputed && (
          <div className="text-[11px] font-semibold text-brand-strong">
            {formatPercent(imputedShare(score))} estimated from benchmarks the model did not run; the interval understates the uncertainty.
          </div>
        )}
      </div>
    );
  }
  return (
    <div className="grid grid-cols-1 sm:grid-cols-[minmax(0,1fr)_minmax(0,18rem)] gap-x-6 gap-y-2 py-4 border-b border-border last:border-0">
      <div>
        <div className="text-sm font-semibold text-ink">{name}</div>
        <div className="mt-0.5 text-xs text-muted-foreground leading-relaxed">{description}</div>
      </div>
      <div>{body}</div>
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4 py-2 border-b border-border last:border-0 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium text-ink text-right">{value}</span>
    </div>
  );
}

export default async function ModelReportPage(props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const models = getJoinedModels();
  const model = models.find(m => m.yamlId === params.id);
  if (!model) notFound();

  const taxonomy = getTaxonomy();
  const benchmarkLabels = getBenchmarkLabels();
  const prediction = model.prediction;
  const run = getRunType(model);
  const tokens = prediction.usage?.tokens ?? null;
  const taskUsage = prediction.usage?.tasks ?? {};
  const coverage = prediction.coverage;
  const name = model.metadata?.name || model.yamlId;

  const subcategoryMedians = Object.fromEntries(
    Object.keys(taxonomy.subcategories).map(key => [
      key,
      median(models.map(m => m.prediction.subcategories?.[key]).filter(hasMeasuredScore).map(s => s.predicted_score)),
    ])
  );

  const tasksByIndex = Object.fromEntries(
    INDICES.map(index => [
      index.id,
      Object.entries(prediction.tasks)
        .filter(([task]) => benchmarkLabels[task]?.index === index.id)
        .sort(([a], [b]) => a.localeCompare(b)),
    ])
  );
  const tasksRun = Object.values(prediction.tasks).filter(t => t.status !== 'imputed' && t.status !== 'missing').length;

  return (
    <div className="container mx-auto px-4 max-w-6xl py-10 space-y-12">
      <Link href="/models" className="text-sm font-semibold text-muted-foreground hover:text-brand-strong transition-colors">
        ← All models
      </Link>

      <header className="space-y-3 max-w-3xl">
        <div className={`flex flex-wrap items-center gap-2 ${EYEBROW_CLASS}`}>
          <span>{model.metadata?.organization || 'Unknown organization'}</span>
          {model.metadata?.release_date && (
            <>
              <span aria-hidden>·</span>
              <span>Released {model.metadata.release_date}</span>
            </>
          )}
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-ink">{name}</h1>
        {model.metadata?.description && (
          <p className="text-base text-muted-foreground leading-relaxed">{model.metadata.description}</p>
        )}
        {run && (
          <div className="flex items-start gap-2 text-sm text-muted-foreground">
            <RunBadge run={run} />
            <span className="leading-5">{RUN_TYPES[run].description}</span>
          </div>
        )}
      </header>

      <section>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {INDICES.map(index => <IndexCard key={index.id} models={models} model={model} index={index} />)}
        </div>
        <p className="mt-3 text-xs text-muted-foreground leading-relaxed max-w-3xl">
          {THETA_MEAN} is the average of the {models.length} evaluated models and 15 points is one standard deviation; the tick marks {THETA_MEAN}. Small numbers and shaded bars show the 90% interval.
        </p>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
        <div className="lg:col-span-8 space-y-12">
          <section>
            <SectionHeading title="Evaluation areas">
              Predicted accuracy on all items of each evaluation area, with its 90% interval. Areas mix benchmarks of different
              difficulty, so these percentages are not comparable across areas; the tick marks the median model.
            </SectionHeading>
            <div className="space-y-6">
              {INDICES.map(index => {
                const subcategories = Object.entries(taxonomy.subcategories).filter(([, s]) => s.index === index.id);
                return (
                  <div key={index.id} className="rounded-2xl border border-border bg-white px-5 py-2">
                    <div className="pt-3 pb-1 text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">{index.label}</div>
                    {subcategories.map(([key, info]) => (
                      <AreaRow
                        key={key}
                        name={info.name}
                        description={info.description}
                        score={prediction.subcategories?.[key]}
                        panelMedian={subcategoryMedians[key]}
                      />
                    ))}
                  </div>
                );
              })}
            </div>
          </section>

          <section>
            <SectionHeading title="Benchmarks">
              {tasksRun} of {Object.keys(prediction.tasks).length} benchmarks run. Scores are predicted accuracy on all of a benchmark’s items,
              from the model’s answers on the core items and predictions for the rest, so every model is scored the same way whatever it ran;
              ± shows the 90% range of those predictions. Benchmarks that were not run have no score here.
            </SectionHeading>
            <div className="rounded-2xl border border-border bg-white overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    <th className="px-4 py-3 text-left">Benchmark</th>
                    <th className="px-3 py-3 text-left">Status</th>
                    <th className="px-3 py-3 text-right">Items answered</th>
                    <th className="px-3 py-3 text-right">Score</th>
                    <th className="px-4 py-3 text-right">Tokens</th>
                  </tr>
                </thead>
                {INDICES.map(index => (
                  <tbody key={index.id}>
                    <tr className="bg-muted/60">
                      <td colSpan={5} className="px-4 py-2 text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">{index.label}</td>
                    </tr>
                    {tasksByIndex[index.id].map(([task, t]) => {
                      const status = TASK_STATUS[t.status] ?? TASK_STATUS.missing;
                      const notRun = t.status === 'imputed' || t.status === 'missing';
                      const usage = taskUsage[task];
                      const answered = usage?.observed_items ?? t.observations;
                      const margin = t.predicted_score_error ? 1.645 * t.predicted_score_error : 0;
                      const taskTokens = usage ? usage.model_tokens : null;
                      return (
                        <tr key={task} className="border-b border-border last:border-0">
                          <td className="px-4 py-2.5 font-medium text-ink">{task}</td>
                          <td className="px-3 py-2.5">
                            <div className="flex flex-wrap gap-1">
                              <span title={status.description} className={`inline-flex items-center whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-semibold leading-4 ${status.className}`}>
                                {status.label}
                              </span>
                              {usage && !notRun && usage.run !== 'partial' && <RunBadge run={usage.run} short />}
                            </div>
                          </td>
                          <td className="px-3 py-2.5 text-right tabular text-xs text-muted-foreground whitespace-nowrap">
                            {notRun ? '—' : `${answered.toLocaleString()} / ${t.population_items.toLocaleString()}`}
                          </td>
                          <td
                            className="px-3 py-2.5 text-right tabular whitespace-nowrap"
                            title={margin ? '90% range of the predictions for the items outside the model’s core answers.' : undefined}
                          >
                            {notRun || t.predicted_score === null ? (
                              <span className="text-muted-foreground/60">—</span>
                            ) : (
                              <>
                                <span className="font-semibold text-ink">{formatPercent(t.predicted_score)}</span>
                                {margin >= 0.005 && <span className="ml-1 text-[11px] text-muted-foreground">±{Math.round(margin * 100)}</span>}
                              </>
                            )}
                          </td>
                          <td className="px-4 py-2.5 text-right tabular text-xs text-muted-foreground whitespace-nowrap">
                            {taskTokens ? formatTokens(taskTokens) : '—'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                ))}
              </table>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              Tokens of the evaluated model, measured from the run logs; judge models are not counted. Core runs answered only the core subset of a benchmark; calibration runs answered all its items.
            </p>
          </section>
        </div>

        <aside className="lg:col-span-4 space-y-6">
          <div className="rounded-2xl border border-border bg-white p-5">
            <h2 className="text-lg font-extrabold tracking-tight text-ink">Evaluation cost</h2>
            {tokens && run ? (
              <>
                <div className="mt-3 font-serif text-3xl font-extrabold tracking-tight tabular text-teal-strong">
                  {run !== 'core' ? '~' : ''}{formatTokens(tokens.core)}
                </div>
                <div className="text-sm text-muted-foreground">tokens on the core subset{run !== 'core' ? ' (estimated)' : ' (measured)'}</div>
                <div className="relative mt-4 h-2 rounded-full bg-muted" aria-hidden>
                  <div className="absolute inset-y-0 left-0 rounded-full bg-teal" style={{ width: `${Math.max((tokens.core / tokens.full) * 100, 2)}%` }} />
                </div>
                <div className="mt-2 flex justify-between text-xs text-muted-foreground tabular">
                  <span>vs {run !== 'full' ? '~' : ''}{formatTokens(tokens.full)} on all items{run !== 'full' ? ' (estimated)' : ' (measured)'}</span>
                  {tokens.reduction !== null && <span className="font-semibold text-teal-strong">−{formatPercent(tokens.reduction)}</span>}
                </div>
                <div className="mt-4">
                  <DetailRow label="Tokens measured" value={formatTokens(tokens.measured)} />
                  <DetailRow label="Benchmarks with token counts" value={tokens.tasks} />
                </div>
              </>
            ) : (
              <p className="mt-2 text-sm text-muted-foreground">The logs for this run have no token counts.</p>
            )}
            {coverage && (
              <div className={tokens ? '' : 'mt-3'}>
                <DetailRow label="Core items answered" value={`${coverage.samples_completed.toLocaleString()} / ${coverage.samples_total.toLocaleString()}`} />
                <DetailRow label="Items answered in total" value={coverage.population_samples_observed.toLocaleString()} />
              </div>
            )}
            <Link href="/methodology#efficiency" className="mt-3 inline-block text-xs font-semibold text-brand-strong hover:underline">How we estimate this</Link>
          </div>

          <div className="rounded-2xl border border-border bg-white p-5">
            <h2 className="text-lg font-extrabold tracking-tight text-ink">Model details</h2>
            <div className="mt-2">
              <DetailRow label="Parameters" value={formatParams(model)} />
              <DetailRow label="Architecture" value={model.metadata?.specs?.architecture || '—'} />
              <DetailRow label="Open weights" value={model.metadata?.open_weights == null ? '—' : model.metadata.open_weights ? 'Yes' : 'No'} />
              <DetailRow label="Context window" value={model.metadata?.specs?.context ? `${(model.metadata.specs.context / 1000).toFixed(0)}K` : '—'} />
              <DetailRow label="License" value={model.metadata?.license || '—'} />
            </div>
            {model.metadata?.sources?.length ? (
              <div className="mt-4">
                <div className="text-sm text-muted-foreground mb-2">Sources</div>
                <div className="flex flex-col gap-1.5">
                  {model.metadata.sources.map(source => (
                    <a key={source.url} href={source.url} target="_blank" rel="noopener noreferrer" className="text-sm font-medium text-brand-strong hover:underline truncate">
                      {source.title || source.url}
                    </a>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </aside>
      </div>
    </div>
  );
}
