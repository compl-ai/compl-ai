import { OutcomeBar, OutcomeLegend, OutcomeShares } from '@/components/OutcomeBar';
import Link from 'next/link';
import { dataSourceUrl, type BenchmarkHealth, type BenchmarkMeta, type SampleAnalysis } from '@/lib/data';
import { SATURATION } from '@/lib/outcomes';
import { INDEX_TONE, INDICES, formatPercent, formatRelease } from '@/lib/scores';
import { BaseCard, BaseCardTitle, MicroLabel, Stat, StatGrid } from '@/components/ui/base-card';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card';
import { benchmarkAnchor, formatRatio, type CatalogBenchmark } from '@/lib/catalog';

const pct = (v: number) => `${Math.round(v * 100)}`;
const BANDS = 20;
// Area tags shown on a card before the rest collapse into a +N tag, so every card footer is one line.
const VISIBLE_AREAS = 2;
const TAG = 'whitespace-nowrap rounded-full border border-border bg-muted/40 px-2 py-0.5 text-[11px] font-medium text-ink/80';
const CHART_H = 48;

// Describes the score spread in words from the exported means of the top, middle and bottom third.
function spreadSentence(saturation: string, m: { top: number; middle: number; bottom: number }) {
  if (saturation === 'saturated') {
    return `Even the middle third of models averages ${formatPercent(m.middle)}; the bottom third ${formatPercent(m.bottom)}.`;
  }
  return `The top third averages ${formatPercent(m.top)}, the middle ${formatPercent(m.middle)} and the bottom ${formatPercent(m.bottom)}.`;
}

// Light [1, 2, 1] smoothing of the band counts, for the shape of the line only: with ~50 models
// in 20 bands, single models otherwise show up as spikes. Hover titles keep the real counts.
function smooth(counts: number[]) {
  return counts.map((c, i) => (counts[Math.max(0, i - 1)] + 2 * c + counts[Math.min(counts.length - 1, i + 1)]) / 4);
}

// Smooth line through the band counts (at band centres, held flat to the edges). Monotone cubic,
// so the curve never dips below zero or overshoots a peak between two bands.
function curve(counts: number[], peak: number) {
  const n = counts.length;
  const xs = [0, ...counts.map((_, i) => ((i + 0.5) * 100) / n), 100];
  const ys = [counts[0], ...counts, counts[n - 1]].map(c => CHART_H - 2 - (c / peak) * (CHART_H - 4));
  const d = xs.slice(1).map((x, i) => (ys[i + 1] - ys[i]) / (x - xs[i]));
  const m = xs.map((_, i) => {
    if (i === 0) return d[0];
    if (i === xs.length - 1) return d[d.length - 1];
    return d[i - 1] * d[i] <= 0 ? 0 : (2 * d[i - 1] * d[i]) / (d[i - 1] + d[i]);
  });
  let path = `M${xs[0]},${ys[0].toFixed(2)}`;
  for (let i = 0; i < xs.length - 1; i++) {
    const h = (xs[i + 1] - xs[i]) / 3;
    path += ` C${(xs[i] + h).toFixed(2)},${(ys[i] + m[i] * h).toFixed(2)} ${(xs[i + 1] - h).toFixed(2)},${(ys[i + 1] - m[i + 1] * h).toFixed(2)} ${xs[i + 1].toFixed(2)},${ys[i + 1].toFixed(2)}`;
  }
  return path;
}

// Counts the scores in BANDS equal bands; the top band includes 100%. Falls back to the exported
// 10 bands if the scores don't cover the same models.
function bands(scores: number[] | undefined, models: number, exported: { low: number; high: number; count: number }[]) {
  if (!scores || scores.length !== models) return exported;
  const counts = Array<number>(BANDS).fill(0);
  for (const v of scores) counts[Math.min(BANDS - 1, Math.floor(v * BANDS))] += 1;
  return counts.map((count, i) => ({ low: i / BANDS, high: (i + 1) / BANDS, count }));
}

// Release · Paper · Code · Data, from public/benchmarks.yaml. Each part only shows when filled in.
function BenchmarkMetaLine({ meta }: { meta?: BenchmarkMeta }) {
  if (!meta) return null;
  const data = dataSourceUrl(meta.data_source);
  const links = [
    meta.paper && { key: 'paper', label: 'Paper', href: meta.paper },
    meta.repo && { key: 'code', label: 'Code', href: meta.repo },
    data && { key: 'data', label: 'Data', href: data, title: meta.data_source?.id },
  ].filter(Boolean) as { key: string; label: string; href: string; title?: string }[];
  const parts = [
    ...(meta.release_date ? [<span key="release" title={`Released ${meta.release_date}`}>{formatRelease(meta.release_date)}</span>] : []),
    ...links.map(l => (
      <a key={l.key} href={l.href} title={l.title} target="_blank" rel="noreferrer" className="hover:text-link hover:underline">
        {l.label}
      </a>
    )),
  ];
  if (!parts.length) return null;
  return (
    <p className="mt-1.5 flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
      {parts.map((part, i) => (
        <span key={i} className="whitespace-nowrap">
          {part}{i < parts.length - 1 && ' ·'}
        </span>
      ))}
    </p>
  );
}

export function BenchmarkHealthGrid({ health, analysis, scores, catalog, meta }: {
  health: BenchmarkHealth;
  analysis: SampleAnalysis;
  scores: Record<string, number[]>;
  catalog: CatalogBenchmark[];
  meta: Record<string, BenchmarkMeta>;
}) {
  const definition = analysis.outcomes.definition;
  const byName = Object.fromEntries(catalog.map(b => [b.name, b]));
  return (
    <div className="space-y-10">
      <details className="group rounded-xl border border-border/70 bg-card px-4 py-3">
        <summary className="flex cursor-pointer list-none items-center gap-2 text-sm font-semibold text-ink [&::-webkit-details-marker]:hidden">
          <span className="inline-block text-muted-foreground transition-transform group-open:rotate-90" aria-hidden>›</span>
          How to read these cards
        </summary>
      <dl className="mt-3 grid gap-x-8 gap-y-4 text-sm leading-relaxed text-muted-foreground md:grid-cols-2 xl:grid-cols-4">
        <div>
          <dt className="font-semibold text-ink">Answer outcomes</dt>
          <dd className="mt-1">How all models’ answers to the core questions ended. Hover a bar for the reasons.</dd>
          <dd className="mt-2">
            <OutcomeLegend definition={definition} only={['correct', 'wrong', 'did_not_finish', 'blocked']} />
          </dd>
        </div>
        <div>
          <dt className="font-semibold text-ink">Signal</dt>
          <dd className="mt-1">How closely the score follows a model’s overall level on the index.</dd>
          <dd className="mt-1">
            <strong className="text-ink">Above 1×:</strong> stronger models reliably score higher, and the benchmark counts more.
          </dd>
          <dd className="mt-1">
            <strong className="text-ink">Below 1×:</strong> it follows ability less closely and counts less.
          </dd>
          <dd className="mt-1">Measured across all models, so a benchmark with strong signal can still be saturated among the top models.</dd>
        </div>
        <div>
          <dt className="font-semibold text-ink">Compression</dt>
          <dd className="mt-1">How many times fewer questions CORE runs than the full benchmark.</dd>
        </div>
        <div>
          <dt className="font-semibold text-ink">Score spread</dt>
          <dd className="mt-1">Models per 5-point band of predicted score.</dd>
          <dd className="mt-1">
            <strong className={SATURATION.not_saturated.className}>Separates models:</strong> scores are still spread out.
          </dd>
          <dd className="mt-1">
            <strong className={SATURATION.saturated_at_top.className}>Saturated at the top:</strong> the top third averages 90% or more.
          </dd>
          <dd className="mt-1">
            <strong className={SATURATION.saturated.className}>Saturated:</strong> the middle third of models averages 95% or more.
          </dd>
        </div>
      </dl>
      </details>

      {INDICES.map(index => {
        const tasks = Object.entries(health.tasks)
          .filter(([, t]) => t.index === index.id)
          .sort((a, b) => a[0].localeCompare(b[0]));
        if (!tasks.length) return null;
        const tone = INDEX_TONE[index.id];
        // Signal is shown relative to the geometric mean discrimination of the index, the centre of the
        // log-normal prior on the discriminations, so 1.00× is an average benchmark of that index.
        const signalMean = Math.exp(
          tasks.reduce((sum, [, t]) => sum + Math.log(t.benchmark_discrimination), 0) / tasks.length,
        );
        return (
          <div key={index.id} data-index={index.id} className="space-y-4">
            <h3 className="flex items-center gap-2 text-lg font-extrabold tracking-tight text-ink">
              <span className={`size-2.5 rounded-full ${tone.bar}`} aria-hidden />
              {index.label}
            </h3>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {tasks.map(([name, t]) => {
                const saturation = t.saturation ? SATURATION[t.saturation] : null;
                const outcomes = analysis.task_outcomes[name];
                const histogram = bands(scores[name], t.models, t.score_histogram);
                const shape = smooth(histogram.map(b => b.count));
                const line = curve(shape, Math.max(1, ...shape));
                const b = byName[name];
                const m = meta[name];
                return (
                  <BaseCard key={name} id={benchmarkAnchor(name)} className="scroll-mt-24 gap-3 sm:gap-4 target:ring-2 target:ring-ink/30">
                    <div>
                      <BaseCardTitle className="[overflow-wrap:anywhere]">{m?.name ?? name}</BaseCardTitle>
                      <BenchmarkMetaLine meta={m} />
                    </div>

                    {outcomes && (
                      <div>
                        <MicroLabel>Answer outcomes</MicroLabel>
                        <OutcomeBar
                          shares={outcomes.outcomes}
                          definition={definition}
                          footer={`The answers of ${outcomes.models} models to ${t.core_items} core questions.`}
                        />
                        <OutcomeShares shares={outcomes.outcomes} definition={definition} />
                      </div>
                    )}

                    <StatGrid>
                      <Stat
                        label="Signal"
                        value={`${(t.benchmark_discrimination / signalMean).toFixed(2)}×`}
                        sub="vs index average"
                        title={
                          <>
                            <span className="block">How closely the score follows a model’s overall level on {index.label}, compared with the average {index.label} benchmark.</span>
                            <span className="mt-1.5 block">
                              <strong>Above 1×:</strong> stronger models reliably score higher here, and the benchmark counts more in the index.
                            </span>
                            <span className="mt-1 block">
                              <strong>Below 1×:</strong> the score follows ability less closely, and the benchmark counts less.
                            </span>
                          </>
                        }
                      />
                      <Stat
                        label="Models"
                        value={t.models}
                        sub="evaluated"
                        title="Models that answered at least one of the benchmark’s core questions. The chart below shows their predicted scores."
                      />
                      {b?.coreItems ? (
                        <Stat
                          label="Compression"
                          value={formatRatio(b.populationItems / b.coreItems)}
                          sub={`${b.coreItems.toLocaleString()} of ${b.populationItems.toLocaleString()}`}
                          title={
                            <>
                              CORE runs {b.coreItems.toLocaleString()} of the benchmark’s {b.populationItems.toLocaleString()} questions.
                              Each question is estimated at the benchmark’s median token use, so this is also the estimated token saving.
                            </>
                          }
                        />
                      ) : null}
                    </StatGrid>

                    <div>
                      <svg
                        viewBox={`0 0 100 ${CHART_H}`}
                        preserveAspectRatio="none"
                        className="block h-12 w-full overflow-visible text-ink/55"
                        role="img"
                        aria-label={`Predicted scores of ${t.models} models in ${100 / histogram.length}-point bands`}
                      >
                        <defs>
                          <linearGradient id={`health-fill-${name}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="currentColor" stopOpacity={0.4} />
                            <stop offset="100%" stopColor="currentColor" stopOpacity={0.03} />
                          </linearGradient>
                        </defs>
                        <path d={`${line} L100,${CHART_H} L0,${CHART_H} Z`} fill={`url(#health-fill-${name})`} />
                        <line x1="0" y1={CHART_H} x2="100" y2={CHART_H} className="stroke-track" strokeWidth={1} vectorEffect="non-scaling-stroke" />
                        <path d={line} fill="none" stroke="currentColor" strokeWidth={1.25} strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
                        {histogram.map((b, i) => (
                          <rect key={b.low} x={(i * 100) / histogram.length} y={0} width={100 / histogram.length} height={CHART_H} fill="transparent">
                            <title>{`${b.count} ${b.count === 1 ? 'model' : 'models'} scored ${pct(b.low)}–${pct(b.high)}%`}</title>
                          </rect>
                        ))}
                      </svg>
                      <div className="mt-1 flex justify-between text-[10px] tabular text-muted-foreground">
                        <span>0%</span>
                        <span>50%</span>
                        <span>100%</span>
                      </div>
                      {saturation && (
                        <p className="mt-2 text-xs leading-relaxed text-muted-foreground" title={saturation.description}>
                          <span className={`font-semibold ${saturation.className}`}>{saturation.label}.</span>{' '}
                          {spreadSentence(t.saturation!, t.third_means)}
                        </p>
                      )}
                    </div>

                    {b && b.areas.length > 0 && (
                      <div className="mt-auto flex items-center gap-1.5 overflow-hidden border-t border-border/60 pt-3">
                        {b.areas.slice(0, VISIBLE_AREAS).map(a => (
                          <Link
                            key={a.key}
                            href={`/mapping#${a.slug}`}
                            title={`${a.items.toLocaleString()} of its questions are labelled ${a.name}`}
                            className={`${TAG} min-w-0 truncate hover:border-ink/40 hover:text-ink`}
                          >
                            {a.name}
                          </Link>
                        ))}
                        {b.areas.length > VISIBLE_AREAS && (
                          <HoverCard>
                            <HoverCardTrigger
                              render={<span />}
                              delay={100}
                              closeDelay={200}
                              className={`${TAG} shrink-0 cursor-default tabular`}
                            >
                              +{b.areas.length - VISIBLE_AREAS}
                            </HoverCardTrigger>
                            <HoverCardContent className="w-auto max-w-64 p-2">
                              <p className="mb-1.5 text-[11px] text-muted-foreground">Also labelled</p>
                              <div className="flex flex-wrap gap-1.5">
                                {b.areas.slice(VISIBLE_AREAS).map(a => (
                                  <Link
                                    key={a.key}
                                    href={`/mapping#${a.slug}`}
                                    title={`${a.items.toLocaleString()} of its questions are labelled ${a.name}`}
                                    className={`${TAG} hover:border-ink/40 hover:text-ink`}
                                  >
                                    {a.name}
                                  </Link>
                                ))}
                              </div>
                            </HoverCardContent>
                          </HoverCard>
                        )}
                      </div>
                    )}
                  </BaseCard>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
