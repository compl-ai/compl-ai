import type { Metadata } from 'next';
import { pageMetadata } from '@/lib/site';
import Link from 'next/link';
import { getBenchmarkHealth, getBenchmarkMeta, getPredictions, getSampleAnalysis, getTaxonomy } from '@/lib/data';
import { INDICES, type IndexId } from '@/lib/scores';
import { Eyebrow } from '@/components/layout/Eyebrow';
import { BenchmarkHealthGrid } from '@/components/BenchmarkHealth';
import { IndexScope, IndexStrip, type IndexFilter, type StripStat } from '@/components/IndexStrip';
import { getBenchmarkCatalog, type CatalogBenchmark } from '@/lib/catalog';

export const metadata: Metadata = pageMetadata('Benchmarks', 'Every benchmark in ComplAI CORE: what it measures, how much it counts and how far CORE compresses it.');

export default function BenchmarksPage() {
  const taxonomy = getTaxonomy();
  const catalog = getBenchmarkCatalog();
  // Predicted scores per benchmark of the models that answered at least one of its core questions,
  // the same models benchmark_health.json counts, so the health cards can show finer bands than its 10.
  const healthScores: Record<string, number[]> = {};
  for (const m of Object.values(getPredictions().models ?? {}) as { tasks?: Record<string, { predicted_score: number | null; observations: number }> }[]) {
    for (const [task, t] of Object.entries(m.tasks ?? {})) {
      if (t.observations > 0 && t.predicted_score !== null) (healthScores[task] ??= []).push(t.predicted_score);
    }
  }

  const totalItems = catalog.reduce((sum, b) => sum + b.populationItems, 0);
  const stripStats = (benchmarks: CatalogBenchmark[], coreItems: number | undefined): StripStat[] => {
    const fullItems = benchmarks.reduce((sum, b) => sum + b.populationItems, 0);
    return [
      {
        label: 'Benchmarks',
        value: benchmarks.length.toLocaleString(),
        // Sized by CORE questions: full-suite sizes range from 30 to 58,140, which would hide most benchmarks.
        segments: benchmarks.filter(b => b.coreItems).map(b => ({
          key: b.name,
          index: b.index as IndexId,
          size: b.coreItems!,
          title: (
            <>
              <span className="font-semibold">{b.name}</span>: {b.coreItems!.toLocaleString()} CORE questions, from{' '}
              {b.populationItems.toLocaleString()} in the full benchmark.
            </>
          ),
        })),
      },
      { label: 'Full suite', value: fullItems.toLocaleString(), title: 'Questions in the full benchmarks, the pool CORE is calibrated on.' },
      { label: 'CORE', value: coreItems?.toLocaleString() ?? '—', title: 'Questions a new model actually runs. Its results on the full suite are predicted from them.' },
    ];
  };
  const stats = {
    all: stripStats(catalog, taxonomy.total_subset_items),
    ...Object.fromEntries(INDICES.map(index => [
      index.id,
      stripStats(catalog.filter(b => b.index === index.id), taxonomy.indices[index.id]?.subset_count),
    ])),
  } as Record<IndexFilter, StripStat[]>;
  const link = 'font-semibold text-link hover:underline';

  return (
    <div className="frame">
    <IndexScope className="cols-10 pt-10 sm:pt-22 pb-12 space-y-12">
      <section aria-labelledby="benchmarks-heading" className="space-y-8">
        <header className="max-w-5xl">
          <Eyebrow>Benchmarks</Eyebrow>
          <h1 id="benchmarks-heading" className="text-4xl md:text-5xl font-extrabold tracking-tight text-ink">All benchmarks</h1>
          <p className="mt-4 text-base text-muted-foreground leading-relaxed">
            ComplAI CORE scores every model on three independent indices: Capability, Reliability and Safety. Each index is
            built from established benchmarks, {catalog.length} in total with {totalItems.toLocaleString()} questions. A new
            model runs {taxonomy.total_subset_items.toLocaleString()} of these questions, and CORE predicts how it would do on
            the rest. Every question is also labelled with the ability or behaviour it tests; the{' '}
            <Link href="/mapping" className={link}>AI Act mapping</Link> groups the results by those evaluation areas and
            maps them to the EU AI Act.
          </p>
        </header>
        <IndexStrip stats={stats} />
      </section>

      <section id="health" aria-labelledby="health-heading" className="scroll-mt-24 space-y-6">
        <BenchmarkHealthGrid health={getBenchmarkHealth()} analysis={getSampleAnalysis()} scores={healthScores} catalog={catalog} meta={getBenchmarkMeta()} />
      </section>
    </IndexScope>
    </div>
  );
}
