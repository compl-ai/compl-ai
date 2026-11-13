import type { Metadata } from 'next';
import { pageMetadata } from '@/lib/site';
import Link from 'next/link';
import { getBenchmarkMeta, getGrounding, type GroundingRef } from '@/lib/data';
import { INDICES, type IndexId } from '@/lib/scores';
import { Eyebrow } from '@/components/layout/Eyebrow';
import { IndexScope, IndexStrip, type IndexFilter, type StripStat } from '@/components/IndexStrip';
import { AiActMap, type MapArea, type MapLink, type WholeEvaluationRef } from '@/components/AiActMap';
import { benchmarkAnchor, getBenchmarkCatalog, getExplorerSubcategories, getIndexReferences } from '@/lib/catalog';

export const metadata: Metadata = pageMetadata('AI Act mapping', 'How ComplAI CORE’s benchmarks map to the technical requirements of the EU AI Act.');

export default function MappingPage() {
  const grounding = getGrounding();
  const meta = getBenchmarkMeta();
  const subcategories = getExplorerSubcategories(getBenchmarkCatalog());

  const areas: MapArea[] = subcategories.map(s => ({
    key: s.key,
    slug: s.slug,
    name: s.name,
    index: s.index as IndexId,
    description: s.description,
    fullItems: s.fullItems,
    coreItems: s.coreItems,
    gap: s.gap,
    benchmarks: s.benchmarks.map(b => ({ label: meta[b]?.name ?? b, href: `/benchmarks#${benchmarkAnchor(b)}` })),
  }));
  // Links CORE doesn't test yet (coverage "none") are left out of the diagram and listed as not tested.
  const links: MapLink[] = subcategories.flatMap(s =>
    s.links.filter(l => l.coverage !== 'none').map(l => ({ area: s.key, ref: l.id, direct: l.strength === 'direct' })),
  );
  const linked = new Set(links.map(l => l.ref));
  const untestedLinks = subcategories.flatMap(s => s.links.filter(l => l.coverage === 'none').map(l => l.id));
  const untested = [...new Set([...grounding.gaps, ...untestedLinks.filter(id => !linked.has(id))])];
  const scopes: Record<string, WholeEvaluationRef['scopes']> = {};
  for (const id of grounding.all) (scopes[id] ??= []).push('all');
  for (const index of INDICES) for (const id of grounding.indices[index.id] ?? []) (scopes[id] ??= []).push(index.id);
  const wholeEvaluation = Object.entries(scopes).map(([id, s]) => ({ id, scopes: s }));
  const used = new Set([...linked, ...untested, ...Object.keys(scopes)]);
  const refs = Object.fromEntries(Object.entries(grounding.refs).filter(([id]) => used.has(id))) as Record<string, GroundingRef>;

  const stripStats = (index: IndexFilter, refCount: number): StripStat[] => {
    const shown = subcategories.filter(s => index === 'all' || s.index === index);
    const scored = shown.filter(s => !s.gap);
    return [
      {
        label: 'Areas',
        value: shown.length.toLocaleString(),
        // Sized by full-suite questions: the population every CORE result estimates.
        segments: shown.map(s => ({
          key: s.key,
          index: s.index as IndexId,
          size: s.fullItems,
          title: (
            <>
              <span className="font-semibold">{s.name}</span>: {s.fullItems.toLocaleString()} questions in the full suite,{' '}
              {s.coreItems.toLocaleString()} in CORE{s.gap ? '. Too few in CORE for a separate score.' : '.'}
            </>
          ),
        })),
      },
      { label: 'Scored', value: `${scored.length} of ${shown.length}`, title: 'Evaluation areas with enough CORE questions for a separate score.' },
      { label: 'Provisions', value: refCount.toLocaleString(), title: 'EU AI Act provisions and Code of Practice measures linked to these results.' },
    ];
  };
  const indexRefs = Object.fromEntries(INDICES.map(index => [index.id, getIndexReferences(index.id)]));
  const allRefs = new Set(Object.values(indexRefs).flatMap(ids => [...ids]));
  const stats = {
    all: stripStats('all', allRefs.size),
    ...Object.fromEntries(INDICES.map(index => [index.id, stripStats(index.id, indexRefs[index.id].size)])),
  } as Record<IndexFilter, StripStat[]>;

  return (
    <div className="frame">
      <div className="cols-10 pt-10 sm:pt-22 pb-12">
        <IndexScope className="space-y-12">
          <section aria-labelledby="mapping-heading" className="space-y-8">
            <header className="max-w-5xl">
              <Eyebrow>AI Act mapping</Eyebrow>
              <h1 id="mapping-heading" className="text-4xl md:text-5xl font-extrabold tracking-tight text-ink">
                How ComplAI CORE maps to the EU AI Act
              </h1>
              <p className="mt-4 text-base text-muted-foreground leading-relaxed">
                In every index, we label all samples with the evaluation area they test, from multi-step reasoning and
                factual accuracy to fairness and resistance to prompt injection (see{' '}
                <Link href="/methodology#labeling" className="font-semibold text-link hover:underline">How it works</Link>). Each evaluation area maps to the provisions
                of the EU AI Act and the General-Purpose AI Code of Practice that it informs, based on the technical
                requirements of COMPL-AI
                <sup className="text-[0.7em] font-semibold">
                  [<a href="#ref-complai" className="text-link hover:underline">1</a>]
                </sup>
                . One evaluation area can inform several provisions, and one provision can draw on several evaluation areas.
                We don’t use evaluation areas to compute the index scores; that is what the{' '}
                <Link href="/benchmarks" className="font-semibold text-link hover:underline">benchmarks</Link> are for.
                Instead, they show which parts of the regulation ComplAI CORE addresses.
              </p>
            </header>
            <IndexStrip stats={stats} />
          </section>

          <section id="mapping" aria-label="Mapping of evaluation areas to provisions" className="scroll-mt-24 space-y-10">
            <AiActMap areas={areas} refs={refs} links={links} wholeEvaluation={wholeEvaluation} untested={untested} />
            <div className="max-w-5xl space-y-3 text-xs leading-relaxed text-muted-foreground">
              <p>
                A benchmark can test a behaviour a provision names without covering the whole obligation, so this mapping is
                evidence, not a compliance check. It is a technical interpretation, not legal advice. Articles 10, 13 and 15
                apply to high-risk AI systems; we cite them for the properties they name. Articles 53 and 55 and the Code of
                Practice apply to general-purpose AI models. Following the Code of Practice is not conclusive evidence of
                compliance with the AI Act. Excerpts are quoted from{' '}
                <a href={grounding._meta.sources['EU AI Act']} target="_blank" rel="noopener noreferrer" className="underline hover:text-ink">artificialintelligenceact.eu</a>{' '}
                and{' '}
                <a href={grounding._meta.sources['GPAI Code of Practice']} target="_blank" rel="noopener noreferrer" className="underline hover:text-ink">code-of-practice.ai</a>.
              </p>
              <p id="ref-complai" className="scroll-mt-24">
                [1] {grounding._meta.paper.title}. {grounding._meta.paper.year}.{' '}
                <a href={grounding._meta.paper.url} target="_blank" rel="noopener noreferrer" className="underline hover:text-ink">
                  {grounding._meta.paper.url.replace('https://', '')}
                </a>
              </p>
            </div>
          </section>
        </IndexScope>
      </div>
    </div>
  );
}
