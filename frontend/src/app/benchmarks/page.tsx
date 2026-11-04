import { BenchmarkList, type BenchmarkRow } from '@/components/BenchmarkList';
import { MeasurementExplorer, type ExplorerSubcategory } from '@/components/MeasurementExplorer';
import { CoverageOverview, type CoverageColumn, type CoverageItem } from '@/components/CoverageOverview';
import { getAreaDescriptions, getBenchmarkLabels, getEfficiency, getGrounding, getPredictions, getTaxonomy, type LabelScore } from '@/lib/data';
import { INDEX_TONE, INDICES } from '@/lib/scores';
import { Eyebrow } from '@/components/layout/Eyebrow';

export default function BenchmarksPage() {
  const taxonomy = getTaxonomy();
  const grounding = getGrounding();
  const areaDescriptions = getAreaDescriptions();
  const benchmarks = getBenchmarkLabels();
  const efficiencyTasks = getEfficiency()?.tasks ?? {};
  // Scored item counts, as on the model pages. Labels can list more rows than are scored
  // (e.g. several prompt variants of one llm_rules item), so they are not counted here.
  const firstModel = Object.values(getPredictions().models ?? {})[0] as { subcategories?: Record<string, LabelScore> } | undefined;
  const subcategoryItems = Object.fromEntries(
    Object.entries(firstModel?.subcategories ?? {}).map(([key, score]) => [key, score.population_items]),
  );
  const slug = (key: string) => key.replace(':', '-');
  const benchmarksBySubcategory: Record<string, Record<string, number>> = {};
  const rows: BenchmarkRow[] = Object.entries(benchmarks).map(([name, labels]) => {
    const subcategoryIdx = labels.columns.indexOf('subcategory');
    const tagsIdx = labels.columns.indexOf('tags');
    const subcategories: Record<string, number> = {};
    // Per-benchmark labelled task counts, collected for the evaluation-area explorer.
    const modalities = new Set<string>();
    for (const row of labels.data) {
      const subcategory = row[subcategoryIdx] as string;
      subcategories[subcategory] = (subcategories[subcategory] ?? 0) + 1;
      (benchmarksBySubcategory[subcategory] ??= {})[name] = (benchmarksBySubcategory[subcategory][name] ?? 0) + 1;
      for (const tag of (row[tagsIdx] as string[]) ?? []) {
        if (tag.startsWith('modality:') || tag.startsWith('agent:')) modalities.add(tag);
      }
    }
    const efficiency = efficiencyTasks[name];
    return {
      name,
      index: labels.index,
      areas: Object.entries(subcategories)
        .sort((a, b) => b[1] - a[1])
        .map(([key]) => ({ key, slug: slug(key), name: taxonomy.subcategories[key]?.name ?? key })),
      modalities: [...modalities].sort(),
      populationItems: efficiency?.population_items ?? labels.data.length,
      coreItems: efficiency?.subset_items ?? null,
      tokensPerItem: efficiency?.tokens_per_item_median ?? null,
    };
  }).sort((a, b) => a.index.localeCompare(b.index) || a.name.localeCompare(b.name));

  const totalItems = rows.reduce((sum, r) => sum + r.populationItems, 0);
  const subcategoryEntries = INDICES.flatMap(index =>
    Object.entries(taxonomy.subcategories).filter(([, s]) => s.index === index.id),
  );
  const explorerSubcategories: ExplorerSubcategory[] = subcategoryEntries.map(([key, s]) => ({
    key,
    slug: slug(key),
    name: s.name,
    description: areaDescriptions[key] ?? s.description,
    index: s.index,
    fullItems: subcategoryItems[key] ?? 0,
    coreItems: s.subset_count,
    gap: s.gap ?? null,
    benchmarks: Object.entries(benchmarksBySubcategory[key] ?? {}).sort((a, b) => b[1] - a[1]).map(([name]) => name),
    lineage: grounding.subcategories[key]?.lineage ?? [],
    links: grounding.subcategories[key]?.refs ?? [],
  }));

  // Evidence for each mapped reference. Direct: a scored area tests what it names, or it is an evaluation duty cited for
  // all indices or a whole index. Related: only related links, or direct links from areas too small to score.
  // Not evaluated: no CORE benchmark tests it.
  const usage: Record<string, { label: string; href: string; direct: boolean; scored: boolean; none: boolean }[]> = {};
  for (const [key, s] of subcategoryEntries) {
    for (const link of grounding.subcategories[key]?.refs ?? []) {
      (usage[link.id] ??= []).push({
        label: s.name,
        href: `#${slug(key)}`,
        direct: link.strength === 'direct',
        scored: !s.gap,
        none: link.coverage === 'none',
      });
    }
  }
  const indexUsage: Record<string, string[]> = {};
  for (const index of INDICES) {
    for (const id of grounding.indices[index.id] ?? []) (indexUsage[id] ??= []).push(`${index.label} index`);
  }
  const columns: Record<CoverageColumn['tone'], CoverageItem[]> = { supported: [], partial: [], none: [] };
  for (const [id, item] of Object.entries(grounding.refs)) {
    const uses = usage[id] ?? [];
    const usages = [
      ...(grounding.all.includes(id) ? [{ label: 'All indices' }] : []),
      ...(indexUsage[id] ?? []).map(label => ({ label })),
      ...uses.filter(u => !u.none).map(({ label, href }) => ({ label, href })),
    ];
    if (grounding.gaps.includes(id) || (uses.length > 0 && uses.every(u => u.none))) {
      columns.none.push({ id, item, usages: [] });
    } else if (grounding.all.includes(id) || (indexUsage[id] && uses.length === 0) || uses.some(u => u.direct && u.scored && !u.none)) {
      columns.supported.push({ id, item, usages });
    } else if (uses.length > 0) {
      columns.partial.push({ id, item, usages });
    }
  }
  const byFramework = (a: CoverageItem, b: CoverageItem) => a.item.framework.localeCompare(b.item.framework);
  const link = 'font-semibold text-link hover:underline';

  return (
    <div className="frame">
    <div className="cols-10 pt-22 pb-12 space-y-16">
      <section aria-labelledby="indices-heading" className="space-y-8">
        <header className="max-w-3xl">
          <Eyebrow>Benchmarks</Eyebrow>
          <h1 id="indices-heading" className="text-4xl md:text-5xl font-extrabold tracking-tight text-ink">What CORE measures</h1>
          <p className="mt-4 text-base text-muted-foreground leading-relaxed">
            ComplAI CORE scores every model on three independent indices: Capability, Reliability and Safety. Each index is
            built from established benchmarks, {rows.length} in total with {totalItems.toLocaleString()} tasks. A new model
            runs {taxonomy.total_subset_items.toLocaleString()} of these tasks, and CORE predicts how it would do on the rest.
            The measurements build on the{' '}
            <a href={grounding._meta.paper.url} target="_blank" rel="noopener noreferrer" className={link}>
              2024 COMPL-AI framework
            </a>
            , our technical interpretation of the EU AI Act.
          </p>
        </header>

        <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
          {INDICES.map(index => {
            const indexRows = rows.filter(r => r.index === index.id);
            const tone = INDEX_TONE[index.id];
            const stats: [string, string, boolean][] = [
              ['Benchmarks', indexRows.length.toLocaleString(), false],
              ['Full-suite tasks', indexRows.reduce((sum, r) => sum + r.populationItems, 0).toLocaleString(), false],
              ['CORE tasks', taxonomy.indices[index.id]?.subset_count.toLocaleString() ?? '—', true],
            ];
            return (
              <div key={index.id} className="flex flex-col overflow-hidden rounded-2xl border border-border bg-card">
                <div className={`h-1.5 ${tone.bar}`} aria-hidden />
                <div className="flex flex-1 flex-col p-6 md:p-7">
                  <h2 className="text-2xl md:text-3xl font-extrabold tracking-tight text-ink">{index.label}</h2>
                  <p className="mt-2 text-[15px] leading-relaxed text-ink/75">{index.description}</p>
                  <dl className="mt-auto pt-6 text-sm">
                    {stats.map(([label, value, core]) => (
                      <div
                        key={label}
                        className={`flex items-baseline justify-between gap-3 px-3 py-2 ${core ? `rounded-lg ${tone.soft}` : 'border-b border-border'}`}
                      >
                        <dt className={core ? 'font-semibold text-ink' : 'text-muted-foreground'}>{label}</dt>
                        <dd className={`tabular font-bold text-ink ${core ? 'text-xl' : 'text-lg'}`}>{value}</dd>
                      </div>
                    ))}
                  </dl>
                  <a href={`#areas-${index.id}`} className="mt-4 text-sm font-semibold text-ink hover:underline">
                    Explore its evaluation areas ↓
                  </a>
                </div>
              </div>
            );
          })}
        </div>
        <p className="max-w-3xl text-sm leading-relaxed text-muted-foreground">
          <span className="font-semibold text-ink">Full-suite tasks</span> are every task in the benchmarks, the pool CORE is
          calibrated on. <span className="font-semibold text-ink">CORE tasks</span> are the subset a new model actually runs;
          its results on the full suite are predicted from them.
        </p>
      </section>

      <section id="grounding" className="scroll-mt-24 space-y-6">
        <div className="max-w-3xl">
          <Eyebrow>Evaluation areas</Eyebrow>
          <h2 className="text-3xl font-extrabold tracking-tight text-ink">Explore evaluation areas</h2>
          <p className="mt-3 text-base text-muted-foreground leading-relaxed">
            Every task is also labelled with the ability or behaviour it tests, from multi-step reasoning and factual
            accuracy to fairness and resistance to prompt injection. These evaluation areas cut across benchmarks and give a
            finer view of a model’s results; they are not used to compute the index scores. Each area traces back to a
            requirement of the original COMPL-AI framework and relates to specific parts of the EU AI Act and the Code of
            Practice.
          </p>
        </div>
        <MeasurementExplorer
          indices={INDICES.map(({ id, label }) => ({ id, label }))}
          subcategories={explorerSubcategories}
          refs={grounding.refs}
          paperUrl={grounding._meta.paper.url}
        />
      </section>

      <section id="coverage" className="scroll-mt-24 space-y-6">
        <div className="max-w-3xl">
          <Eyebrow>Regulatory coverage</Eyebrow>
          <h2 className="text-3xl font-extrabold tracking-tight text-ink">What CORE covers, and what it doesn’t</h2>
          <p className="mt-3 text-base text-muted-foreground leading-relaxed">
            Our mapping lists {Object.keys(grounding.refs).length} references: EU AI Act provisions and General-Purpose AI
            Code of Practice commitments, measures and risk topics. They are grouped by how much evidence CORE’s results
            provide. Evidence is not compliance: CORE can test a behaviour a provision names without assessing the whole
            obligation. We list the gaps on purpose, since several systemic risks in the Code of Practice need agentic or
            domain-specific tests that CORE doesn’t run yet.
          </p>
        </div>
        <CoverageOverview
          columns={[
            {
              tone: 'supported',
              title: 'Direct evaluation evidence',
              description: 'A scored evaluation area tests the behaviour the reference names, or the reference calls for model evaluations of the kind CORE runs.',
              items: columns.supported.sort(byFramework),
            },
            {
              tone: 'partial',
              title: 'Related or partial evidence',
              description: 'CORE tests something related, or the closest evaluation area has too few CORE tasks for a separate score.',
              items: columns.partial.sort(byFramework),
            },
            {
              tone: 'none',
              title: 'Not currently evaluated',
              description: 'In our mapping, but no CORE benchmark tests it yet.',
              items: columns.none.sort(byFramework),
            },
          ]}
          footnote={
            <div className="max-w-3xl space-y-3 text-xs leading-relaxed text-muted-foreground">
              {grounding.requirements_not_in_core.length > 0 && (
                <p>
                  Also not covered: requirements from the 2024 COMPL-AI framework whose benchmarks aren’t part of CORE.{' '}
                  {grounding.requirements_not_in_core.map((r, i) => (
                    <span key={r.name}>
                      {i > 0 && '; '}
                      <span className="font-semibold text-ink/80">{r.name}</span> ({r.benchmarks.join(', ')})
                    </span>
                  ))}
                  .
                </p>
              )}
              <p>
                These references are a technical interpretation, not legal advice or a compliance determination. Articles 10,
                13 and 15 apply to high-risk AI systems; we cite them for the properties they name. Articles 53 and 55 and the
                Code of Practice apply to general-purpose AI models. Following the Code of Practice is not conclusive evidence
                of compliance with the AI Act. Excerpts are quoted from{' '}
                <a href={grounding._meta.sources['EU AI Act']} target="_blank" rel="noopener noreferrer" className="underline hover:text-ink">artificialintelligenceact.eu</a>{' '}
                and{' '}
                <a href={grounding._meta.sources['GPAI Code of Practice']} target="_blank" rel="noopener noreferrer" className="underline hover:text-ink">code-of-practice.ai</a>.
              </p>
            </div>
          }
        />
      </section>

      <section id="catalog" className="scroll-mt-24 space-y-6">
        <div className="max-w-3xl">
          <Eyebrow>Catalog</Eyebrow>
          <h2 className="text-3xl font-extrabold tracking-tight text-ink">All benchmarks</h2>
          <p className="mt-3 text-base text-muted-foreground leading-relaxed">
            The {rows.length} source benchmarks behind CORE. Each belongs to one index, and its tasks are labelled by
            evaluation area, so one benchmark can inform several areas.
          </p>
        </div>
        <BenchmarkList benchmarks={rows} />
      </section>
    </div>
    </div>
  );
}
