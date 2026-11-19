// Shared data shaping for the Benchmarks and AI Act mapping pages (server side, at build time).
import { getAreaDescriptions, getBenchmarkLabels, getCosts, getGrounding, getPredictions, getTaxonomy, type GroundingLink, type LabelScore } from '@/lib/data';
import { INDICES, type IndexId } from '@/lib/scores';

export const areaSlug = (key: string) => key.replace(':', '-');
export const benchmarkAnchor = (name: string) => `benchmark-${name}`;

export interface CatalogArea {
  key: string;
  slug: string;
  name: string;
  // Full-suite questions of the benchmark labelled with this area.
  items: number;
}

export interface CatalogBenchmark {
  name: string;
  index: string;
  // Evaluation areas its questions are labelled with, most questions first.
  areas: CatalogArea[];
  populationItems: number;
  coreItems: number | null;
  tokensPerItem: number | null;
}

export function getBenchmarkCatalog(): CatalogBenchmark[] {
  const taxonomy = getTaxonomy();
  const efficiency = getCosts().suite.tasks;
  return Object.entries(getBenchmarkLabels()).map(([name, labels]) => {
    const subcategoryIdx = labels.columns.indexOf('subcategory');
    const counts: Record<string, number> = {};
    for (const row of labels.data) {
      const key = row[subcategoryIdx] as string;
      counts[key] = (counts[key] ?? 0) + 1;
    }
    const eff = efficiency[name];
    return {
      name,
      index: labels.index,
      areas: Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .map(([key, items]) => ({ key, slug: areaSlug(key), name: taxonomy.subcategories[key]?.name ?? key, items })),
      populationItems: eff?.population_items ?? labels.data.length,
      coreItems: eff?.subset_items ?? null,
      tokensPerItem: eff?.tokens_per_item_median ?? null,
    };
  }).sort((a, b) => a.index.localeCompare(b.index) || a.name.localeCompare(b.name));
}

// Estimated full-suite tokens ÷ CORE tokens. The exporter estimates both at each benchmark's median
// tokens per question, so for one benchmark this is simply its full-suite ÷ CORE questions.
export function tokenCompression(benchmarks: CatalogBenchmark[]) {
  let full = 0;
  let core = 0;
  for (const b of benchmarks) {
    if (b.coreItems === null || b.tokensPerItem === null) continue;
    full += b.populationItems * b.tokensPerItem;
    core += b.coreItems * b.tokensPerItem;
  }
  return core > 0 ? full / core : null;
}

export const formatRatio = (ratio: number) => `${ratio < 10 ? ratio.toFixed(1) : Math.round(ratio).toLocaleString()}×`;

export interface ExplorerSubcategory {
  key: string;
  slug: string;
  name: string;
  description: string;
  index: string;
  fullItems: number;
  coreItems: number;
  gap: string | null;
  // Ordered by number of contributing questions, largest first.
  benchmarks: string[];
  lineage: string[];
  links: GroundingLink[];
}

function subcategoryEntries() {
  const taxonomy = getTaxonomy();
  return INDICES.flatMap(index => Object.entries(taxonomy.subcategories).filter(([, s]) => s.index === index.id));
}

export function getExplorerSubcategories(catalog: CatalogBenchmark[]): ExplorerSubcategory[] {
  const grounding = getGrounding();
  const areaDescriptions = getAreaDescriptions();
  // Scored item counts, as on the model pages. Labels can list more rows than are scored
  // (e.g. several prompt variants of one llm_rules item), so they are not counted here.
  const firstModel = Object.values(getPredictions().models ?? {})[0] as { subcategories?: Record<string, LabelScore> } | undefined;
  return subcategoryEntries().map(([key, s]) => ({
    key,
    slug: areaSlug(key),
    name: s.name,
    description: areaDescriptions[key] ?? s.description,
    index: s.index,
    fullItems: firstModel?.subcategories?.[key]?.population_items ?? 0,
    coreItems: s.subset_count,
    gap: s.gap ?? null,
    benchmarks: catalog
      .map(b => ({ name: b.name, items: b.areas.find(a => a.key === key)?.items ?? 0 }))
      .filter(b => b.items > 0)
      .sort((a, b) => b.items - a.items)
      .map(b => b.name),
    lineage: grounding.subcategories[key]?.lineage ?? [],
    links: grounding.subcategories[key]?.refs ?? [],
  }));
}

// References cited for an index: those its areas test, those cited for the whole index and those cited for all indices.
export function getIndexReferences(index: IndexId) {
  const grounding = getGrounding();
  const ids = new Set<string>([...grounding.all, ...(grounding.indices[index] ?? [])]);
  for (const [key, s] of subcategoryEntries()) {
    if (s.index !== index) continue;
    for (const link of grounding.subcategories[key]?.refs ?? []) if (link.coverage !== 'none') ids.add(link.id);
  }
  return ids;
}
