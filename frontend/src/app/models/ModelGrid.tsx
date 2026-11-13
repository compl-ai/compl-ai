'use client';
import { useMemo, useState } from 'react';
import { ModelFilterBar, type WeightsFilter } from '@/components/ModelFilterBar';
import type { Weights } from '@/components/home/explorer';

export type GridItem = {
  id: string;
  name: string;
  org: string | null;
  release: string | null;
  params: number | null;
  weights: Weights;
  capability: number | null;
  reliability: number | null;
  safety: number | null;
  card: React.ReactNode;
};

type Key = 'release' | 'name' | 'org' | 'params' | 'capability' | 'reliability' | 'safety';

const SORTS: { id: string; label: string; key: Key; desc: boolean }[] = [
  { id: 'newest', label: 'Newest first', key: 'release', desc: true },
  { id: 'oldest', label: 'Oldest first', key: 'release', desc: false },
  { id: 'org', label: 'Organization', key: 'org', desc: false },
  { id: 'name', label: 'Name', key: 'name', desc: false },
  { id: 'largest', label: 'Largest first', key: 'params', desc: true },
  { id: 'smallest', label: 'Smallest first', key: 'params', desc: false },
  { id: 'capability', label: 'Capability', key: 'capability', desc: true },
  { id: 'reliability', label: 'Reliability', key: 'reliability', desc: true },
  { id: 'safety', label: 'Safety', key: 'safety', desc: true },
];

// Missing values always sort last; ties fall back to newest first, then name.
function compare(a: GridItem, b: GridItem, key: Key, desc: boolean): number {
  const x = a[key], y = b[key];
  if (x == null || y == null) return x == null && y == null ? 0 : x == null ? 1 : -1;
  const c = typeof x === 'string' ? x.localeCompare(y as string) : x - (y as number);
  return desc ? -c : c;
}

export function ModelGrid({ items }: { items: GridItem[] }) {
  const [sortId, setSortId] = useState('newest');
  const [search, setSearch] = useState('');
  const [weights, setWeights] = useState<WeightsFilter>('all');
  const sorted = useMemo(() => {
    const { key, desc } = SORTS.find(s => s.id === sortId)!;
    const term = search.trim().toLowerCase();
    const matches = items.filter(m =>
      (weights === 'all' || m.weights === weights) &&
      (m.name.toLowerCase().includes(term) || (m.org || '').toLowerCase().includes(term)));
    return matches.sort((a, b) =>
      compare(a, b, key, desc) || compare(a, b, 'release', true) || a.name.localeCompare(b.name));
  }, [items, sortId, search, weights]);

  return (
    <div className="cols-10 space-y-4">
      <ModelFilterBar
        search={search} onSearch={setSearch}
        sorts={SORTS} sortId={sortId} onSort={setSortId}
        weights={weights} onWeights={setWeights}
        shown={sorted.length} total={items.length}
      />
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {sorted.map(item => <div key={item.id} className="flex">{item.card}</div>)}
      </div>
      {sorted.length === 0 && <p className="text-sm text-muted-foreground">No models match these filters.</p>}
    </div>
  );
}
