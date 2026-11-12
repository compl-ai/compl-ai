'use client';
import { useMemo, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Segmented } from '@/components/home/explorerUi';
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

type WeightsFilter = 'all' | 'open' | 'proprietary';
const WEIGHTS: { id: WeightsFilter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'open', label: 'Open' },
  { id: 'proprietary', label: 'Proprietary' },
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
      <div className="flex flex-col gap-3 md:flex-row md:items-center">
        <Input
          type="search"
          placeholder="Filter by model or organization…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="h-10 w-full rounded-full bg-card px-4 text-sm md:w-72"
        />
        <label className="relative w-full md:w-72">
          <span className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-sm text-muted-foreground">Sort by</span>
          <select
            value={sortId}
            onChange={e => setSortId(e.target.value)}
            className="h-10 w-full appearance-none rounded-full border border-input bg-card pl-[4.5rem] pr-10 text-sm text-foreground outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
          >
            {SORTS.map(s => <option key={s.id} value={s.id}>{s.label}</option>)}
          </select>
          <ChevronDown className="pointer-events-none absolute right-4 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
        </label>
        <Segmented label="Weights" value={weights} options={WEIGHTS} onChange={setWeights} />
        <p className="text-sm tabular-nums text-muted-foreground md:ml-auto">
          {sorted.length === items.length ? `${items.length} models` : `${sorted.length} of ${items.length} models`}
        </p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {sorted.map(item => <div key={item.id} className="flex">{item.card}</div>)}
      </div>
      {sorted.length === 0 && <p className="text-sm text-muted-foreground">No models match these filters.</p>}
    </div>
  );
}
