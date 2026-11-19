'use client';
import { ChevronDown } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Segmented } from '@/components/home/explorerUi';

export type WeightsFilter = 'all' | 'open' | 'proprietary';
const WEIGHTS: { id: WeightsFilter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'open', label: 'Open' },
  { id: 'proprietary', label: 'Proprietary' },
];

// Search, sort, open/proprietary filter and match count, shared by the models grid and the rankings table.
export function ModelFilterBar({ search, onSearch, sorts, sortId, onSort, weights, onWeights, shown, total }: {
  search: string;
  onSearch: (v: string) => void;
  sorts: { id: string; label: string }[];
  sortId: string;
  onSort: (id: string) => void;
  weights: WeightsFilter;
  onWeights: (w: WeightsFilter) => void;
  shown: number;
  total: number;
}) {
  return (
    <div className="flex flex-col gap-3 md:flex-row md:items-center">
      <Input
        type="search"
        placeholder="Filter by model or organization…"
        value={search}
        onChange={e => onSearch(e.target.value)}
        className="h-10 w-full rounded-full bg-card px-4 text-sm md:w-72"
      />
      <label className="relative w-full md:w-72">
        <span className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-sm text-muted-foreground">Sort by</span>
        <select
          value={sortId}
          onChange={e => onSort(e.target.value)}
          className="h-10 w-full appearance-none rounded-full border border-input bg-card pl-[4.5rem] pr-10 text-sm text-foreground outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
        >
          {sorts.map(s => <option key={s.id} value={s.id}>{s.label}</option>)}
        </select>
        <ChevronDown className="pointer-events-none absolute right-4 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
      </label>
      <Segmented label="Weights" value={weights} options={WEIGHTS} onChange={onWeights} />
      <p className="text-sm tabular-nums text-muted-foreground md:ml-auto">
        {shown === total ? `${total} models` : `${shown} of ${total} models`}
      </p>
    </div>
  );
}
