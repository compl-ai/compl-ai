'use client';
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import Link from 'next/link';
import { cn } from 'cn';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { INDEX_TONE, INDICES, type IndexId } from '@/lib/scores';

export type IndexFilter = IndexId | 'all';

// setFilter(f, false) changes the filter without touching the URL, for callers that set their own hash.
const FilterContext = createContext<{ filter: IndexFilter; setFilter: (f: IndexFilter, updateHash?: boolean) => void }>({
  filter: 'all',
  setFilter: () => {},
});

export const useIndexFilter = () => useContext(FilterContext);

// `#safety` (and the older `#areas-safety`) select an index; other hashes, such as area or benchmark anchors, leave it as is.
function indexFromHash(hash: string): IndexId | null {
  const id = hash.replace(/^#(areas-)?/, '');
  return INDICES.some(i => i.id === id) ? (id as IndexId) : null;
}

// Holds the selected index, kept in the URL hash. Descendants with a data-index attribute are hidden
// unless they belong to it (see globals.css).
export function IndexScope({ className, children }: { className?: string; children: ReactNode }) {
  const [filter, setFilterState] = useState<IndexFilter>('all');
  // A hash target that was hidden by the filter; scrolled to once the new filter has rendered.
  const pendingTarget = useRef<HTMLElement | null>(null);

  useEffect(() => {
    pendingTarget.current?.scrollIntoView();
    pendingTarget.current = null;
  }, [filter]);

  useEffect(() => {
    const read = () => {
      const { hash } = window.location;
      const index = indexFromHash(hash);
      if (index) return setFilterState(index);
      if (!hash) return setFilterState('all');
      // An anchor inside another index's group, such as a benchmark card, switches the strip to that index.
      const target = document.getElementById(decodeURIComponent(hash.slice(1)));
      const owner = target?.closest<HTMLElement>('[data-index]')?.dataset.index;
      if (!target || !owner || target.offsetParent) return;
      setFilterState(owner as IndexId);
      pendingTarget.current = target;
    };
    read();
    window.addEventListener('hashchange', read);
    return () => window.removeEventListener('hashchange', read);
  }, []);

  const setFilter = useCallback((f: IndexFilter, updateHash = true) => {
    setFilterState(f);
    if (!updateHash) return;
    const { pathname, search } = window.location;
    window.history.replaceState(null, '', f === 'all' ? `${pathname}${search}` : `#${f}`);
  }, []);

  return (
    <FilterContext.Provider value={{ filter, setFilter }}>
      <div data-index-filter={filter} className={className}>{children}</div>
    </FilterContext.Provider>
  );
}

export interface StripSegment {
  key: string;
  index: IndexId;
  size: number;
  title: ReactNode;
}

export interface StripStat {
  label: string;
  value: string;
  title?: ReactNode;
  // Shown as a full-width stacked bar next to the value, one segment per part.
  segments?: StripSegment[];
}

const ALL_DESCRIPTION =
  'ComplAI CORE scores every model on three independent indices. Pick one to show only its part of the page.';

// Selector for the three indices, with page-specific stats on each. Shared by the Benchmarks and AI Act mapping pages.
export function IndexStrip({ stats }: { stats: Record<IndexFilter, StripStat[]> }) {
  const { filter, setFilter } = useIndexFilter();
  const options: { id: IndexFilter; label: string }[] = [{ id: 'all', label: 'All indices' }, ...INDICES];
  const description = filter === 'all' ? ALL_DESCRIPTION : INDICES.find(i => i.id === filter)!.description;

  return (
    <div className="space-y-3">
      <div role="group" aria-label="Filter by index" className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {options.map(o => {
          const active = o.id === filter;
          const tone = o.id === 'all' ? null : INDEX_TONE[o.id];
          return (
            <button
              key={o.id}
              type="button"
              aria-pressed={active}
              onClick={() => setFilter(o.id)}
              className={cn(
                'group flex cursor-pointer flex-col overflow-hidden rounded-xl border text-left transition-colors',
                active ? 'border-ink/50 bg-card ring-1 ring-ink/15' : 'border-border/70 bg-card hover:border-ink/30',
              )}
            >
              <span className={cn('flex flex-1 flex-col gap-3 p-3.5', active && (tone ? tone.soft : 'bg-muted/50'))}>
                <span className="flex items-center justify-between gap-2">
                  <span className="font-serif text-lg font-bold leading-tight tracking-tight text-ink">{o.label}</span>
                  <span
                    className={cn(
                      'size-3.5 shrink-0 rounded-full border',
                      active ? 'border-ink bg-ink shadow-[inset_0_0_0_2.5px_var(--card)]' : 'border-ink/30',
                    )}
                    aria-hidden
                  />
                </span>
                <span className="grid grid-cols-2 gap-x-3 gap-y-2">
                  {stats[o.id].map(s => <StripStatItem key={s.label} stat={s} />)}
                </span>
              </span>
            </button>
          );
        })}
      </div>
      <p className="max-w-4xl text-sm leading-relaxed text-muted-foreground">
        {description}{' '}
        <Link href="/methodology#how" className="font-semibold text-link hover:underline">How the indices are computed →</Link>
      </p>
    </div>
  );
}

function StripStatItem({ stat }: { stat: StripStat }) {
  if (stat.segments) return <StripBarItem stat={stat} segments={stat.segments} />;
  const body = (
    <>
      <span className="block truncate text-[10px] font-semibold uppercase tracking-[0.02em] text-muted-foreground">{stat.label}</span>
      <span className="block text-[15px] font-semibold tabular text-ink">{stat.value}</span>
    </>
  );
  if (!stat.title) return <span className="min-w-0">{body}</span>;
  return (
    <Tooltip>
      <TooltipTrigger render={<span />} className="min-w-0 cursor-default">
        {body}
      </TooltipTrigger>
      <TooltipContent side="bottom" className="block max-w-64 text-xs leading-snug">
        {stat.title}
      </TooltipContent>
    </Tooltip>
  );
}

function StripBarItem({ stat, segments }: { stat: StripStat; segments: StripSegment[] }) {
  return (
    <span className="col-span-2 min-w-0">
      <span className="block truncate text-[10px] font-semibold uppercase tracking-[0.02em] text-muted-foreground">{stat.label}</span>
      <span className="flex items-center gap-2.5">
        <span className="text-[15px] font-semibold tabular text-ink">{stat.value}</span>
        <span className="flex h-2.5 flex-1 gap-0.5" aria-hidden>
          {segments.map(seg => (
            <Tooltip key={seg.key}>
              <TooltipTrigger
                render={<span />}
                style={{ flexGrow: seg.size }}
                className={cn('min-w-0.5 basis-0 cursor-default rounded-[2px] transition-opacity hover:opacity-70', INDEX_TONE[seg.index].bar)}
              />
              <TooltipContent side="bottom" className="block max-w-64 text-xs leading-snug">
                {seg.title}
              </TooltipContent>
            </Tooltip>
          ))}
        </span>
      </span>
    </span>
  );
}
