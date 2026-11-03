'use client';
import { useState } from 'react';
import Link from 'next/link';
import { Input } from '@/components/ui/input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { JoinedModelData } from '@/lib/data';
import { ArrowDown, ArrowUp, ArrowUpDown } from 'lucide-react';
import { INDICES, NO_SCORE_REASON, THETA_MEAN, formatInterval, formatScore, formatTokens, getIndexTheta, getRunType, parseParams, scaleTheta } from '@/lib/scores';

// Fixed display range for the interval bars, in scaled points.
const BAR_MIN = 40;
const BAR_MAX = 150;
const toPct = (v: number) => Math.min(Math.max((v - BAR_MIN) / (BAR_MAX - BAR_MIN), 0), 1) * 100;


function formatParams(m: JoinedModelData): string {
  const specs = m.metadata?.specs;
  const total = parseParams(specs?.total_params);
  if (!total || isNaN(total)) return '—';
  const active = specs?.active_params ? parseParams(specs.active_params) : null;
  const fmt = (n: number) => `${(n / 1e9).toFixed(0)}B`;
  return active && active !== total && !isNaN(active) ? `${fmt(total)} (${fmt(active)})` : fmt(total);
}

function tokenMultiplier(t: { core: number; full: number }): number | null {
  return t.core > 0 && t.full > 0 ? t.full / t.core : null;
}

function formatMultiplier(m: number): string {
  return m >= 10 ? Math.round(m).toString() : m.toFixed(1);
}

type SortKey = 'name' | 'coverage' | 'params' | 'tokens' | (typeof INDICES)[number]['id'];

function TokensCell({ model }: { model: JoinedModelData }) {
  const run = getRunType(model);
  const t = model.prediction?.usage?.tokens;
  if (!t || !run) {
    return (
      <TableCell className="px-3 py-3 text-right text-muted-foreground/60 text-xs" title="The logs for this run have no token counts.">—</TableCell>
    );
  }
  const coreEst = run !== 'core';
  const fullEst = run !== 'full';
  const corePct = Math.max((t.core / t.full) * 100, 2);
  const multiplier = tokenMultiplier(t);
  const benchmarks = model.prediction?.coverage?.tasks_total;
  const title = [
    `Core subset: ${coreEst ? 'about ' : ''}${t.core.toLocaleString()} tokens${coreEst ? ' (estimated)' : ' (measured)'}`,
    `All items: ${fullEst ? 'about ' : ''}${t.full.toLocaleString()} tokens${fullEst ? ' (estimated)' : ' (measured)'}`,
    `Tokens of the evaluated model (judge models not counted), on the ${t.tasks}${benchmarks ? ` of ${benchmarks}` : ''} benchmarks it ran that have token counts in the logs. Estimates scale the measured tokens per item to the other item count.`,
  ].join('\n');
  return (
    <TableCell className="px-3 py-3" title={title}>
      <div className="flex flex-col items-end gap-1.5">
        <span className="tabular text-[13px] font-semibold text-ink whitespace-nowrap">
          {multiplier !== null ? <>{formatMultiplier(multiplier)}× fewer</> : <>{coreEst ? '~' : ''}{formatTokens(t.core)}</>}
        </span>
        <div className="w-full space-y-0.5" aria-hidden>
          <div className="h-1 rounded-full bg-teal" style={{ width: `${corePct}%` }} />
          <div className="h-1 w-full rounded-full bg-[#D5D8DD]" />
        </div>
        <span className="text-[10px] text-muted-foreground tabular whitespace-nowrap">
          {coreEst ? '~' : ''}{formatTokens(t.core)} vs {fullEst ? '~' : ''}{formatTokens(t.full)} full
        </span>
      </div>
    </TableCell>
  );
}
type SortState = { key: SortKey, direction: 'asc' | 'desc' };

function SortableHead({ label, sortKey, sort, onSort, align = 'left', className = '' }: {
  label: string, sortKey: SortKey, sort: SortState, onSort: (key: SortKey) => void, align?: 'left' | 'right', className?: string
}) {
  const active = sort.key === sortKey;
  const Icon = !active ? ArrowUpDown : sort.direction === 'asc' ? ArrowUp : ArrowDown;
  return (
    <TableHead
      className={`h-11 px-3 text-xs font-semibold uppercase tracking-wider ${align === 'right' ? 'text-right' : ''} ${className}`}
      aria-sort={active ? (sort.direction === 'asc' ? 'ascending' : 'descending') : 'none'}
    >
      <button
        onClick={() => onSort(sortKey)}
        className={`inline-flex items-center gap-1 uppercase tracking-wider transition-colors hover:text-ink ${align === 'right' ? 'flex-row-reverse' : ''} ${active ? 'text-ink' : 'text-muted-foreground'}`}
      >
        {label}
        <Icon className={`w-3 h-3 ${active ? 'text-brand-strong' : 'opacity-40'}`} />
      </button>
    </TableHead>
  );
}

export function ModelDataTable({ initialModels, initialRows, indexTabs = false }: {
  initialModels: JoinedModelData[],
  // Rows shown before "Show all"; searching always shows every match.
  initialRows?: number,
  // Index tabs above the table that pick the ranking dimension.
  indexTabs?: boolean,
}) {
  const [search, setSearch] = useState('');
  const [sort, setSort] = useState<SortState>({ key: 'capability', direction: 'desc' });
  const [expanded, setExpanded] = useState(false);

  const term = search.toLowerCase();
  const filtered = initialModels.filter(m =>
    (m.metadata?.name || m.yamlId).toLowerCase().includes(term) ||
    (m.metadata?.organization || '').toLowerCase().includes(term)
  );

  const sortValue = (m: JoinedModelData): number | string | null => {
    switch (sort.key) {
      case 'name': return m.metadata?.name || m.yamlId;
      case 'coverage': return m.prediction?.coverage?.tasks_completed ?? null;
      case 'params': return parseParams(m.metadata?.specs?.total_params) || null;
      case 'tokens': { const t = m.prediction?.usage?.tokens; return t ? tokenMultiplier(t) : null; }
      default: return getIndexTheta(m, sort.key)?.theta ?? null;
    }
  };

  // Missing values always sort last, whatever the direction.
  const sorted = [...filtered].sort((a, b) => {
    const va = sortValue(a), vb = sortValue(b);
    if (va === null && vb === null) return 0;
    if (va === null) return 1;
    if (vb === null) return -1;
    const cmp = typeof va === 'string' ? va.localeCompare(vb as string) : (va as number) - (vb as number);
    return sort.direction === 'asc' ? cmp : -cmp;
  });

  const limited = initialRows !== undefined && !expanded && !term && sorted.length > initialRows;
  const visible = limited ? sorted.slice(0, initialRows) : sorted;
  const activeIndex = INDICES.find(d => d.id === sort.key);

  const requestSort = (key: SortKey) => {
    setSort(prev => prev.key === key
      ? { key, direction: prev.direction === 'desc' ? 'asc' : 'desc' }
      : { key, direction: key === 'name' ? 'asc' : 'desc' });
  };


  return (
    <div className="space-y-5">
      <div className="flex flex-col sm:flex-row sm:items-center gap-3">
        {indexTabs && (
          <div className="inline-flex bg-muted p-1 rounded-full self-start" role="tablist" aria-label="Rank by index">
            {INDICES.map(d => {
              const isActive = sort.key === d.id;
              return (
                <button
                  key={d.id}
                  role="tab"
                  aria-selected={isActive}
                  onClick={() => setSort({ key: d.id, direction: 'desc' })}
                  className={`px-4 py-1.5 rounded-full text-sm font-semibold transition-all ${
                    isActive ? 'bg-white text-ink shadow-sm' : 'text-muted-foreground hover:text-ink'
                  }`}
                >
                  {d.label}
                </button>
              );
            })}
          </div>
        )}
        <Input
          type="search"
          placeholder="Filter by model or organization…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full sm:w-72 h-10 px-4 rounded-full bg-white text-sm sm:ml-auto"
        />
      </div>
      {indexTabs && activeIndex && <p className="text-sm text-muted-foreground max-w-3xl">{activeIndex.description}</p>}

      <div className="overflow-x-auto">
        <Table className="w-full text-sm">
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <SortableHead sort={sort} onSort={requestSort} label="Model" sortKey="name" className="sticky left-0 bg-white z-10" />
              <TableHead className="h-11 px-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Organization</TableHead>
              <SortableHead sort={sort} onSort={requestSort} label="Token Savings (Eval)" sortKey="tokens" align="right" className="min-w-[130px]" />
              <SortableHead sort={sort} onSort={requestSort} label="Benchmarks" sortKey="coverage" align="right" />
              <SortableHead sort={sort} onSort={requestSort} label="Params" sortKey="params" align="right" />
              {INDICES.map(d => (
                <SortableHead sort={sort} onSort={requestSort} key={d.id} label={d.label} sortKey={d.id} align="right" className="min-w-[150px]" />
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {visible.map(model => {
              const cov = model.prediction?.coverage;
              return (
                <TableRow key={model.id} className="group hover:bg-[#F9FAFB]">
                  <TableCell className="px-3 py-3 font-semibold text-ink truncate max-w-[220px] sticky left-0 bg-white group-hover:bg-[#F9FAFB] transition-colors z-10">
                    <Link href={`/models/${model.yamlId}`} className="hover:underline decoration-brand decoration-2 underline-offset-4">
                      {model.metadata?.name || model.yamlId}
                    </Link>
                  </TableCell>
                  <TableCell className="px-3 py-3 text-muted-foreground truncate max-w-[160px]">
                    {model.metadata?.organization || '—'}
                  </TableCell>
                  <TokensCell model={model} />
                  <TableCell
                    className="px-3 py-3 text-right text-muted-foreground tabular text-xs whitespace-nowrap"
                    title={cov ? `${cov.samples_completed.toLocaleString()} of ${cov.samples_total.toLocaleString()} core items answered` : undefined}
                  >
                    {cov ? `${cov.tasks_completed}/${cov.tasks_total}` : '—'}
                  </TableCell>
                  <TableCell className="px-3 py-3 text-right text-muted-foreground tabular text-xs whitespace-nowrap">
                    {formatParams(model)}
                  </TableCell>
                  {INDICES.map(d => {
                    const t = getIndexTheta(model, d.id);
                    if (!t) {
                      return (
                        <TableCell key={d.id} className="px-3 py-3 text-right text-muted-foreground/60 text-xs" title={NO_SCORE_REASON}>—</TableCell>
                      );
                    }
                    const lo = toPct(scaleTheta(t.interval[0]));
                    const hi = toPct(scaleTheta(t.interval[1]));
                    const point = toPct(scaleTheta(t.theta));
                    return (
                      <TableCell
                        key={d.id}
                        className="px-3 py-3"
                        title={`${d.label} ${formatScore(t.theta)}, 90% interval ${formatInterval(t.interval)}; estimated from ${t.tasks} of ${t.tasks_total} benchmarks`}
                      >
                        <div className="flex flex-col items-end gap-1.5">
                          <span className="tabular text-[13px] font-semibold text-ink whitespace-nowrap">{formatScore(t.theta)}</span>
                          <div className="relative w-full h-1.5 rounded-full bg-[#EDEFF2]">
                            <div className="absolute inset-y-0 w-px bg-[#D5D8DD]" style={{ left: `${toPct(THETA_MEAN)}%` }} />
                            <div
                              className="absolute inset-y-0 rounded-full bg-brand/45"
                              style={{ left: `${lo}%`, width: `${Math.max(hi - lo, 1)}%` }}
                            />
                            <div
                              className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-brand"
                              style={{ left: `${point}%` }}
                            />
                          </div>
                          <span className="text-[10px] text-muted-foreground tabular whitespace-nowrap">{formatInterval(t.interval)} @ 90% CI</span>
                        </div>
                      </TableCell>
                    );
                  })}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
      {limited && (
        <div className="flex justify-center">
          <button
            onClick={() => setExpanded(true)}
            className="rounded-full border border-border bg-white px-5 py-2 text-sm font-semibold text-ink hover:border-ink/30 transition-colors"
          >
            Show all {sorted.length} models
          </button>
        </div>
      )}
      <p className="text-xs text-muted-foreground">
        Scores use a scale where {THETA_MEAN} is the average of the evaluated models and 15 points is one standard deviation. Small numbers show the 90% interval. A dash means too few of that index&apos;s benchmarks were run to estimate a score.
        {' '}Token savings (eval) compares the evaluated model&apos;s tokens (judge models not counted) on a core-subset run with running all items of the same benchmarks: the teal bar is the core run, the grey bar the full run. A tilde marks values extrapolated from the run that was actually done.
      </p>
    </div>
  );
}
