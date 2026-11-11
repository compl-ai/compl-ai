'use client';
import { useState } from 'react';
import Link from 'next/link';
import { Input } from '@/components/ui/input';
import { ProviderLogo } from '@/components/ProviderLogo';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { JoinedModelData } from '@/lib/data';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { ArrowDown, ArrowUp, ArrowUpDown, Info } from 'lucide-react';
import { INDEX_BAR, INDICES, NO_SCORE_REASON, THETA_MEAN, formatInterval, formatMultiplier, formatScore, formatTokens, getIndexTheta, getRunType, parseParams, scaleTheta, scoreBarPct, tokenMultiplier } from '@/lib/scores';


function formatParams(m: JoinedModelData): string {
  const specs = m.metadata?.specs;
  const total = parseParams(specs?.total_params);
  if (!total || isNaN(total)) return '—';
  const active = specs?.active_params ? parseParams(specs.active_params) : null;
  const fmt = (n: number) => `${(n / 1e9).toFixed(0)}B`;
  return active && active !== total && !isNaN(active) ? `${fmt(total)} (${fmt(active)})` : fmt(total);
}

type SortKey = 'name' | 'org' | 'params' | 'tokens' | (typeof INDICES)[number]['id'];

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
          {multiplier !== null ? <>{formatMultiplier(multiplier)}×</> : <>{coreEst ? '~' : ''}{formatTokens(t.core)}</>}
        </span>
        <div className="w-full space-y-0.5" aria-hidden>
          <div className="h-1 rounded-full bg-ink" style={{ width: `${corePct}%` }} />
          <div className="h-1 w-full rounded-full bg-track" />
        </div>
        <span className="text-[10px] text-muted-foreground tabular whitespace-nowrap">
          {coreEst ? '~' : ''}{formatTokens(t.core)} vs {fullEst ? '~' : ''}{formatTokens(t.full)} full
        </span>
      </div>
    </TableCell>
  );
}
type SortState = { key: SortKey, direction: 'asc' | 'desc' };

const EFFICIENCY_INFO = 'How many times fewer tokens the model uses on a core evaluation than on a full evaluation of the same benchmarks. The dark bar is the core run, the grey bar the full run. Judge models are not counted, and a tilde marks values extrapolated from the run that was actually done.';

function SortableHead({ label, sortKey, sort, onSort, align = 'left', className = '', info }: {
  label: string, sortKey: SortKey, sort: SortState, onSort: (key: SortKey) => void, align?: 'left' | 'right', className?: string, info?: string
}) {
  const active = sort.key === sortKey;
  const Icon = !active ? ArrowUpDown : sort.direction === 'asc' ? ArrowUp : ArrowDown;
  return (
    <TableHead
      className={`h-11 px-3 text-xs font-semibold uppercase tracking-wider ${align === 'right' ? 'text-right' : ''} ${className}`}
      aria-sort={active ? (sort.direction === 'asc' ? 'ascending' : 'descending') : 'none'}
    >
      <div className="inline-flex items-center">
      <button
        onClick={() => onSort(sortKey)}
        className={`inline-flex cursor-pointer items-center gap-1 uppercase tracking-wider transition-colors hover:text-ink ${align === 'right' ? 'flex-row-reverse' : ''} ${active ? 'text-ink' : 'text-muted-foreground'}`}
      >
        {label}
        <Icon className={`w-3 h-3 ${active ? 'text-brand-strong' : 'opacity-40'}`} />
      </button>
      {info && (
        <Tooltip>
          <TooltipTrigger
            aria-label={`About ${label}`}
            className="-my-2 -mr-1.5 inline-flex py-2 pl-2 pr-1.5 text-muted-foreground/70 transition-colors hover:text-ink"
          >
            <Info className="w-3.5 h-3.5" />
          </TooltipTrigger>
          <TooltipContent className="normal-case tracking-normal font-normal leading-snug text-[13px]">{info}</TooltipContent>
        </Tooltip>
      )}
      </div>
    </TableHead>
  );
}

export function ModelDataTable({ initialModels, initialRows, heading }: {
  initialModels: JoinedModelData[],
  // Rendered above the filter row.
  heading?: React.ReactNode,
  // Rows shown before "Show all"; searching always shows every match.
  initialRows?: number,
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
      case 'org': return m.metadata?.organization || null;
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

  const requestSort = (key: SortKey) => {
    setSort(prev => prev.key === key
      ? { key, direction: prev.direction === 'desc' ? 'asc' : 'desc' }
      : { key, direction: key === 'name' || key === 'org' ? 'asc' : 'desc' });
  };


  return (
    <div className="space-y-5">
      {heading}
      <div className="flex flex-col gap-3 md:flex-row md:items-center">
        <Input
          type="search"
          placeholder="Filter by model or organization…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="h-10 w-full rounded-full bg-card px-4 text-sm md:w-72"
        />
        <p className="text-sm tabular-nums text-muted-foreground md:ml-auto">
          {filtered.length === initialModels.length ? `${initialModels.length} models` : `${filtered.length} of ${initialModels.length} models`}
        </p>
      </div>

      <div className="overflow-x-auto">
        <Table className="w-full text-sm">
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <SortableHead sort={sort} onSort={requestSort} label="Model" sortKey="name" className="sticky left-0 bg-card z-10" />
              <SortableHead sort={sort} onSort={requestSort} label="Org" sortKey="org" />
              <SortableHead sort={sort} onSort={requestSort} label="Params" sortKey="params" align="right" />
              <SortableHead sort={sort} onSort={requestSort} label="Eval Efficiency" sortKey="tokens" align="right" className="min-w-[130px]" info={EFFICIENCY_INFO} />
              {INDICES.map(d => (
                <SortableHead sort={sort} onSort={requestSort} key={d.id} label={d.label} sortKey={d.id} align="right" className="min-w-[150px]" info={d.description} />
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {visible.map(model => {
              return (
                <TableRow key={model.id} className="group hover:bg-hover">
                  <TableCell className="px-3 py-3 font-semibold text-ink truncate max-w-[220px] sticky left-0 bg-card group-hover:bg-hover transition-colors z-10">
                    <Link href={`/models/${model.yamlId}`} className="hover:underline decoration-brand decoration-2 underline-offset-4">
                      {model.metadata?.name || model.yamlId}
                    </Link>
                  </TableCell>
                  <TableCell className="px-3 py-3 text-muted-foreground max-w-[160px]">
                    <span className="flex items-center gap-2">
                      <ProviderLogo org={model.metadata?.organization} className="text-ink" />
                      <span className="truncate">{model.metadata?.organization || '—'}</span>
                    </span>
                  </TableCell>
                  <TableCell className="px-3 py-3 text-right text-muted-foreground tabular text-xs whitespace-nowrap">
                    {formatParams(model)}
                  </TableCell>
                  <TokensCell model={model} />
                  {INDICES.map(d => {
                    const t = getIndexTheta(model, d.id);
                    if (!t) {
                      return (
                        <TableCell key={d.id} className="px-3 py-3 text-right text-muted-foreground/60 text-xs" title={NO_SCORE_REASON}>—</TableCell>
                      );
                    }
                    const lo = scoreBarPct(scaleTheta(t.interval[0]));
                    const hi = scoreBarPct(scaleTheta(t.interval[1]));
                    const point = scoreBarPct(scaleTheta(t.theta));
                    return (
                      <TableCell
                        key={d.id}
                        className="px-3 py-3"
                        title={`${d.label} ${formatScore(t.theta)}, 90% interval ${formatInterval(t.interval)}; estimated from ${t.tasks} of ${t.tasks_total} benchmarks`}
                      >
                        <div className="flex flex-col items-end gap-1.5">
                          <span className="tabular text-[13px] font-semibold text-ink whitespace-nowrap">{formatScore(t.theta)}</span>
                          <div className="relative w-full h-1.5 rounded-full bg-grid">
                            <div className="absolute inset-y-0 w-px bg-track" style={{ left: `${scoreBarPct(THETA_MEAN)}%` }} />
                            <div
                              className={`absolute inset-y-0 rounded-full ${INDEX_BAR[d.id].band}`}
                              style={{ left: `${lo}%`, width: `${Math.max(hi - lo, 1)}%` }}
                            />
                            <div
                              className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2 h-2 rounded-full ${INDEX_BAR[d.id].dot}`}
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
            className="rounded-full border border-border bg-card px-5 py-2 text-sm font-semibold text-ink hover:border-ink/30 transition-colors"
          >
            Show all {sorted.length} models
          </button>
        </div>
      )}
      <p className="text-xs text-muted-foreground">
        Scores use a scale where {THETA_MEAN} is the average of the evaluated models and 15 points is one standard deviation. Small numbers show the 90% interval. A dash means too few of that index&apos;s benchmarks were run to estimate a score.
      </p>
    </div>
  );
}
