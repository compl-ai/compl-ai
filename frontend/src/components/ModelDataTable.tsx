'use client';
import { useState } from 'react';
import Link from 'next/link';
import { Input } from '@/components/ui/input';
import { ProviderLogo } from '@/components/ProviderLogo';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import type { ExplorerCostRow, JoinedModelData } from '@/lib/data';
import { fmtUsd } from '@/components/home/explorer';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { ArrowDown, ArrowUp, ArrowUpDown, Info } from 'lucide-react';
import { INDEX_BAR, INDICES, NO_SCORE_REASON, THETA_MEAN, formatInterval, formatMultiplier, formatScore, getIndexTheta, getRunType, parseParams, scaleTheta, scoreBarPct } from '@/lib/scores';


function formatParams(m: JoinedModelData): string | null {
  const specs = m.metadata?.specs;
  const total = parseParams(specs?.total_params);
  if (!total || isNaN(total)) return null;
  const active = specs?.active_params ? parseParams(specs.active_params) : null;
  const fmt = (n: number) => `${(n / 1e9).toFixed(0)}B`;
  return active && active !== total && !isNaN(active) ? `${fmt(total)} (${fmt(active)})` : fmt(total);
}

type SortKey = 'name' | 'org' | 'params' | 'cost' | (typeof INDICES)[number]['id'];

function CostCell({ model, cost }: { model: JoinedModelData; cost: ExplorerCostRow | undefined }) {
  const run = getRunType(model);
  if (!cost || !run || !(cost.core_usd > 0) || !(cost.full_usd > 0)) {
    return (
      <TableCell className="px-3 py-3 text-right text-muted-foreground/60 text-xs" title="No list price for this model (self-hosted).">—</TableCell>
    );
  }
  const coreEst = run !== 'core';
  const fullEst = run !== 'full';
  const multiplier = cost.full_usd / cost.core_usd;
  const benchmarks = model.prediction?.coverage?.tasks_total;
  const tokens = model.prediction?.usage?.tokens;
  const title = [
    `CORE subset: ${coreEst ? 'about ' : ''}${fmtUsd(cost.core_usd)}${coreEst ? ' (estimated)' : ' (measured tokens)'}`,
    `All items: ${fullEst ? 'about ' : ''}${fmtUsd(cost.full_usd)}${fullEst ? ' (estimated)' : ' (measured tokens)'}`,
    `${formatMultiplier(multiplier)}× cheaper. Tokens of the evaluated model at ${cost.source} list prices (judge models not counted)${tokens ? `, on the ${tokens.tasks}${benchmarks ? ` of ${benchmarks}` : ''} benchmarks with token counts in the logs` : ''}.`,
  ].join('\n');
  return (
    <TableCell className="px-1.5 sm:px-2 md:px-3 py-3" title={title}>
      <div className="flex flex-col items-end gap-0.5">
        <span className="tabular text-[13px] font-semibold text-ink whitespace-nowrap">{coreEst ? '~' : ''}{fmtUsd(cost.core_usd)}<span className="hidden sm:inline text-[10px] font-medium text-muted-foreground"> (CORE)</span></span>
        <span className="text-[10px] sm:text-[11px] md:text-xs text-muted-foreground tabular whitespace-nowrap">
          vs {fullEst ? '~' : ''}<span className="sm:hidden">{fmtUsdShort(cost.full_usd)}</span><span className="hidden sm:inline">{fmtUsd(cost.full_usd)}</span><span className="hidden md:inline"> full</span>
        </span>
      </div>
    </TableCell>
  );
}
// Phones only: $1,422 -> $1.4k.
const fmtUsdShort = (v: number) => (v >= 1000 ? `$${+(v / 1000).toFixed(1)}k` : fmtUsd(v));

type SortState = { key: SortKey, direction: 'asc' | 'desc' };

const COST_INFO = 'What a CORE evaluation costs at the model\'s list prices, compared with a full evaluation of the same benchmarks. Judge models are not counted, a tilde marks values extrapolated from the run that was actually done, and self-hosted models without a list price show a dash.';

function SortableHead({ label, sortKey, sort, onSort, align = 'left', className = '', info, nowrap = false }: {
  label: string, nowrap?: boolean, sortKey: SortKey, sort: SortState, onSort: (key: SortKey) => void, align?: 'left' | 'right', className?: string, info?: string
}) {
  const active = sort.key === sortKey;
  const Icon = !active ? ArrowUpDown : sort.direction === 'asc' ? ArrowUp : ArrowDown;
  return (
    <TableHead
      className={`h-11 px-1.5 sm:px-2 md:px-3 pb-2.5 md:pb-0 text-[9px] leading-tight sm:text-[11px] sm:leading-normal md:text-xs font-semibold uppercase tracking-normal md:tracking-wider whitespace-normal md:whitespace-nowrap align-bottom md:align-middle ${align === 'right' ? 'text-right' : ''} ${className}`}
      aria-sort={active ? (sort.direction === 'asc' ? 'ascending' : 'descending') : 'none'}
    >
      <div className={`flex items-center ${align === 'right' ? 'justify-end' : ''}`}>
      <button
        onClick={() => onSort(sortKey)}
        className={`inline-flex cursor-pointer items-center gap-1 text-left uppercase transition-colors hover:text-ink ${align === 'right' ? 'flex-row-reverse' : ''} ${active ? 'text-ink' : 'text-muted-foreground'}`}
      >
        <span className={nowrap ? 'whitespace-nowrap' : undefined}>{label}</span>
        <Icon className={`w-3 h-3 shrink-0 ${active ? 'text-brand-strong' : 'hidden sm:block opacity-40'}`} />
      </button>
      {info && (
        <Tooltip>
          <TooltipTrigger
            aria-label={`About ${label}`}
            className="-my-2 -mr-1.5 hidden md:inline-flex py-2 pl-2 pr-1.5 text-muted-foreground/70 transition-colors hover:text-ink"
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

export function ModelDataTable({ initialModels, costs, initialRows, heading }: {
  initialModels: JoinedModelData[],
  // Eval cost rows keyed by model id; models without a list price are absent.
  costs: Record<string, ExplorerCostRow>,
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
      case 'cost': { const c = costs[m.id]; return c && c.core_usd > 0 ? c.core_usd : null; }
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
      : { key, direction: key === 'name' || key === 'org' || key === 'cost' ? 'asc' : 'desc' });
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
              <SortableHead sort={sort} onSort={requestSort} label="Org" sortKey="org" className="hidden md:table-cell" />
              <SortableHead sort={sort} onSort={requestSort} label="Params" sortKey="params" align="right" className="hidden md:table-cell w-px" nowrap />
              <SortableHead sort={sort} onSort={requestSort} label="Eval Cost" sortKey="cost" align="right" className="sm:min-w-[96px] md:min-w-[130px]" info={COST_INFO} nowrap />
              {INDICES.map(d => (
                <SortableHead sort={sort} onSort={requestSort} key={d.id} label={d.label} sortKey={d.id} align="right" className="sm:min-w-[96px] md:min-w-[150px] xl:min-w-[168px] 2xl:min-w-[180px]" info={d.description} />
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {visible.map(model => {
              const params = formatParams(model);
              return (
                <TableRow key={model.id} className="group hover:bg-hover">
                  <TableCell className="px-1.5 sm:px-2 md:px-3 py-3 text-[11px] md:text-sm font-semibold text-ink max-w-[112px] sm:max-w-[132px] md:max-w-[220px] sticky left-0 bg-card group-hover:bg-hover transition-colors z-10">
                    <span className="flex items-start gap-2 md:block md:truncate">
                      <ProviderLogo org={model.metadata?.organization} className="mt-0.5 text-ink md:hidden" />
                      <span className="min-w-0 whitespace-normal md:inline">
                        <Link href={`/models/${model.yamlId}`} className="hover:underline decoration-brand decoration-2 underline-offset-4">
                          {model.metadata?.name || model.yamlId}
                        </Link>
                        {params && <span className="block whitespace-nowrap font-normal text-muted-foreground tabular text-[10px] md:hidden">{params}</span>}
                      </span>
                    </span>
                  </TableCell>
                  <TableCell className="hidden md:table-cell px-3 py-3 text-muted-foreground max-w-[160px]">
                    <span className="flex items-center gap-2">
                      <ProviderLogo org={model.metadata?.organization} className="text-ink" />
                      <span className="truncate">{model.metadata?.organization || '—'}</span>
                    </span>
                  </TableCell>
                  <TableCell className="hidden md:table-cell w-px px-2 py-3 text-right text-muted-foreground tabular text-xs whitespace-nowrap">
                    {params ?? '—'}
                  </TableCell>
                  <CostCell model={model} cost={costs[model.id]} />
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
                        className="px-1.5 sm:px-2 md:px-3 py-3"
                        title={`${d.label} ${formatScore(t.theta)}, 90% interval ${formatInterval(t.interval)}; estimated from ${t.tasks} of ${t.tasks_total} benchmarks`}
                      >
                        <div className="flex flex-col items-end gap-0.5">
                          <span className="tabular text-[13px] font-semibold text-ink whitespace-nowrap">{formatScore(t.theta)}</span>
                          <div className="relative hidden sm:block w-full h-1.5 rounded-full bg-grid">
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
                          <span className="text-[10px] sm:text-[11px] md:text-xs text-muted-foreground tabular whitespace-nowrap">{formatInterval(t.interval)}<span className="hidden md:inline"> @ 90% CI</span></span>
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
