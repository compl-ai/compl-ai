'use client';
import { useState } from 'react';
import Link from 'next/link';
import type { CostRow, RunType } from '@/lib/data';
import { formatMultiplier, formatTokens, rateSource } from '@/lib/scores';
import { fmtUsd } from '@/components/home/explorer';
import { Segmented } from '@/components/home/explorerUi';

type Unit = 'usd' | 'tokens';

export function CostPanel({ cost, run, coreAnswered, coreTotal }: {
  cost: CostRow;
  run: RunType;
  coreAnswered: number | null;
  coreTotal: number | null;
}) {
  const [picked, setPicked] = useState<Unit>('usd');
  const priced = !!(cost.core_usd && cost.full_usd && cost.rate);
  const unit: Unit = priced ? picked : 'tokens';
  const core = (unit === 'usd' ? cost.core_usd : cost.core_tokens) ?? 0;
  const full = (unit === 'usd' ? cost.full_usd : cost.full_tokens) ?? 0;
  const multiplier = core > 0 && full > 0 ? full / core : null;
  const coreEstimated = run !== 'core';
  const fullEstimated = run !== 'full';
  const fmt = (v: number, estimated: boolean) => `${estimated ? '~' : ''}${unit === 'usd' ? fmtUsd(v) : `${formatTokens(v)} tokens`}`;
  const bars = [
    { label: 'CORE subset', value: core, estimated: coreEstimated, className: 'bg-teal-strong' },
    { label: 'Every question', value: full, estimated: fullEstimated, className: 'bg-ink/25' },
  ];

  return (
    <div className="rounded-2xl border border-border bg-card p-5 md:p-6 grid grid-cols-1 md:grid-cols-[minmax(0,15rem)_minmax(0,1fr)] gap-6 md:gap-10 items-start">
      <div>
        {priced && (
          <div className="mb-4">
            <Segmented label="Measure" value={unit} onChange={setPicked} options={[{ id: 'usd', label: 'Cost' }, { id: 'tokens', label: 'Tokens' }]} />
          </div>
        )}
        <div className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">{unit === 'usd' ? 'Eval Cost (CORE)' : 'Eval Tokens (CORE)'}</div>
        <div className="mt-1 font-serif text-5xl font-extrabold tracking-tight tabular text-teal-strong">
          {coreEstimated ? '~' : ''}{unit === 'usd' ? fmtUsd(core) : formatTokens(core)}
        </div>
        {multiplier !== null && (
          <div className="mt-1 text-sm text-ink">
            <span className="font-semibold tabular">{formatMultiplier(multiplier)}×</span> {unit === 'usd' ? 'cheaper than a full eval' : 'fewer tokens than a full eval'}
          </div>
        )}
      </div>
      <div className="space-y-3">
        {bars.map(b => (
          <div key={b.label}>
            <div className="flex items-baseline justify-between gap-3 text-sm">
              <span className="text-muted-foreground">{b.label}</span>
              <span className="tabular">
                <span className="font-semibold text-ink">{fmt(b.value, b.estimated)}</span>
                <span className="ml-1.5 text-xs text-muted-foreground">{b.estimated ? 'estimated' : 'measured'}</span>
              </span>
            </div>
            <div className="mt-1.5 h-2">
              <div className={`h-2 rounded-full ${b.className}`} style={{ width: `${Math.max((b.value / full) * 100, 1)}%` }} />
            </div>
          </div>
        ))}
        <p className="pt-1 text-sm text-muted-foreground leading-relaxed">
          {coreAnswered !== null && coreTotal !== null && <>{coreAnswered.toLocaleString()} of {coreTotal.toLocaleString()} core questions answered. </>}
          Tokens of the evaluated model only, measured in its logs; judge models are not counted.
          {priced ? <> Costs are these tokens at {rateSource(cost.rate!.source)} list prices.</> : <> No list price is available for this model, so only tokens are shown.</>}{' '}
          <Link href="/methodology#efficiency" className="font-semibold text-link hover:underline">How we estimate this</Link>
        </p>
      </div>
    </div>
  );
}
