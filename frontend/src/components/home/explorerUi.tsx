'use client';
import type { CostOverview, ExperimentsData, JoinedModelData } from '@/lib/data';
import type { IndexId } from '@/lib/scores';
import type { ModelLookup } from './explorer';

export const INDEX_COLOR: Record<IndexId, string> = {
  capability: 'var(--color-brand)',
  reliability: 'var(--color-teal-strong)',
  safety: 'var(--color-lavender-strong)',
};

export const INDEX_SOFT: Record<IndexId, string> = {
  capability: 'var(--color-brand-soft)',
  reliability: 'var(--color-teal-soft)',
  safety: 'var(--color-lavender-soft)',
};

export type ViewProps = { e: ExperimentsData; costs: CostOverview; models: JoinedModelData[]; byId: ModelLookup; inset: number };

export function ViewFrame({ controls, summary, status, legend, plotClassName, children }: {
  controls?: React.ReactNode;
  summary?: React.ReactNode;
  status: React.ReactNode;
  legend?: React.ReactNode;
  plotClassName?: string;
  children: React.ReactNode;
}) {
  return (
    <>
      {/* Clears the floating question card on wide screens. */}
      {(controls || legend) && <div className="mb-1 flex flex-col gap-x-6 gap-y-2 md:flex-row md:items-start xl:pl-[418px]">
        {controls && <div className="flex shrink-0 flex-wrap items-center gap-2">{controls}</div>}
        <div className="flex min-w-0 flex-1 flex-wrap content-start items-center gap-x-5 gap-y-1 text-xs text-muted-foreground md:min-h-[34px] md:justify-end md:pt-[9px]">{legend}</div>
      </div>}
      {summary && <p className="mt-2 mb-3 pl-1 text-xs text-muted-foreground xl:pl-[422px]">{summary}</p>}
      <div className={`relative ${plotClassName ?? 'h-[400px] md:h-[520px] xl:h-auto xl:min-h-[520px] xl:flex-1'}`}>{children}</div>
      <p className="mt-3 text-xs text-muted-foreground leading-snug xl:pl-[418px]">{status}</p>
    </>
  );
}

export function Segmented<T extends string>({ label, value, options, onChange }: {
  label: string;
  value: T;
  options: { id: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex h-[34px] items-center rounded-full border border-border bg-card p-[3px]">
      {options.map(o => (
        <button
          key={o.id}
          type="button"
          role="radio"
          aria-checked={o.id === value}
          onClick={() => onChange(o.id)}
          className={`h-full cursor-pointer rounded-full px-3 text-xs font-semibold transition-colors ${o.id === value ? 'bg-ink text-card' : 'text-ink/70 hover:text-ink'}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function SelectFilter({ label, value, onChange, children }: { label: string; value: string; onChange: (v: string) => void; children: React.ReactNode }) {
  return (
    <label className="flex items-center gap-2">
      <span className="sr-only">{label}</span>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        aria-label={label}
        className="h-[34px] rounded-full border border-border bg-card px-3 text-xs font-semibold text-ink focus:outline-none focus:ring-2 focus:ring-brand/40"
      >
        {children}
      </select>
    </label>
  );
}

export function Switch({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="inline-flex h-[34px] cursor-pointer items-center gap-2 px-1 text-xs font-semibold text-ink"
    >
      <span className={`relative h-[18px] w-8 rounded-full transition-colors ${checked ? 'bg-brand' : 'bg-track'}`}>
        <span className={`absolute top-[2px] h-[14px] w-[14px] rounded-full bg-card shadow transition-[left] ${checked ? 'left-[16px]' : 'left-[2px]'}`} />
      </span>
      {label}
    </button>
  );
}

export const Divider = () => <span className="mx-1 hidden h-5 w-px bg-border sm:block" />;

type SwatchKind = 'dot' | 'hollow' | 'square' | 'line' | 'dash' | 'bar' | 'ci' | 'tick';

function Swatch({ kind, color }: { kind: SwatchKind; color: string }) {
  switch (kind) {
    case 'dot':
      return <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />;
    case 'hollow':
      return <span className="inline-block h-2.5 w-2.5 rounded-full border-[1.5px] bg-card" style={{ borderColor: color }} />;
    case 'square':
      return <span className="inline-block h-2.5 w-2.5 rounded-[2px]" style={{ backgroundColor: color }} />;
    case 'line':
      return <span className="inline-block h-0.5 w-4 rounded" style={{ backgroundColor: color }} />;
    case 'dash':
      return <span className="inline-block w-4 border-t-2 border-dashed" style={{ borderColor: color }} />;
    case 'bar':
      return <span className="inline-block h-2.5 w-4 rounded-[2px]" style={{ backgroundColor: color }} />;
    case 'tick':
      return <span className="inline-block h-3 w-0.5 rounded" style={{ backgroundColor: color }} />;
    case 'ci':
      return (
        <span aria-hidden className="relative inline-block h-2.5 w-4">
          <span className="absolute inset-x-0 top-1/2 h-px bg-muted-foreground/70" />
          <span className="absolute inset-y-0 left-0 w-px bg-muted-foreground/70" />
          <span className="absolute inset-y-0 right-0 w-px bg-muted-foreground/70" />
        </span>
      );
  }
}

export function Key({ kind, color = 'var(--muted-foreground)', children }: { kind: SwatchKind; color?: string; children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <Swatch kind={kind} color={color} />
      {children}
    </span>
  );
}

export const KeySep = () => <span className="hidden sm:block w-px self-stretch bg-border" />;

export function ScaleKey() {
  return (
    <span
      className="inline-flex items-center gap-1.5 cursor-help"
      title="Index scores are relative to the calibration panel: 100 is the panel average and 15 points is one standard deviation. They are not percentages."
    >
      <span aria-hidden className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full border border-muted-foreground/60 text-[9px] font-semibold leading-none">i</span>
      Index scores: 100 = calibration-panel average
    </span>
  );
}

// Filled dots are open-weight models, hollow ones proprietary.
export function WeightKeys({ color }: { color: string }) {
  return (
    <>
      <Key kind="dot" color={color}>Open weights</Key>
      <Key kind="hollow" color={color}>Proprietary</Key>
    </>
  );
}
