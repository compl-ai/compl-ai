import type { ReactNode } from 'react';
import { cn } from 'cn';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

// Shared look of the small summary cards (model cards, benchmark health cards): an outlined box,
// a serif title, small uppercase labels and stats with a bold value and an optional muted line under it.

export const baseCardClass = 'flex flex-col rounded-xl border border-border/70 bg-card p-3.5';

export function BaseCard({ id, className, children }: { id?: string; className?: string; children: ReactNode }) {
  return <div id={id} className={cn(baseCardClass, className)}>{children}</div>;
}

export function BaseCardTitle({ className, children }: { className?: string; children: ReactNode }) {
  return <h3 className={cn('font-serif text-base font-bold leading-tight tracking-tight text-ink', className)}>{children}</h3>;
}

export function MicroLabel({ className, icon, children }: { className?: string; icon?: ReactNode; children: ReactNode }) {
  return (
    <div className={cn('flex items-center gap-1 text-[10px] font-semibold uppercase tracking-[0.02em] text-muted-foreground', className)}>
      {icon}
      <span className="truncate">{children}</span>
    </div>
  );
}

// `title` is shown in a hover tooltip under the stat.
export function Stat({ label, value, sub, title }: { label: ReactNode; value: ReactNode; sub?: ReactNode; title?: ReactNode }) {
  const body = (
    <>
      <MicroLabel>{label}</MicroLabel>
      <div className="text-[15px] font-semibold tabular text-ink">{value}</div>
      {sub != null && <div className="whitespace-nowrap text-[10px] tabular text-muted-foreground/80">{sub}</div>}
    </>
  );
  if (title == null) return <div className="min-w-0">{body}</div>;
  return (
    <Tooltip>
      <TooltipTrigger render={<div />} className="min-w-0 cursor-default text-left">
        {body}
      </TooltipTrigger>
      <TooltipContent side="bottom" className="block text-xs leading-snug">
        {title}
      </TooltipContent>
    </Tooltip>
  );
}

export function StatGrid({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn('grid grid-cols-3 gap-2', className)}>{children}</div>;
}
