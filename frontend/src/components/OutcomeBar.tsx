import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import type { Outcome, OutcomeShares, SampleAnalysis } from '@/lib/data';
import { OUTCOMES, TAG_LABELS } from '@/lib/outcomes';

type Definition = SampleAnalysis['outcomes']['definition'];

export function formatShare(share: number): string {
  const pct = share * 100;
  if (pct > 0 && pct < 1) return `${pct.toFixed(1)}%`;
  return `${Math.round(pct)}%`;
}

function OutcomeDetails({ shares, definition, tags, footer }: { shares: OutcomeShares; definition: Definition; tags?: Record<string, number>; footer?: string }) {
  return (
    <div className="w-60 py-0.5 text-left">
      {OUTCOMES.filter(o => shares[o.id] > 0).map(o => {
        const tagCounts = tags ? definition[o.id].tags.filter(t => tags[t]).map(t => [t, tags[t]] as const) : [];
        const showTags = tagCounts.length > 1 || (tagCounts.length === 1 && o.id === 'did_not_finish');
        return (
          <div key={o.id} className="py-0.5">
            <div className="flex items-center gap-2">
              <span className={`size-2 shrink-0 rounded-sm ${o.bar}`} />
              <span className="flex-1">{definition[o.id].label}</span>
              <span className="tabular font-semibold">{formatShare(shares[o.id])}</span>
            </div>
            {showTags && (
              <div className="ml-4 opacity-75">
                {tagCounts.map(([t, n]) => (
                  <div key={t}>{n.toLocaleString()} {TAG_LABELS[t] ?? t}</div>
                ))}
              </div>
            )}
          </div>
        );
      })}
      {footer && <div className="mt-1.5 border-t border-current/20 pt-1.5 opacity-75">{footer}</div>}
    </div>
  );
}

// Stacked bar of the answer outcomes, with the shares (and raw tag counts, when given) on hover.
export function OutcomeBar({
  shares,
  definition,
  tags,
  footer,
  size = 'sm',
}: {
  shares: OutcomeShares;
  definition: Definition;
  tags?: Record<string, number>;
  footer?: string;
  size?: 'sm' | 'md';
}) {
  const label = OUTCOMES.filter(o => shares[o.id] > 0)
    .map(o => `${definition[o.id].label} ${formatShare(shares[o.id])}`)
    .join(', ');
  return (
    <Tooltip>
      <TooltipTrigger aria-label={label} className="block w-full cursor-default py-1">
        <span className={`flex w-full gap-[2px] overflow-hidden rounded-full ${size === 'md' ? 'h-2.5' : 'h-1.5'}`}>
          {OUTCOMES.map(o => (shares[o.id] > 0 ? <span key={o.id} className={`basis-0 ${o.bar}`} style={{ flexGrow: shares[o.id] }} /> : null))}
        </span>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="text-xs leading-snug">
        <OutcomeDetails shares={shares} definition={definition} tags={tags} footer={footer} />
      </TooltipContent>
    </Tooltip>
  );
}

export function OutcomeLegend({ definition, only }: { definition: Definition; only?: Outcome[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
      {OUTCOMES.filter(o => !only || only.includes(o.id)).map(o => (
        <span key={o.id} className="inline-flex items-center gap-1.5">
          <span className={`size-2 rounded-sm ${o.bar}`} />
          {definition[o.id].label}
        </span>
      ))}
    </div>
  );
}
