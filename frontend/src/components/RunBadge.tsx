import type { RunType } from '@/lib/data';
import { RUN_TYPES } from '@/lib/scores';

export function RunBadge({ run, short = false }: { run: RunType | null; short?: boolean }) {
  if (!run) return <span className="text-muted-foreground/60 text-xs">—</span>;
  const meta = RUN_TYPES[run];
  return (
    <span
      title={meta.description}
      className={`inline-flex items-center whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-semibold leading-4 ${meta.className}`}
    >
      {short ? meta.short : meta.label}
    </span>
  );
}
