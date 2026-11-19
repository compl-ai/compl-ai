// A horizontal interval on a fixed range (or a custom position function): a shaded range, a point estimate and an optional reference tick.
export function IntervalBar({ interval, point, min = 0, max = 1, position, reference, muted = false, tone, className = '' }: {
  interval: readonly [number, number] | null;
  point: number | null;
  min?: number;
  max?: number;
  position?: (v: number) => number;
  reference?: number;
  muted?: boolean;
  // Band and point classes; defaults to brand orange.
  tone?: { band: string; dot: string };
  className?: string;
}) {
  const toPct = position ?? ((v: number) => Math.min(Math.max((v - min) / (max - min), 0), 1) * 100);
  return (
    <div className={`relative h-1.5 rounded-full bg-muted ${className}`} aria-hidden>
      {reference !== undefined && (
        <div className="absolute -inset-y-0.5 w-px bg-ink/25" style={{ left: `${toPct(reference)}%` }} />
      )}
      {interval && (
        <div
          className={`absolute inset-y-0 rounded-full ${muted ? 'bg-ink/15' : tone?.band ?? 'bg-brand/45'}`}
          style={{ left: `${toPct(interval[0])}%`, width: `${Math.max(toPct(interval[1]) - toPct(interval[0]), 1)}%` }}
        />
      )}
      {point !== null && (
        <div
          className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2 h-2 rounded-full ${muted ? 'bg-ink/40' : tone?.dot ?? 'bg-brand'}`}
          style={{ left: `${toPct(point)}%` }}
        />
      )}
    </div>
  );
}
