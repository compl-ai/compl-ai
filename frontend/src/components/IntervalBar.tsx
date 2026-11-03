// A horizontal interval on a fixed range: a shaded range, a point estimate and an optional reference tick.
export function IntervalBar({ interval, point, min, max, reference, muted = false, className = '' }: {
  interval: readonly [number, number] | null;
  point: number | null;
  min: number;
  max: number;
  reference?: number;
  muted?: boolean;
  className?: string;
}) {
  const toPct = (v: number) => Math.min(Math.max((v - min) / (max - min), 0), 1) * 100;
  return (
    <div className={`relative h-1.5 rounded-full bg-muted ${className}`} aria-hidden>
      {reference !== undefined && (
        <div className="absolute -inset-y-0.5 w-px bg-ink/25" style={{ left: `${toPct(reference)}%` }} />
      )}
      {interval && (
        <div
          className={`absolute inset-y-0 rounded-full ${muted ? 'bg-ink/15' : 'bg-brand/45'}`}
          style={{ left: `${toPct(interval[0])}%`, width: `${Math.max(toPct(interval[1]) - toPct(interval[0]), 1)}%` }}
        />
      )}
      {point !== null && (
        <div
          className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2 h-2 rounded-full ${muted ? 'bg-ink/40' : 'bg-brand'}`}
          style={{ left: `${toPct(point)}%` }}
        />
      )}
    </div>
  );
}
