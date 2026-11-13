// A small histogram of the other models' scores on a 0–1 scale, with this model's score as a vertical line
// and its interval as a shaded band behind the bars.
export function ScoreDistribution({ field, point, interval, reference, bins = 20, tone, className = '' }: {
  field: number[];
  point: number;
  interval: readonly [number, number] | null;
  // An optional reference score, such as the median model, drawn as a thin tick.
  reference?: number | null;
  bins?: number;
  // Band and line classes; defaults to brand orange.
  tone?: { band: string; dot: string };
  className?: string;
}) {
  const counts = Array<number>(bins).fill(0);
  for (const v of field) counts[Math.min(bins - 1, Math.max(0, Math.floor(v * bins)))]++;
  const peak = Math.max(1, ...counts);
  const pct = (v: number) => Math.min(Math.max(v, 0), 1) * 100;
  return (
    <div className={`relative h-7 ${className}`} aria-hidden>
      {interval && (
        <div
          className={`absolute inset-y-0 opacity-60 ${tone?.band ?? 'bg-brand/45'}`}
          style={{ left: `${pct(interval[0])}%`, width: `${Math.max(pct(interval[1]) - pct(interval[0]), 0.5)}%` }}
        />
      )}
      <div className="absolute inset-0 flex items-end gap-px">
        {counts.map((c, i) => (
          <div key={i} className="flex-1 rounded-t-[1px] bg-ink/20" style={{ height: c ? `${Math.max((c / peak) * 100, 8)}%` : 0 }} />
        ))}
      </div>
      <div className="absolute inset-x-0 bottom-0 h-px bg-ink/25" />
      {reference != null && (
        <div className="absolute -top-0.5 bottom-0 w-px -translate-x-1/2 bg-ink/60" style={{ left: `${pct(reference)}%` }} />
      )}
      <div
        className={`absolute -top-1 bottom-0 w-0.5 -translate-x-1/2 rounded-full ${tone?.dot ?? 'bg-brand'}`}
        style={{ left: `${pct(point)}%` }}
      />
    </div>
  );
}
