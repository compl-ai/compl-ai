import Link from 'next/link';
import type { JoinedModelData } from '@/lib/data';
import { INDICES, formatInterval, formatScore, getIndexTheta } from '@/lib/scores';
import type { AreaStanding } from '@/app/models/strengths';

function formatRelease(date: string): string {
  const parsed = new Date(`${date}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return date;
  return parsed.toLocaleDateString('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' });
}

function StandingList({ label, items, tone }: { label: string; items: AreaStanding[]; tone: 'strong' | 'weak' }) {
  return (
    <div className={`rounded-xl px-3 py-2.5 ${tone === 'strong' ? 'bg-teal/15' : 'bg-muted'}`}>
      <div className={`text-[11px] font-semibold uppercase tracking-[0.12em] ${tone === 'strong' ? 'text-teal-strong' : 'text-muted-foreground'}`}>
        {label}
      </div>
      <p className="mt-1 text-[13px] leading-snug text-ink">
        {items.map((s, i) => (
          <span key={s.key} title={`Rank ${s.rank} of ${s.of} models in this evaluation area`}>
            {s.name}
            {i < items.length - 1 && ', '}
          </span>
        ))}
      </p>
    </div>
  );
}

export function ModelCard({ model, strong, weak }: { model: JoinedModelData; strong: AreaStanding[]; weak: AreaStanding[] }) {
  const meta = model.metadata;
  const name = meta?.name || model.id;
  return (
    <Link
      href={`/models/${model.yamlId}`}
      className="group flex flex-col rounded-2xl border border-border bg-white p-4 transition-colors hover:border-brand/60"
    >
      <h2 className="font-serif text-xl font-extrabold tracking-tight text-ink group-hover:text-brand-strong transition-colors">{name}</h2>
      <p className="mt-1 text-[13px] text-muted-foreground">
        {meta?.organization || 'Unknown organization'}
        {meta?.release_date && <> · {formatRelease(meta.release_date)}</>}
      </p>

      <div className="mt-4 grid grid-cols-3 gap-1.5">
        {INDICES.map(index => {
          const theta = getIndexTheta(model, index.id);
          return (
            <div key={index.id} className="rounded-xl border border-border px-2 py-2" title={index.description}>
              <div className="truncate text-[10px] font-semibold uppercase tracking-[0.02em] text-muted-foreground">{index.label}</div>
              {theta ? (
                <>
                  <div className="mt-1 font-serif text-2xl font-extrabold tracking-tight tabular text-ink">{formatScore(theta.theta)}</div>
                  <div className="text-[11px] text-muted-foreground tabular whitespace-nowrap">{formatInterval(theta.interval)}</div>
                </>
              ) : (
                <>
                  <div className="mt-1 font-serif text-2xl font-extrabold tracking-tight text-muted-foreground/50">—</div>
                  <div className="text-[11px] text-muted-foreground">No score</div>
                </>
              )}
            </div>
          );
        })}
      </div>

      {(strong.length > 0 || weak.length > 0) && (
        <div className="mt-3 space-y-2">
          {strong.length > 0 && <StandingList label="Excels at" items={strong} tone="strong" />}
          {weak.length > 0 && <StandingList label="Falls behind on" items={weak} tone="weak" />}
        </div>
      )}
    </Link>
  );
}
