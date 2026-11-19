import Link from 'next/link';
import { ArrowDownRight, ArrowUpRight } from 'lucide-react';
import type { JoinedModelData } from '@/lib/data';
import { INDICES, formatInterval, formatRelease, formatScore, getIndexTheta } from '@/lib/scores';
import type { AreaStanding } from '@/app/models/strengths';
import { paramsOf } from '@/components/home/explorer';
import { ProviderLogo } from '@/components/ProviderLogo';
import { MicroLabel, Stat, StatGrid, baseCardClass } from '@/components/ui/base-card';

function formatSize(n: number): string {
  return n >= 1e12 ? `${+(n / 1e12).toFixed(1)}T` : `${Math.round(n / 1e9)}B`;
}

function WeightsTag({ open }: { open: boolean }) {
  return (
    <span className={`shrink-0 rounded-full px-2 py-px font-medium ${open ? 'bg-teal-soft text-teal-strong' : 'bg-lavender-soft text-lavender-strong'}`}>
      {open ? 'Open' : 'Proprietary'}
    </span>
  );
}

function StandingList({ label, items, tone }: { label: string; items: AreaStanding[]; tone: 'strong' | 'weak' }) {
  const Icon = tone === 'strong' ? ArrowUpRight : ArrowDownRight;
  return (
    <div>
      <MicroLabel icon={<Icon className={`size-3.5 shrink-0 ${tone === 'strong' ? 'text-success' : 'text-brand'}`} strokeWidth={2.5} aria-hidden />}>
        {label}
      </MicroLabel>
      <p className="mt-0.5 sm:mt-1 text-xs leading-snug text-ink">
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

/** Org logo · release · size · Open/Proprietary, shared by the card and the model page. */
export function ModelMeta({
  model,
  className = '',
  logoClassName = '',
  showActive = false,
  longDate = false,
}: {
  model: JoinedModelData;
  className?: string;
  logoClassName?: string;
  /** Spell out active parameters for mixture-of-experts models, e.g. "8B (5B active)". */
  showActive?: boolean;
  /** "Released July 2026" instead of "Jul '26". */
  longDate?: boolean;
}) {
  const meta = model.metadata;
  const total = paramsOf(model);
  const active = meta?.specs?.active_params;
  const parts: { key: string; text: string; title?: string }[] = [
    { key: 'org', text: meta?.organization || 'Unknown organization' },
    ...(meta?.release_date ? [{ key: 'release', text: formatRelease(meta.release_date, longDate) }] : []),
    ...(total
      ? [{ key: 'size', text: showActive && active && active !== total ? `${formatSize(total)} (${formatSize(active)} active)` : formatSize(total), title: active && active !== total ? `${formatSize(total)} total, ${formatSize(active)} active parameters` : `${formatSize(total)} parameters` }]
      : []),
  ];
  // Each part is its own flex item so a narrow card wraps between parts, never inside one.
  return (
    <p className={`flex flex-wrap items-center text-muted-foreground ${className}`}>
      <ProviderLogo org={meta?.organization} className={`text-ink ${logoClassName}`} />
      {parts.map((part, i) => (
        <span key={part.key} title={part.title} className="whitespace-nowrap">
          {part.text}{i < parts.length - 1 && ' ·'}
        </span>
      ))}
      {meta?.open_weights != null && <WeightsTag open={meta.open_weights} />}
    </p>
  );
}

export function ModelCard({ model, strong, weak }: { model: JoinedModelData; strong: AreaStanding[]; weak: AreaStanding[] }) {
  const meta = model.metadata;
  const name = meta?.name || model.id;
  return (
    <Link
      href={`/models/${model.yamlId}`}
      className={`group w-full transition-colors hover:border-ink/30 ${baseCardClass}`}
    >
      <h2 className="font-serif text-base font-bold leading-tight tracking-tight text-ink group-hover:text-link transition-colors">{name}</h2>
      <ModelMeta model={model} className="mt-1.5 gap-1.5 text-xs" />

      <StatGrid className="mt-4 sm:mt-6">
        {INDICES.map(index => {
          const theta = getIndexTheta(model, index.id);
          return (
            <Stat
              key={index.id}
              title={index.description}
              label={index.label}
              value={theta ? formatScore(theta.theta) : <span className="text-muted-foreground/50">—</span>}
              sub={theta ? formatInterval(theta.interval) : 'No score'}
            />
          );
        })}
      </StatGrid>

      {(strong.length > 0 || weak.length > 0) && (
        <div className="mt-4 sm:mt-6 space-y-2 sm:space-y-3">
          {strong.length > 0 && <StandingList label="Excels at" items={strong} tone="strong" />}
          {weak.length > 0 && <StandingList label="Falls behind on" items={weak} tone="weak" />}
        </div>
      )}
    </Link>
  );
}
