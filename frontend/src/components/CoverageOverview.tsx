import type { GroundingRef } from '@/lib/data';

// Frameworks are named in text rather than colour-coded, so the index colours keep their meaning.
export const FRAMEWORK_LABEL: Record<GroundingRef['framework'], string> = {
  'EU AI Act': 'AI Act',
  'GPAI Code of Practice': 'Code of Practice',
};

export interface CoverageItem {
  id: string;
  item: GroundingRef;
  // Where CORE touches the reference: evaluation-area anchors, or a whole index.
  usages: { label: string; href?: string }[];
}

export interface CoverageColumn {
  title: string;
  description: string;
  tone: 'supported' | 'partial' | 'none';
  items: CoverageItem[];
}

// Shades of ink only: darker means more direct evidence.
const TONE: Record<CoverageColumn['tone'], { bar: string; rule: string; title: string }> = {
  supported: { bar: 'bg-ink/80', rule: 'border-t-2 border-ink/80', title: 'text-ink' },
  partial: { bar: 'bg-ink/35', rule: 'border-t-2 border-ink/35', title: 'text-ink' },
  none: { bar: 'bg-ink/10', rule: 'border-t-2 border-dashed border-ink/25', title: 'text-ink/70' },
};

export function CoverageOverview({ columns, footnote }: { columns: CoverageColumn[]; footnote?: React.ReactNode }) {
  const total = columns.reduce((sum, c) => sum + c.items.length, 0);
  return (
    <div className="space-y-8">
      <figure className="space-y-2">
        <div className="flex h-3 w-full gap-0.5 overflow-hidden rounded-full" aria-hidden>
          {columns.map(c => (
            <div key={c.tone} className={TONE[c.tone].bar} style={{ width: `${(c.items.length / total) * 100}%` }} />
          ))}
        </div>
        <figcaption className="text-xs text-muted-foreground">
          {total} mapped references:{' '}
          {columns.map((c, i) => (
            <span key={c.tone}>
              {i > 0 && ', '}
              <span className="font-semibold text-ink">{c.items.length}</span> {c.title.toLowerCase()}
            </span>
          ))}
          .
        </figcaption>
      </figure>
      <div className="grid grid-cols-1 gap-10 md:grid-cols-3 md:gap-8">
        {columns.map(c => (
          <section key={c.tone} className={`pt-3 ${TONE[c.tone].rule}`}>
            <div className="flex items-baseline gap-2">
              <h3 className={`text-lg font-extrabold tracking-tight ${TONE[c.tone].title}`}>{c.title}</h3>
              <span className="ml-auto tabular text-lg font-bold text-ink/70">{c.items.length}</span>
            </div>
            <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">{c.description}</p>
            <ul className="mt-5 space-y-4">
              {c.items.map(({ id, item: r, usages }) => (
                <li key={id} className="text-sm leading-snug">
                  <span className="block text-xs text-muted-foreground">
                    {FRAMEWORK_LABEL[r.framework]} · {r.short}
                  </span>
                  <a
                    href={r.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    title={r.excerpt}
                    className={`font-semibold hover:underline ${c.tone === 'none' ? 'text-ink/70' : 'text-ink'}`}
                  >
                    {r.title}
                  </a>
                  {usages.length > 0 && (
                    <span className="mt-0.5 block text-xs text-muted-foreground">
                      In CORE:{' '}
                      {usages.map((u, i) => (
                        <span key={u.label}>
                          {i > 0 && ', '}
                          {u.href ? (
                            <a href={u.href} className="text-ink/80 underline decoration-ink/25 underline-offset-2 hover:decoration-ink">
                              {u.label}
                            </a>
                          ) : (
                            u.label
                          )}
                        </span>
                      ))}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
      {footnote}
    </div>
  );
}
