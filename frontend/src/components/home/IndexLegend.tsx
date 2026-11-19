import type { IndexId } from '@/lib/scores';

export const CATEGORIES: { id: IndexId; label: string; dot: string; question: React.ReactNode }[] = [
  {
    id: 'capability',
    label: 'Capability',
    dot: 'bg-capability',
    question: <>How well does the model solve complex reasoning, math, coding, multimodal, and multilingual tasks?</>,
  },
  {
    id: 'safety',
    label: 'Safety',
    dot: 'bg-safety',
    question: <>How well does the model resist harmful requests, deception, manipulation, security threats, and biased behavior?</>,
  },
  {
    id: 'reliability',
    label: 'Reliability',
    dot: 'bg-reliability',
    question: <>How consistently does the model provide factual answers, follow instructions, and remain robust to changes in inputs?</>,
  },
];

export function IndexLegend({ horizontal = false }: { horizontal?: boolean }) {
  return (
    <dl className={`${horizontal ? 'grid gap-5 md:grid-cols-3 md:gap-8 xl:gap-10' : 'space-y-5'} font-plex text-[13px] leading-[1.55]`} aria-label="The three indices">
      {CATEGORIES.map(c => (
        <div key={c.id} className="flex gap-2.5">
          <span className={`mt-[6px] h-2.5 w-2.5 shrink-0 rounded-full ${c.dot}`} aria-hidden />
          <div>
            <dt className="inline font-semibold text-ink">{c.label}</dt>{' '}
            <dd className="inline text-muted-foreground">{c.question}</dd>
          </div>
        </div>
      ))}
    </dl>
  );
}
