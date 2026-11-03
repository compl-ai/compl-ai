import type { SuiteEfficiency } from '@/lib/data';
import { CATEGORIES } from '@/components/home/IndexLegend';
import { CoreMark3D } from '@/components/home/CoreMark3D';

const BUTTON = 'font-plex rounded-full px-4 py-1.5 text-base font-medium transition-colors';

export function HomeHero({ benchmarks }: { benchmarks: number | null }) {
  return (
    <section className="max-w-[1400px] mx-auto px-4 md:px-8 lg:px-20 xl:px-36 pt-16 md:pt-28 pb-16 md:pb-24 grid gap-10 md:grid-cols-[1fr_auto] md:items-center">
      <div>
        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight text-ink leading-[1.02]">
          Run fewer tests.<br />Measure more.
        </h1>
        <p className="mt-8 font-plex text-[20px] text-muted-foreground max-w-[780px] leading-relaxed">
          Comprehensive AI evaluation at a fraction of the cost.
          <br />
          Evaluate AI models on a compact set of calibrated tasks and predict their performance
          on {benchmarks ?? 'the'} established benchmarks across capability, safety, and reliability.
        </p>
        <div className="mt-10 flex flex-wrap gap-3">
          <a href="#run-core" className={`${BUTTON} border border-ink bg-ink text-white hover:bg-ink/85`}>Run CORE</a>
          <a href="#results" className={`${BUTTON} border border-ink bg-white text-ink hover:bg-muted`}>Explore Results</a>
        </div>
      </div>
      <CoreMark3D className="hidden md:block w-40 lg:w-48 h-auto md:mr-8 lg:mr-16 xl:mr-0 md:-translate-y-5 md:scale-115" />
    </section>
  );
}

function StatTag({ kind }: { kind: 'evaluate' | 'predict' | 'score' }) {
  const marker = {
    evaluate: <span className="h-2.5 w-2.5 rounded-[3px] bg-brand/30" />,
    predict: <span className="h-2.5 w-2.5 rounded-[3px] border-[1.5px] border-dashed border-ink/40" />,
    score: (
      <span className="inline-flex gap-[2px]">
        {CATEGORIES.map(c => <span key={c.id} className={`h-1.5 w-1.5 rounded-full ${c.dot}`} />)}
      </span>
    ),
  }[kind];
  return (
    <span className="flex items-center gap-1.5 font-plex text-[11px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
      {marker}
      {kind}
    </span>
  );
}

// "Evaluate" is what a model actually runs; "predict" is the full suite that CORE scores are projected onto.
function FlowArrow() {
  return (
    <svg aria-hidden viewBox="0 0 40 24" className="hidden md:block md:self-center md:-translate-x-1.5 h-6 w-10 text-ink/30" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12h32M26 3l9 9-9 9" />
    </svg>
  );
}

export function CoreStats({ efficiency }: { efficiency: SuiteEfficiency }) {
  const benchmarks = Object.keys(efficiency.tasks ?? {}).length;
  const number = 'font-serif text-2xl lg:text-3xl xl:text-[36px] font-extrabold tracking-tight text-ink tabular leading-none';
  const label = 'mt-2 font-plex text-sm text-muted-foreground leading-snug max-w-[18rem]';

  return (
    <section className="max-w-[1400px] mx-auto px-4 md:px-8 pb-16" aria-label="CORE at a glance">
      <div className="grid gap-8 md:grid-cols-[minmax(0,18rem)_auto_minmax(0,18rem)_auto_minmax(13rem,1fr)] lg:grid-cols-[minmax(0,18rem)_auto_minmax(0,18rem)_auto_minmax(16rem,1fr)] md:items-start md:gap-6 xl:gap-10 border-y border-border py-5 lg:py-6 px-0 [--bleed:24px] min-[1440px]:[--bleed:40px] md:-mx-[var(--bleed)] md:px-[var(--bleed)]">
        <div>
          <StatTag kind="evaluate" />
          <p className={`mt-2 ${number}`}>{efficiency.subset_items.toLocaleString()} <span className="text-lg lg:text-xl">tasks</span></p>
          <p className={label}>Run a compact evaluation instead of the full suite.</p>
        </div>
        <FlowArrow />
        <div>
          <StatTag kind="predict" />
          <p className={`mt-2 ${number}`}>
            {benchmarks} <span className="text-lg lg:text-xl">benchmarks</span>
          </p>
          <p className={label}>
            {efficiency.population_items.toLocaleString()} tasks: performance predicted across the full evaluation suite.
          </p>
        </div>
        <FlowArrow />
        <div>
          <StatTag kind="score" />
          <ul className="mt-2 space-y-1.5">
            {CATEGORIES.map(c => (
              <li key={c.id} className="font-plex text-[13px] text-muted-foreground leading-snug">
                <span className="mr-1.5 font-serif text-[15px] font-extrabold tracking-tight text-ink">{c.label}</span>
                {c.question}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
