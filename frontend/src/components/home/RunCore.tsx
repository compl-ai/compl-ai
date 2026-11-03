import Link from 'next/link';
import { CopyCommand } from '@/components/home/CopyCommand';
import { README_CORE_URL, README_QUICKSTART_URL } from '@/lib/site';
import { Eyebrow } from '@/components/layout/Eyebrow';

// Commands are copied from the repository README; keep them in sync with it.
const INSTALL = [
  'git clone https://github.com/compl-ai/compl-ai.git',
  'cd compl-ai',
  'uv sync',
  'source .venv/bin/activate',
  'export OPENAI_API_KEY=your_key',
  'export HF_TOKEN=your_token',
];

const STEPS = [
  {
    title: 'Evaluate',
    text: 'Run your model on the CORE tasks. This is the default for complai eval.',
    lines: ['complai eval openai/gpt-5-nano --log-dir logs/'],
  },
  {
    title: 'Predict',
    text: 'Turn the CORE results into predicted performance across the full evaluation suite.',
    lines: ['complai predict logs/<folder_name> --output predicted.json'],
  },
  {
    title: 'Score',
    text: 'Get capability, safety and reliability scores and compare them with the published results for the same CORE version.',
    lines: null,
  },
] as const;

export function RunCore({ subsetItems }: { subsetItems: number | null }) {
  return (
    <section id="run-core" className="scroll-mt-24 border-t border-border bg-muted/30">
      <div className="max-w-[1400px] mx-auto px-4 md:px-8 py-16 md:py-20">
        <div className="max-w-3xl">
          <Eyebrow>Run CORE</Eyebrow>
          <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight text-ink">Evaluate your own model</h2>
          <p className="mt-4 text-base text-muted-foreground leading-relaxed">
            CORE is open source. Evaluate on the {subsetItems ? `${subsetItems.toLocaleString()}-task ` : ''}set with any supported provider,
            predict the full-suite results, and score your model against the models on this site.
          </p>
        </div>

        <div className="mt-10 grid gap-6 lg:grid-cols-3">
          {STEPS.map((s, i) => (
            <div key={s.title} className="flex min-w-0 flex-col rounded-2xl border border-border bg-white p-6">
              <div className="flex items-center gap-3">
                <span className="flex h-7 w-7 items-center justify-center rounded-full bg-brand font-mono text-xs font-semibold text-ink">{i + 1}</span>
                <h3 className="text-xl font-bold text-ink">{s.title}</h3>
              </div>
              <p className="mt-3 text-sm text-muted-foreground leading-relaxed">{s.text}</p>
              <div className="mt-auto pt-4">
                {s.lines ? (
                  <CopyCommand lines={[...s.lines]} />
                ) : (
                  <div className="flex flex-wrap gap-2">
                    <a href="#results" className="rounded-full border border-border px-4 py-2 text-sm font-semibold text-ink hover:border-ink/30">Results</a>
                    <Link href="/models" className="rounded-full border border-border px-4 py-2 text-sm font-semibold text-ink hover:border-ink/30">All models</Link>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-10 grid gap-6 lg:grid-cols-[1fr_1fr] lg:items-start">
          <div className="min-w-0">
            <h3 className="text-lg font-bold text-ink">Install</h3>
            <p className="mt-1 mb-3 text-sm text-muted-foreground">
              Requires <code className="font-mono text-[13px]">uv</code>. <code className="font-mono text-[13px]">HF_TOKEN</code> needs access to the gated benchmark datasets.
            </p>
            <CopyCommand lines={INSTALL} />
          </div>
          <div className="lg:pt-9 space-y-2 text-sm">
            <p className="text-muted-foreground">
              Check that every task runs first with a one-item smoke test: <code className="font-mono text-[13px] text-ink">complai eval openai/gpt-5-nano --smoke 1</code>. Smoke runs are never used for scores.
            </p>
            <ul className="space-y-1.5 pt-2">
              <li><a href={README_QUICKSTART_URL} target="_blank" rel="noreferrer" className="font-semibold text-brand-strong hover:underline">Quickstart and providers on GitHub →</a></li>
              <li><a href={README_CORE_URL} target="_blank" rel="noreferrer" className="font-semibold text-brand-strong hover:underline">CORE subset evaluation →</a></li>
              <li><Link href="/methodology" className="font-semibold text-brand-strong hover:underline">How predictions are made →</Link></li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
