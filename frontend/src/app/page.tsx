import { getEfficiency, getJoinedModels } from '@/lib/data';
import { CoreStats, HomeHero } from '@/components/home/HomeHero';
import { ResultsExplorer } from '@/components/home/ResultsExplorer';
import { computeFindings } from '@/components/home/explorer';
import { RunCore } from '@/components/home/RunCore';
import { ModelDataTable } from '@/components/ModelDataTable';
import { Eyebrow } from '@/components/layout/Eyebrow';

function SectionHeading({ eyebrow, title, children }: { eyebrow: string; title: string; children?: React.ReactNode }) {
  return (
    <div className="max-w-3xl mb-8">
      <Eyebrow>{eyebrow}</Eyebrow>
      <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight text-ink">{title}</h2>
      {children && <p className="mt-4 text-base text-muted-foreground leading-relaxed">{children}</p>}
    </div>
  );
}

export default function Home() {
  const models = getJoinedModels();
  const efficiency = getEfficiency();

  return (
    <main className="min-h-screen">
      <HomeHero benchmarks={efficiency?.tasks ? Object.keys(efficiency.tasks).length : null} />
      {efficiency && <CoreStats efficiency={efficiency} />}

      <section id="results" className="scroll-mt-24 max-w-[1400px] mx-auto px-4 md:px-8 pt-8 md:pt-14 pb-16">
        <div className="mb-14">
          <div className="max-w-3xl">
            <Eyebrow>Results</Eyebrow>
            <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight text-ink">How do today&apos;s AI models compare?</h2>
            <p className="mt-4 text-base text-muted-foreground leading-relaxed">
              Every model gets three separate scores, one each for safety, capability, and reliability. For each category, we estimate the model&apos;s ability
              (&theta;) with item response theory, which accounts for how difficult each task is. We publish &theta; rescaled so that
              100 is the average evaluated model and 15 points is one standard deviation. Higher is better; scores are not percentages.
            </p>
          </div>
        </div>
        <ResultsExplorer models={models} findings={computeFindings(models)} />
      </section>

      <section id="rankings" className="scroll-mt-24 max-w-[1400px] mx-auto px-4 md:px-8 pt-8 md:pt-14 pb-20">
        <SectionHeading eyebrow="Rankings" title="Model rankings">
          Rank models by one index at a time. Click a model for its evaluation-area and benchmark breakdown.
        </SectionHeading>
        <ModelDataTable initialModels={models} initialRows={15} indexTabs />
      </section>

      <RunCore subsetItems={efficiency?.subset_items ?? null} />
    </main>
  );
}
