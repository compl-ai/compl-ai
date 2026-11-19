import { getCostOverview, getCosts, getExperiments, getJoinedModels, getLastUpdated } from '@/lib/data';
import { formatDate } from '@/lib/scores';
import { CoreStats, HomeHero } from '@/components/home/HomeHero';
import { ResultsExplorer } from '@/components/home/ResultsExplorer';
import { RunCore } from '@/components/home/RunCore';
import { ModelDataTable } from '@/components/ModelDataTable';
import { Eyebrow } from '@/components/layout/Eyebrow';

function SectionHeading({ eyebrow, title, children, className = 'mb-8' }: { eyebrow?: string; title: string; children?: React.ReactNode; className?: string }) {
  return (
    <div className={`max-w-3xl ${className}`}>
      {eyebrow && <Eyebrow>{eyebrow}</Eyebrow>}
      <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight text-ink">{title}</h2>
      {children && <p className="mt-4 text-base text-muted-foreground leading-relaxed">{children}</p>}
    </div>
  );
}

export default function Home() {
  const models = getJoinedModels();
  const suite = getCosts().suite;
  const lastUpdated = getLastUpdated();
  const explorer = getExperiments();
  const costOverview = getCostOverview();
  const costs = Object.fromEntries(costOverview.rows.map(r => [r.id, r]));

  return (
    <main className="min-h-screen">
      <HomeHero benchmarks={Object.keys(suite.tasks).length} />
      <CoreStats suite={suite} />

      <section id="results" className="scroll-mt-24 frame pt-8 md:pt-14 pb-20">
        <div className="cols-10">
        <div className="mb-14">
          <Eyebrow>Results</Eyebrow>
          <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight text-ink">How do today&rsquo;s AI models compare?</h2>
          {lastUpdated && <p className="mt-2 text-sm text-faint-foreground">Model evaluation data last updated {formatDate(lastUpdated)}</p>}
          <div className="max-w-3xl">
            <p className="mt-4 text-base text-muted-foreground leading-relaxed">
              Every model gets three separate scores, one each for safety, capability, and reliability. For each category, we estimate the model&apos;s ability
              (&theta;) with item response theory, which accounts for how difficult each task is. We publish &theta; rescaled so that
              100 is the average evaluated model and 15 points is one standard deviation. Higher is better; scores are not percentages.
            </p>
          </div>
        </div>
        <ResultsExplorer models={models} explorer={explorer} costs={costOverview} />
        <div id="rankings" className="scroll-mt-24 mt-12 md:mt-16">
        <ModelDataTable
          initialModels={models}
          costs={costs}
          initialRows={15}
          heading={
            <SectionHeading title="Model rankings" className="" />
          }
        />
        </div>
        </div>
      </section>

      <RunCore subsetItems={suite.subset_items} />
    </main>
  );
}
