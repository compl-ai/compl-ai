import type { Metadata } from 'next';
import { pageMetadata } from '@/lib/site';
import { getJoinedModels, getTaxonomy } from '@/lib/data';
import { ModelCard } from '@/components/ModelCard';
import { Eyebrow } from '@/components/layout/Eyebrow';
import { getIndexTheta, scaleTheta, type IndexId } from '@/lib/scores';
import { paramsOf } from '@/components/home/explorer';
import { areaStandings } from './strengths';
import { ModelGrid, type GridItem } from './ModelGrid';

export const metadata: Metadata = pageMetadata('Models', 'Every model evaluated with ComplAI CORE, with Capability, Reliability and Safety scores.');

export default function ModelsPage() {
  const models = getJoinedModels();
  const standings = areaStandings(models, getTaxonomy());
  const score = (m: (typeof models)[number], id: IndexId) => {
    const t = getIndexTheta(m, id);
    return t ? scaleTheta(t.theta) : null;
  };
  const items: GridItem[] = models.map(m => ({
    id: m.id,
    name: m.metadata?.name || m.id,
    org: m.metadata?.organization || null,
    release: m.metadata?.release_date || null,
    params: paramsOf(m),
    capability: score(m, 'capability'),
    reliability: score(m, 'reliability'),
    safety: score(m, 'safety'),
    card: <ModelCard model={m} {...standings(m)} />,
  }));

  return (
    <div className="frame pt-22 pb-12 space-y-14">
      <header className="cols-10">
        <div className="max-w-3xl">
        <Eyebrow>Models</Eyebrow>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-ink">All evaluated models</h1>
        <p className="mt-4 text-base text-muted-foreground leading-relaxed">
          The {models.length} models in the ComplAI CORE results, newest first. Strengths and weaknesses compare each
          evaluation-area score with the other models: top or bottom quarter. Open a model for its full evaluation-area and
          benchmark breakdown.
        </p>
        </div>
      </header>

      <ModelGrid items={items} />
    </div>
  );
}
