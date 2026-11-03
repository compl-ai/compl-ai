import { getJoinedModels, getTaxonomy } from '@/lib/data';
import { ModelCard } from '@/components/ModelCard';
import { Eyebrow } from '@/components/layout/Eyebrow';
import { getIndexTheta } from '@/lib/scores';
import { areaStandings } from './strengths';

export default function ModelsPage() {
  const models = getJoinedModels();
  const standings = areaStandings(models, getTaxonomy());
  const capability = (m: (typeof models)[number]) => getIndexTheta(m, 'capability')?.theta ?? -Infinity;
  const sorted = [...models].sort((a, b) => capability(b) - capability(a));

  return (
    <div className="container mx-auto px-4 max-w-7xl py-12 space-y-8">
      <header className="max-w-3xl">
        <Eyebrow>Models</Eyebrow>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-ink">All evaluated models</h1>
        <p className="mt-4 text-base text-muted-foreground leading-relaxed">
          The {models.length} models in the ComplAI CORE results, ordered by capability. Strengths and weaknesses compare each
          evaluation-area score with the other models: top or bottom quarter. Open a model for its full evaluation-area and
          benchmark breakdown.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {sorted.map(model => (
          <ModelCard key={model.id} model={model} {...standings(model)} />
        ))}
      </div>
    </div>
  );
}
