import { getJoinedModels } from '@/lib/data';
import { ModelDataTable } from '@/components/ModelDataTable';
import { Badge } from '@/components/ui/badge';

export default function ModelsPage() {
  const models = getJoinedModels();

  return (
    <div className="container mx-auto px-4 max-w-7xl py-12 space-y-8">
      <div className="max-w-3xl space-y-3">
        <h1 className="text-3xl font-medium tracking-tight">Evaluate exactly what matters</h1>
        <p className="text-gray-500 text-sm">
          Browse the {models.length} models evaluated in the COMPL-AI Index. Sort by parameter size, organization, or dive deep into specific domain performance profiles.
        </p>
      </div>

      <ModelDataTable initialModels={models} />
    </div>
  );
}
