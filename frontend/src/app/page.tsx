import { Suspense } from 'react';
import { getJoinedModels, getLabels } from '@/lib/data';
import { IndexExplorer } from '@/components/IndexExplorer';

export default function Home() {
  const models = getJoinedModels();
  const labels = getLabels();

  return (
    <main className="min-h-screen">
      <Suspense fallback={<div className="h-screen flex items-center justify-center text-gray-500">Loading Index...</div>}>
        <IndexExplorer models={models} schema={labels._schema} />
      </Suspense>
    </main>
  );
}
