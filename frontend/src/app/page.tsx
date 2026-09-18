import { Suspense } from 'react';
import { getJoinedModels } from '@/lib/data';
import { IndexExplorer } from '@/components/IndexExplorer';

export default function Home() {
  const models = getJoinedModels();

  return (
    <main className="min-h-screen">
      <Suspense fallback={<div className="h-screen flex items-center justify-center text-gray-500">Loading Index...</div>}>
        <IndexExplorer models={models} />
      </Suspense>
    </main>
  );
}
