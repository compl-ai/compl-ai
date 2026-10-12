import { getLabels } from '@/lib/data';
import { BenchmarkList } from '@/components/BenchmarkList';

export default function BenchmarksPage() {
  const labels = getLabels();
  
  const tasks: any[] = [];
  const indexCounts: Record<string, number> = {};
  const indexVsModality: Record<string, Record<string, number>> = {};
  
  Object.keys(labels).forEach(taskName => {
    if (taskName === '_schema') return;
    
    const taskObj = labels[taskName] || {};
    let sampleCount = 0;
    
    if (taskObj.columns && Array.isArray(taskObj.data)) {
      sampleCount = taskObj.data.length;
      
      const index = taskObj.index as string;
      const tagsIdx = taskObj.columns.indexOf('tags');
      indexCounts[index] = (indexCounts[index] || 0) + sampleCount;
      if (!indexVsModality[index]) indexVsModality[index] = {};
      
      taskObj.data.forEach((row: any[]) => {
        const tags = (row[tagsIdx] as string[]) || [];
        tags.forEach(tag => {
          if (tag.startsWith('modality:')) {
            const mod = tag.split(':')[1];
            indexVsModality[index][mod] = (indexVsModality[index][mod] || 0) + 1;
          }
        });
      });
    } else {
      sampleCount = Object.keys(taskObj).length;
    }
    
    tasks.push({ name: taskName, sampleCount });
  });

  const indicesList = Object.keys(indexCounts).sort();
  const modalitiesSet = new Set<string>();
  Object.values(indexVsModality).forEach(mods => {
    Object.keys(mods).forEach(m => modalitiesSet.add(m));
  });
  const modalitiesList = Array.from(modalitiesSet).sort();

  return (
    <div className="container mx-auto px-4 max-w-7xl py-12 space-y-16">
      <div className="space-y-4 max-w-3xl">
        <h1 className="text-3xl font-medium tracking-tight text-gray-900">Taxonomy & Coverage</h1>
        <p className="text-gray-500 text-sm">
          COMPL-AI evaluates systems based on the specific content of individual test items, not just the name of the benchmark. Each benchmark belongs to one of 3 indices (Capability, Reliability, Safety), and its samples are labelled with sub-categories within that index.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
        <div className="space-y-4">
          <h2 className="text-lg font-medium text-gray-900 border-b pb-2">Sample Distribution by Index</h2>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="py-2 text-left font-medium text-gray-500">Index</th>
                <th className="py-2 text-right font-medium text-gray-500">Evaluated Samples</th>
              </tr>
            </thead>
            <tbody>
              {indicesList.map(index => (
                <tr key={index} className="border-b border-gray-100 last:border-0 hover:bg-gray-50">
                  <td className="py-2.5 font-medium text-gray-900 capitalize">{index}</td>
                  <td className="py-2.5 text-right font-mono text-xs text-gray-600">{indexCounts[index].toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="space-y-4">
          <h2 className="text-lg font-medium text-gray-900 border-b pb-2">Index × Modality Heatmap</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="py-2 text-left font-medium text-gray-500">Index</th>
                  {modalitiesList.map(m => (
                    <th key={m} className="py-2 text-right font-medium text-gray-500 capitalize">{m}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {indicesList.map(index => (
                  <tr key={index} className="border-b border-gray-100 last:border-0 hover:bg-gray-50">
                    <td className="py-2.5 font-medium text-gray-900 capitalize">{index}</td>
                    {modalitiesList.map(m => {
                      const count = indexVsModality[index]?.[m] || 0;
                      return (
                        <td key={m} className="py-2 text-right">
                          {count > 0 ? (
                            <span className="inline-block px-2 py-1 bg-primary/10 text-primary font-mono text-xs font-semibold rounded-sm">
                              {count.toLocaleString()}
                            </span>
                          ) : (
                            <span className="text-gray-300 font-mono text-xs">—</span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="pt-8">
        <div className="space-y-4 max-w-2xl mb-8">
          <h2 className="text-lg font-medium text-gray-900 border-b pb-2">Source Tasks</h2>
        </div>
        <BenchmarkList initialTasks={tasks} />
      </div>
    </div>
  );
}
