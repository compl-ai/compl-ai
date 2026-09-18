import { getJoinedModels } from '@/lib/data';
import { notFound } from 'next/navigation';
import { DomainBarChart } from '@/components/DomainBarChart';
import Link from 'next/link';

export function generateStaticParams() {
  const models = getJoinedModels();
  return models.map((model) => ({
    id: model.yamlId,
  }));
}

export default async function ModelReportPage(props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const models = getJoinedModels();
  const model = models.find(m => m.yamlId === params.id);

  if (!model) {
    notFound();
  }

  const globalAvgs: Record<string, number> = {};
  ['capability', 'safety', 'security-privacy', 'reliability', 'fairness-bias'].forEach(domain => {
    const scores = models.map(m => m.prediction.domains?.[domain] || 0).filter(s => s > 0);
    globalAvgs[domain] = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
  });

  const barData = [
    { domain: 'Capability', score: model.prediction.domains?.['capability'] || 0, globalAvg: globalAvgs['capability'] },
    { domain: 'Safety', score: model.prediction.domains?.['safety'] || 0, globalAvg: globalAvgs['safety'] },
    { domain: 'Security', score: model.prediction.domains?.['security-privacy'] || 0, globalAvg: globalAvgs['security-privacy'] },
    { domain: 'Reliability', score: model.prediction.domains?.['reliability'] || 0, globalAvg: globalAvgs['reliability'] },
    { domain: 'Fairness & Bias', score: model.prediction.domains?.['fairness-bias'] || 0, globalAvg: globalAvgs['fairness-bias'] },
  ];

  const sortedByOverall = [...models].sort((a, b) => (b.prediction.predicted_score || 0) - (a.prediction.predicted_score || 0));
  const rank = sortedByOverall.findIndex(m => m.id === model.id) + 1;

  return (
    <div className="container mx-auto px-4 max-w-5xl py-8 space-y-12">
      <Link href="/models" className="text-sm text-gray-400 hover:text-gray-900 transition-colors inline-flex items-center">
        ← Back to models
      </Link>
      
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 border-b pb-8">
        <div className="space-y-3 max-w-2xl">
          <div className="flex flex-wrap items-center gap-3 text-xs font-medium text-gray-500">
            <span>{model.metadata?.organization || 'Unknown Org'}</span>
            {model.metadata?.release_date && (
              <>
                <span className="w-1 h-1 rounded-full bg-gray-300"></span>
                <span>Released {model.metadata.release_date}</span>
              </>
            )}
          </div>
          <h1 className="text-4xl font-medium tracking-tight text-gray-900">{model.metadata?.name || model.yamlId}</h1>
          <p className="text-sm text-gray-500 leading-relaxed max-w-xl">
            {model.metadata?.description || "No official description available for this model."}
          </p>
        </div>
        
        <div className="text-right">
          <div className="text-xs font-medium text-gray-500 mb-1 uppercase tracking-wider">Overall Score</div>
          <div className="text-5xl font-medium text-primary">{(model.prediction.predicted_score * 100).toFixed(1)}</div>
          <div className="text-xs text-gray-400 mt-2">Rank #{rank} of {models.length}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-12 gap-12">
        <div className="md:col-span-8 space-y-6">
          <h2 className="text-lg font-medium tracking-tight">Domain Profile</h2>
          <div className="pt-2">
            <DomainBarChart data={barData} />
          </div>
          
          <div className="pt-12">
            <h2 className="text-lg font-medium tracking-tight mb-4">Task Breakdown</h2>
            <div className="overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="py-2 text-left font-medium text-gray-500">Source Task</th>
                    <th className="py-2 text-right font-medium text-gray-500">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(model.prediction.tasks).map(([taskName, taskData]) => {
                    const val = taskData.predicted_score;
                    return (
                      <tr key={taskName} className="border-b border-gray-100 last:border-0 hover:bg-gray-50">
                        <td className="py-2.5 font-medium text-gray-900">{taskName}</td>
                        <td className="py-2.5 text-right font-mono text-xs text-gray-600">
                          {val > 0 ? (val * 100).toFixed(1) : <span className="text-gray-300">—</span>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        
        <div className="md:col-span-4 space-y-6">
          <h2 className="text-lg font-medium tracking-tight">Metadata</h2>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-500">Parameters</span>
              <span className="font-medium">
                {(() => {
                  const specs = model.metadata?.specs;
                  if (!specs || !specs.total_params) return '—';
                  const parseP = (v: any) => typeof v === 'number' ? v : Number(String(v).replace(/_/g, ''));
                  const t = parseP(specs.total_params);
                  const a = specs.active_params ? parseP(specs.active_params) : null;
                  if (isNaN(t) || t === 0) return '—';
                  const tStr = (t / 1e9).toFixed(0) + 'B';
                  if (a && a !== t && !isNaN(a)) {
                    return `${tStr} (${(a / 1e9).toFixed(0)}B active)`;
                  }
                  return tStr;
                })()}
              </span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-500">Architecture</span>
              <span className="font-medium">{model.metadata?.specs?.architecture || '—'}</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-500">Open Weights</span>
              <span className="font-medium">{model.metadata?.open_weights ? 'Yes' : 'No'}</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-500">Context Window</span>
              <span className="font-medium">{model.metadata?.specs?.context ? `${(model.metadata.specs.context / 1000).toFixed(0)}K` : '—'}</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-500">License</span>
              <span className="font-medium">{model.metadata?.license || '—'}</span>
            </div>
            {model.metadata?.sources?.length ? (
               <div className="pt-4">
                 <span className="text-gray-500 block mb-2">Sources</span>
                 <div className="flex flex-col gap-1.5">
                   {model.metadata.sources.map((s: any, i: number) => (
                     <a key={i} href={s.url} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline block truncate">
                       {s.title}
                     </a>
                   ))}
                 </div>
               </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
