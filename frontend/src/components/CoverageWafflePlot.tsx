'use client';

import { JoinedModelData } from '@/lib/data';

interface CoverageWafflePlotProps {
  models: JoinedModelData[];
}

export function CoverageWafflePlot({ models }: CoverageWafflePlotProps) {
  // 1. Gather all unique benchmarks and sort alphabetically
  const benchmarks = Array.from(
    new Set(
      models.flatMap((m) => Object.keys(m.prediction.tasks || {}))
    )
  ).sort();

  // 2. Sort models by total_params descending
  const sortedModels = [...models].sort((a, b) => {
    const paramsA = a.metadata?.specs?.total_params || 0;
    const paramsB = b.metadata?.specs?.total_params || 0;
    return paramsB - paramsA; // larger models first
  });

  return (
    <div className="w-full bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden flex flex-col mt-8">
      <div className="p-4 border-b border-gray-100 flex items-center justify-between bg-gray-50/50">
        <h3 className="font-semibold text-gray-900 flex items-center gap-2">
          <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
          </svg>
          Benchmark Coverage Grid
        </h3>
        <div className="flex items-center gap-4 text-xs text-gray-500">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-sm bg-blue-500 border border-blue-600"></div>
            Completed
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-sm bg-amber-400 border border-amber-500"></div>
            Partial
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-sm bg-gray-100 border border-gray-200"></div>
            Missing
          </div>
        </div>
      </div>
      
      <div className="overflow-x-auto relative pb-8">
        <table className="w-full text-left border-collapse border-spacing-0 text-sm">
          <thead className="sticky top-0 z-20 bg-white">
            <tr>
              <th className="sticky left-0 z-30 bg-white p-3 min-w-[200px] border-b border-r border-gray-200 font-medium text-gray-500 align-bottom">
                <div className="mb-2">Model</div>
                <div className="text-[10px] font-normal text-gray-400">Sorted by Parameters</div>
              </th>
              {benchmarks.map((b) => (
                <th
                  key={b}
                  className="p-2 border-b border-gray-200 align-bottom h-48 whitespace-nowrap bg-white"
                >
                  <div className="w-6 relative h-full">
                    <span className="absolute bottom-0 left-1/2 origin-bottom-left -rotate-45 text-[11px] text-gray-600 font-medium tracking-tight">
                      {b}
                    </span>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="overflow-y-auto">
            {sortedModels.map((m, idx) => (
              <tr key={m.id} className={`hover:bg-gray-50 group ${idx !== sortedModels.length - 1 ? 'border-b border-gray-100' : ''}`}>
                <td className="sticky left-0 z-10 bg-white group-hover:bg-gray-50 px-3 py-1.5 border-r border-gray-200 text-xs font-medium text-gray-900 truncate max-w-[200px] transition-colors">
                  {m.metadata?.name || m.id.split('/').pop()}
                  <div className="text-[10px] text-gray-400 font-normal mt-0.5">
                    {m.metadata?.specs?.total_params ? `${(m.metadata.specs.total_params / 1e9).toFixed(0)}B params` : 'Unknown size'}
                  </div>
                </td>
                {benchmarks.map((b) => {
                  const taskRes = m.prediction.tasks?.[b];
                  const status = taskRes?.status || 'missing';

                  let bgClass = 'bg-gray-100 border-gray-200 hover:bg-gray-200';
                  let titleStatus = 'Missing';

                  if (status === 'ok') {
                    bgClass = 'bg-blue-500 border-blue-600 shadow-[inset_0_1px_0_rgba(255,255,255,0.2)] hover:bg-blue-600';
                    titleStatus = 'Completed';
                  } else if (status === 'partial') {
                    bgClass = 'bg-amber-400 border-amber-500 shadow-[inset_0_1px_0_rgba(255,255,255,0.2)] hover:bg-amber-500';
                    titleStatus = 'Partial';
                  }

                  return (
                    <td key={b} className="p-1 min-w-[32px] text-center">
                      <div
                        className={`w-5 h-5 mx-auto rounded-sm border transition-colors ${bgClass}`}
                        title={`${m.metadata?.name || m.id}: ${b} (${titleStatus})`}
                      />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
