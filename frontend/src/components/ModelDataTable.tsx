'use client';
import { useState, useMemo } from 'react';
import Link from 'next/link';
import { Input } from '@/components/ui/input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { JoinedModelData } from '@/lib/data';
import { ArrowUpDown } from 'lucide-react';

export function ModelDataTable({ initialModels }: { initialModels: JoinedModelData[] }) {
  const [search, setSearch] = useState('');
  const [viewMode, setViewMode] = useState<'domains' | 'benchmarks'>('domains');
  const [showActuals, setShowActuals] = useState(false);
  const [sortConfig, setSortConfig] = useState<{ key: string, direction: 'asc' | 'desc' }>({
    key: 'overall',
    direction: 'desc'
  });

  const allBenchmarks = useMemo(() => {
    const b = new Set<string>();
    initialModels.forEach(m => {
      Object.keys(m.prediction?.tasks || {}).forEach(k => b.add(k));
    });
    return Array.from(b).sort();
  }, [initialModels]);

  const getScore = (m: JoinedModelData, key: string) => {
    if (key === 'overall') return m.prediction?.predicted_score ?? null;
    if (viewMode === 'domains') {
      return m.prediction?.domains?.[key] ?? null;
    } else {
      const task = m.prediction?.tasks?.[key];
      if (task && task.status !== 'missing') {
        return task.predicted_score ?? null;
      }
      return null;
    }
  };

  const hasGroundTruth = useMemo(() => initialModels.some(m => m.groundTruth && Object.keys(m.groundTruth).length > 0), [initialModels]);

  const getActualScore = (m: JoinedModelData, key: string) => {
    if (key === 'overall') return null;
    if (m.groundTruth && typeof m.groundTruth[key] === 'number') {
      return m.groundTruth[key];
    }
    return null;
  };

  const filteredModels = initialModels.filter(m => {
    const term = search.toLowerCase();
    const name = (m.metadata?.name || m.yamlId).toLowerCase();
    const org = (m.metadata?.organization || '').toLowerCase();
    return name.includes(term) || org.includes(term);
  });

  const sortedModels = [...filteredModels].sort((a, b) => {
    if (sortConfig.key === 'name') {
      const nameA = a.metadata?.name || a.yamlId;
      const nameB = b.metadata?.name || b.yamlId;
      return sortConfig.direction === 'asc' ? nameA.localeCompare(nameB) : nameB.localeCompare(nameA);
    }
    if (sortConfig.key === 'params') {
      const parseP = (v: any) => typeof v === 'number' ? v : Number(String(v || '0').replace(/_/g, ''));
      const pA = parseP(a.metadata?.specs?.total_params);
      const pB = parseP(b.metadata?.specs?.total_params);
      return sortConfig.direction === 'asc' ? pA - pB : pB - pA;
    }
    if (sortConfig.key === 'coverage') {
      const cA = a.prediction?.coverage?.tasks_completed || 0;
      const cB = b.prediction?.coverage?.tasks_completed || 0;
      return sortConfig.direction === 'asc' ? cA - cB : cB - cA;
    }
    if (sortConfig.key === 'samples') {
      const sA = a.prediction?.coverage?.samples_completed || 0;
      const sB = b.prediction?.coverage?.samples_completed || 0;
      return sortConfig.direction === 'asc' ? sA - sB : sB - sA;
    }
    
    // Domain/Benchmark sorting
    const sA = getScore(a, sortConfig.key) ?? -1;
    const sB = getScore(b, sortConfig.key) ?? -1;
    return sortConfig.direction === 'asc' ? sA - sB : sB - sA;
  });

  const requestSort = (key: string) => {
    let direction: 'asc' | 'desc' = 'desc';
    if (sortConfig.key === key && sortConfig.direction === 'desc') {
      direction = 'asc';
    }
    setSortConfig({ key, direction });
  };

  const SortableHead = ({ label, sortKey, align = 'left', className = '' }: { label: string, sortKey: string, align?: 'left'|'right'|'center', className?: string }) => (
    <TableHead 
      className={`cursor-pointer hover:text-gray-900 transition-colors h-10 px-2 border-b text-xs font-medium ${align === 'right' ? 'text-right' : align === 'center' ? 'text-center' : ''} ${className}`}
      onClick={() => requestSort(sortKey)}
    >
      <div className={`flex items-center gap-1 ${align === 'right' ? 'justify-end' : align === 'center' ? 'justify-center' : ''} ${sortConfig.key === sortKey ? 'text-primary' : 'text-gray-500'}`}>
        {label}
        <ArrowUpDown className="w-3 h-3 opacity-40" />
      </div>
    </TableHead>
  );

  const DOMAIN_COLUMNS = ['overall', 'capability', 'safety', 'security-privacy', 'reliability', 'fairness-bias'];
  const columnsToRender = viewMode === 'domains' ? DOMAIN_COLUMNS : ['overall', ...allBenchmarks];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 max-w-full">
        <div className="relative">
          <Input 
            type="search" 
            placeholder="Filter models..." 
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-64 h-10 pl-4 pr-4 rounded-full border-gray-200 bg-white text-sm shadow-[0_2px_10px_rgba(0,0,0,0.02)] focus-visible:ring-blue-600"
          />
        </div>
        <div className="flex items-center gap-3">
          {hasGroundTruth && (
            <label className="flex items-center gap-1.5 text-xs font-medium text-gray-500 cursor-pointer hover:text-gray-900 transition-colors pr-2 border-r border-gray-200" title="Shows True Population Scores extracted from full dataset logs">
              <input 
                type="checkbox" 
                checked={showActuals}
                onChange={(e) => setShowActuals(e.target.checked)}
                className="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
              />
              Show True Scores
            </label>
          )}
          <div className="flex bg-gray-100 p-1 rounded-full">
            <button
              onClick={() => {
                setViewMode('domains');
                if (!DOMAIN_COLUMNS.includes(sortConfig.key) && sortConfig.key !== 'name' && sortConfig.key !== 'coverage' && sortConfig.key !== 'samples' && sortConfig.key !== 'params') {
                  setSortConfig({ key: 'overall', direction: 'desc' });
                }
              }}
              className={`px-4 py-1.5 rounded-full text-xs font-medium transition-all ${
                viewMode === 'domains' 
                  ? 'bg-white text-gray-900 shadow-sm' 
                  : 'text-gray-500 hover:text-gray-900'
              }`}
            >
              Domains
            </button>
            <button
              onClick={() => setViewMode('benchmarks')}
              className={`px-4 py-1.5 rounded-full text-xs font-medium transition-all ${
                viewMode === 'benchmarks' 
                  ? 'bg-white text-gray-900 shadow-sm' 
                  : 'text-gray-500 hover:text-gray-900'
              }`}
            >
              Benchmarks
            </button>
          </div>
        </div>
      </div>

      <div className="overflow-x-auto pb-4">
        <Table className="w-full text-sm">
          <TableHeader>
            <TableRow className="border-b border-gray-200 hover:bg-transparent">
              <SortableHead label="Model" sortKey="name" className="sticky left-0 bg-white z-10 shadow-[2px_0_5px_-2px_rgba(0,0,0,0.05)]" />
              <TableHead className="h-12 px-3 border-b text-xs font-semibold text-gray-500 uppercase tracking-wider">Org</TableHead>
              <SortableHead label="Tasks" sortKey="coverage" align="right" />
              <SortableHead label="Samples" sortKey="samples" align="right" />
              <SortableHead label="Params" sortKey="params" align="right" />
              {viewMode === 'domains' ? (
                <>
                  <SortableHead label="Overall" sortKey="overall" align="right" />
                  <SortableHead label="Capability" sortKey="capability" align="right" />
                  <SortableHead label="Safety" sortKey="safety" align="right" />
                  <SortableHead label="Security" sortKey="security-privacy" align="right" />
                  <SortableHead label="Reliability" sortKey="reliability" align="right" />
                  <SortableHead label="Fairness" sortKey="fairness-bias" align="right" />
                </>
              ) : (
                <>
                  <SortableHead label="Overall" sortKey="overall" align="right" />
                  {allBenchmarks.map(b => (
                    <SortableHead key={b} label={b} sortKey={b} align="right" />
                  ))}
                </>
              )}
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedModels.map((model) => (
              <TableRow key={model.id} className="group hover:bg-gray-50 border-b border-gray-100 transition-colors">
                <TableCell className="px-3 py-3 font-semibold text-gray-900 truncate max-w-[200px] sticky left-0 bg-white group-hover:bg-gray-50 z-10 shadow-[2px_0_5px_-2px_rgba(0,0,0,0.05)]">
                  <Link href={`/models/${model.yamlId}`} className="hover:text-blue-600 transition-colors">
                    {model.metadata?.name || model.yamlId}
                  </Link>
                </TableCell>
                <TableCell className="px-3 py-3 text-gray-500 truncate max-w-[150px]">
                  {model.metadata?.organization || '—'}
                </TableCell>
                <TableCell className="px-3 py-3 text-right text-gray-500 font-mono text-xs whitespace-nowrap">
                  {model.prediction?.coverage ? `${model.prediction.coverage.tasks_completed}/${model.prediction.coverage.tasks_total}` : '—'}
                </TableCell>
                <TableCell className="px-3 py-3 text-right text-gray-500 font-mono text-xs whitespace-nowrap">
                  {model.prediction?.coverage ? `${(model.prediction.coverage.samples_completed/1000).toFixed(1)}k/${(model.prediction.coverage.samples_total/1000).toFixed(1)}k` : '—'}
                </TableCell>
                <TableCell className="px-3 py-3 text-right text-gray-500 font-mono text-xs whitespace-nowrap">
                  {(() => {
                    const specs = model.metadata?.specs;
                    if (!specs || !specs.total_params) return '—';
                    const parseP = (v: any) => typeof v === 'number' ? v : Number(String(v).replace(/_/g, ''));
                    const t = parseP(specs.total_params);
                    const a = specs.active_params ? parseP(specs.active_params) : null;
                    if (isNaN(t) || t === 0) return '—';
                    const tStr = (t / 1e9).toFixed(0) + 'B';
                    if (a && a !== t && !isNaN(a)) {
                      return `${tStr} (${(a / 1e9).toFixed(0)}B)`;
                    }
                    return tStr;
                  })()}
                </TableCell>
                
                {columnsToRender.map(key => {
                  const score = getScore(model, key);
                  const actualScore = getActualScore(model, key);
                  
                  const isSorted = sortConfig.key === key;
                  const display = score !== null ? (score * 100).toFixed(1) : '—';
                  const displayActual = actualScore !== null ? (actualScore * 100).toFixed(1) : null;
                  
                  let barColor = isSorted ? 'bg-blue-600' : 'bg-gray-300';
                  let textColor = isSorted ? 'font-bold text-gray-900' : 'text-gray-500 font-medium';
                  let gapColor = 'text-gray-400 font-normal';
                  
                  if (score !== null && actualScore !== null) {
                    const delta = Math.abs(score * 100 - actualScore * 100);
                    if (delta > 10) {
                      gapColor = 'font-bold text-red-500';
                    } else if (delta > 5) {
                      gapColor = 'font-bold text-amber-500';
                    } else if (delta > 2.5) {
                      gapColor = 'font-bold text-blue-500';
                    }
                  }
                  
                  return (
                    <TableCell key={key} className="px-3 py-3 min-w-[100px]">
                      {score !== null ? (
                        <div className="flex flex-col items-end gap-1.5">
                          <span className={`font-mono text-xs ${textColor}`}>
                            {display}
                            {showActuals && displayActual !== null && (
                              <span className={`text-[10px] ml-1 ${gapColor}`}>
                                ({displayActual})
                              </span>
                            )}
                          </span>
                          <div className="w-full h-1 bg-gray-100 rounded-full overflow-hidden flex justify-end">
                            <div 
                              className={`h-full rounded-full ${barColor}`} 
                              style={{ width: `${score * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      ) : (
                        <div className="text-right text-gray-300 text-xs font-mono">—</div>
                      )}
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
