'use client';
import { useState } from 'react';
import Link from 'next/link';
import { Input } from '@/components/ui/input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { JoinedModelData } from '@/lib/data';
import { ArrowUpDown } from 'lucide-react';

export function ModelDataTable({ initialModels }: { initialModels: JoinedModelData[] }) {
  const [search, setSearch] = useState('');
  const [sortConfig, setSortConfig] = useState<{ key: string, direction: 'asc' | 'desc' }>({
    key: 'overall',
    direction: 'desc'
  });

  const getScore = (m: JoinedModelData, key: string) => {
    if (key === 'overall') return m.prediction?.predicted_score || 0;
    return m.prediction?.domains?.[key] || 0;
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
    
    // Domain sorting
    const sA = getScore(a, sortConfig.key);
    const sB = getScore(b, sortConfig.key);
    return sortConfig.direction === 'asc' ? sA - sB : sB - sA;
  });

  const requestSort = (key: string) => {
    let direction: 'asc' | 'desc' = 'desc';
    if (sortConfig.key === key && sortConfig.direction === 'desc') {
      direction = 'asc';
    }
    setSortConfig({ key, direction });
  };

  const SortableHead = ({ label, sortKey, align = 'left' }: { label: string, sortKey: string, align?: 'left'|'right'|'center' }) => (
    <TableHead 
      className={`cursor-pointer hover:text-gray-900 transition-colors h-10 px-2 border-b text-xs font-medium ${align === 'right' ? 'text-right' : align === 'center' ? 'text-center' : ''}`}
      onClick={() => requestSort(sortKey)}
    >
      <div className={`flex items-center gap-1 ${align === 'right' ? 'justify-end' : align === 'center' ? 'justify-center' : ''} ${sortConfig.key === sortKey ? 'text-primary' : 'text-gray-500'}`}>
        {label}
        <ArrowUpDown className="w-3 h-3 opacity-40" />
      </div>
    </TableHead>
  );

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center max-w-full">
        <div className="relative">
          <Input 
            type="search" 
            placeholder="Filter models..." 
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-64 h-10 pl-4 pr-4 rounded-full border-gray-200 bg-white text-sm shadow-[0_2px_10px_rgba(0,0,0,0.02)] focus-visible:ring-blue-600"
          />
        </div>
      </div>

      <div className="overflow-x-auto">
        <Table className="w-full text-sm">
          <TableHeader>
            <TableRow className="border-b border-gray-200 hover:bg-transparent">
              <SortableHead label="Model" sortKey="name" />
              <TableHead className="h-12 px-3 border-b text-xs font-semibold text-gray-500 uppercase tracking-wider">Org</TableHead>
              <SortableHead label="Params" sortKey="params" align="right" />
              <SortableHead label="Overall" sortKey="overall" align="right" />
              <SortableHead label="Capability" sortKey="capability" align="right" />
              <SortableHead label="Safety" sortKey="safety" align="right" />
              <SortableHead label="Security" sortKey="security-privacy" align="right" />
              <SortableHead label="Reliability" sortKey="reliability" align="right" />
              <SortableHead label="Fairness" sortKey="fairness-bias" align="right" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedModels.map((model) => (
              <TableRow key={model.id} className="group hover:bg-gray-50/50 border-b border-gray-100 transition-colors">
                <TableCell className="px-3 py-3 font-semibold text-gray-900 truncate max-w-[200px]">
                  <Link href={`/models/${model.yamlId}`} className="hover:text-blue-600 transition-colors">
                    {model.metadata?.name || model.yamlId}
                  </Link>
                </TableCell>
                <TableCell className="px-3 py-3 text-gray-500 truncate max-w-[150px]">
                  {model.metadata?.organization || '—'}
                </TableCell>
                <TableCell className="px-3 py-3 text-right text-gray-500 font-mono text-xs">
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
                
                {['overall', 'capability', 'safety', 'security-privacy', 'reliability', 'fairness-bias'].map(domain => {
                  const score = getScore(model, domain);
                  const isSorted = sortConfig.key === domain;
                  const display = score > 0 ? (score * 100).toFixed(1) : '—';
                  
                  return (
                    <TableCell key={domain} className="px-3 py-3 w-[100px]">
                      {score > 0 ? (
                        <div className="flex flex-col items-end gap-1.5">
                          <span className={`font-mono text-xs ${isSorted ? 'font-bold text-gray-900' : 'text-gray-500 font-medium'}`}>
                            {display}
                          </span>
                          <div className="w-full h-1 bg-gray-100 rounded-full overflow-hidden flex justify-end">
                            <div 
                              className={`h-full rounded-full ${isSorted ? 'bg-blue-600' : 'bg-gray-300'}`} 
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
