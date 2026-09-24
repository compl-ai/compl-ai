'use client';
import { useState, useEffect, useMemo } from 'react';
import { useSearchParams, useRouter, usePathname } from 'next/navigation';
import { JoinedModelData } from '@/lib/data';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, ZAxis, Cell } from 'recharts';
import { ShieldCheck, MessageSquareWarning, Target, Users, Gauge } from 'lucide-react';
import { ModelDataTable } from '@/components/ModelDataTable';
import { CoverageWafflePlot } from '@/components/CoverageWafflePlot';

const DOMAINS = [
  { id: 'overall', label: 'Overall', icon: null, badge: '' },
  { id: 'security-privacy', label: 'Security & Privacy', icon: ShieldCheck, badge: '' },
  { id: 'safety', label: 'Safety', icon: MessageSquareWarning, badge: '' },
  { id: 'reliability', label: 'Reliability', icon: Target, badge: '' },
  { id: 'fairness-bias', label: 'Fairness & Bias', icon: Users, badge: '' },
  { id: 'capability', label: 'Capability', icon: Gauge, badge: '' },
];

const ORG_COLORS: Record<string, string> = {
  'OpenAI': '#EC4899',
  'Google': '#10B981',
  'Anthropic': '#8B5CF6',
  'Meta': '#F97316',
  'xAI': '#0284C7',
  'Mistral AI': '#F43F5E',
  'Alibaba': '#3B82F6',
  'DeepSeek': '#06B6D4',
  'NVIDIA': '#84CC16',
  'Zhipu AI': '#6366F1',
  'Microsoft': '#0EA5E9',
  'Tencent': '#14B8A6',
  'ByteDance': '#EAB308',
  'Moonshot AI': '#A855F7',
  'Mimo AI': '#F472B6',
  'MiniMax': '#FB923C',
  'StepFun': '#4ADE80',
  'Swiss AI': '#EF4444',
  'IBM': '#1D4ED8',
  'Allen AI': '#0D9488',
  'Ai2': '#0D9488',
  'Qwen': '#3B82F6',
  'Default': '#9CA3AF'
};

export function IndexExplorer({ models, schema }: { models: JoinedModelData[], schema?: any }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const domainParam = searchParams.get('domain');
  
  const [activeDomain, setActiveDomain] = useState(domainParam || 'overall');
  const [xAxisMode, setXAxisMode] = useState<'date' | 'params'>('date');
  const [isMounted, setIsMounted] = useState(false);

  const dynamicDomains = DOMAINS.map(d => {
    if (d.id === 'overall') {
      return { ...d, badge: schema?.total_subset_items?.toString() || d.badge };
    }
    const domInfo = schema?.domains?.[d.id];
    if (domInfo && domInfo.subset_weight_pct) {
      return { ...d, badge: domInfo.subset_weight_pct };
    }
    return d;
  });

  useEffect(() => {
    setIsMounted(true);
    if (domainParam && DOMAINS.some(d => d.id === domainParam)) {
      setActiveDomain(domainParam);
    }
  }, [domainParam]);

  const handleDomainChange = (val: string) => {
    setActiveDomain(val);
    const params = new URLSearchParams(searchParams.toString());
    params.set('domain', val);
    router.replace(`${pathname}?${params.toString()}`, { scroll: false });
  };

  const getScore = (m: JoinedModelData, domain: string = activeDomain) => {
    if (domain === 'overall') return m.prediction?.predicted_score || 0;
    return m.prediction?.domains?.[domain] || 0;
  };

  const parseParams = (v: any) => typeof v === 'number' ? v : Number(String(v || '0').replace(/_/g, ''));

  // Heuristic to extract params from ID if metadata is missing
  const guessParams = (id: string) => {
    const matchB = id.match(/(\d+(?:\.\d+)?)[bB]/);
    if (matchB) return parseFloat(matchB[1]) * 1e9;
    const matchM = id.match(/(\d+(?:\.\d+)?)[mM]/);
    if (matchM) return parseFloat(matchM[1]) * 1e6;
    return 0;
  };

  const sortedModels = [...models].sort((a, b) => getScore(b) - getScore(a));

  const chartData = sortedModels
    .filter(m => {
      if (xAxisMode === 'date') return !!m.metadata?.release_date;
      const p = parseParams(m.metadata?.specs?.total_params) || guessParams(m.yamlId);
      return p > 0;
    })
    .map(m => {
      const org = m.metadata?.organization || Object.keys(ORG_COLORS).find(k => m.yamlId.toLowerCase().includes(k.toLowerCase())) || 'Other';
      let orgKey = Object.keys(ORG_COLORS).find(k => org.includes(k)) || 'Default';
      
      let xVal = 0;
      if (xAxisMode === 'date') {
        xVal = new Date(m.metadata!.release_date!).getTime();
      } else {
        xVal = parseParams(m.metadata?.specs?.total_params) || guessParams(m.yamlId);
      }

      return {
        id: m.yamlId,
        name: m.metadata?.name || m.yamlId,
        org: org,
        fill: ORG_COLORS[orgKey],
        x: xVal,
        y: getScore(m) * 100,
        label: (getScore(m) > 0.6 || orgKey !== 'Default') ? (m.metadata?.name || m.yamlId) : '',
        date: m.metadata?.release_date ? new Date(m.metadata.release_date).getTime() : 0,
        params: parseParams(m.metadata?.specs?.total_params) || guessParams(m.yamlId)
      };
    });

  const dynamicDateTicks = useMemo(() => {
    if (xAxisMode !== 'date' || chartData.length === 0) return undefined;
    const min = Math.min(...chartData.map(d => d.x));
    const max = Math.max(...chartData.map(d => d.x));
    if (min === max) return [min];
    
    // Create 6 evenly spaced explicit tick marks across the timeline
    const ticks = [];
    const step = (max - min) / 5;
    for (let i = 0; i <= 5; i++) {
      ticks.push(min + i * step);
    }
    return ticks;
  }, [chartData, xAxisMode]);

  const dateFormatter = (tick: number) => {
    return new Date(tick).toLocaleDateString(undefined, { timeZone: 'UTC', month: 'short', year: 'numeric' });
  };
  
  const paramsFormatter = (tick: number) => {
    if (tick >= 1e9) return (tick / 1e9).toFixed(0) + 'B';
    if (tick >= 1e6) return (tick / 1e6).toFixed(0) + 'M';
    return tick.toString();
  };

  const CustomTooltip = ({ active, payload }: { active?: boolean, payload?: any[] }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white border shadow-sm p-3 text-xs z-50 rounded-xl min-w-[150px]">
          <div className="font-bold mb-1" style={{ color: data.fill }}>{data.name}</div>
          <div className="text-gray-500 mb-3">{data.org}</div>
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between gap-4">
              <span className="text-gray-500">Score:</span>
              <span className="font-mono font-medium">{data.y.toFixed(1)}</span>
            </div>
            {data.date > 0 && (
              <div className="flex justify-between gap-4">
                <span className="text-gray-500">Date:</span>
                <span className="font-mono">{new Date(data.date).toLocaleDateString(undefined, { timeZone: 'UTC' })}</span>
              </div>
            )}
            {data.params > 0 && (
              <div className="flex justify-between gap-4">
                <span className="text-gray-500">Params:</span>
                <span className="font-mono">{paramsFormatter(data.params)}</span>
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full flex flex-col items-center">
      {/* Huge Centered Hero */}
      <div className="pt-20 pb-16 text-center w-full">
        <h1 className="text-6xl md:text-7xl font-extrabold tracking-tight text-gray-900 mb-8">Index</h1>
        
        {/* Mirror the selected domain as a single elegant active pill at the top, or hide it if we prefer sidebar */}
        {/* We keep it to preserve the visual identity from the previous request */}
        <div className="flex justify-center w-full">
          <div className="inline-flex flex-wrap items-center bg-white p-1.5 rounded-full shadow-[0_8px_30px_rgb(0,0,0,0.06)] border border-gray-100 gap-1">
            {DOMAINS.map(d => {
              const Icon = d.icon;
              const isActive = activeDomain === d.id;
              return (
                <button
                  key={d.id}
                  onClick={() => handleDomainChange(d.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                    isActive 
                      ? 'bg-gray-800 text-white shadow-sm' 
                      : 'bg-transparent text-gray-500 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  {Icon && <Icon className={`w-4 h-4 ${isActive ? 'text-gray-300' : 'text-gray-400'}`} />}
                  <div className="flex items-center gap-1.5">
                    {d.label}
                    {d.badge ? (
                      <span className={`text-[10px] font-mono ${isActive ? 'text-gray-400' : 'text-gray-400'}`}>
                        ({d.badge})
                      </span>
                    ) : null}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <div className="w-full max-w-[1600px] px-4 flex flex-col xl:flex-row gap-8 pb-20">
        <div className="flex-1 space-y-4 min-w-0">
          <div className="bg-white border border-gray-200 rounded-3xl p-6 shadow-[0_8px_30px_rgb(0,0,0,0.04)] relative">
            <h2 className="absolute top-6 left-8 text-lg font-bold text-gray-900 z-10">
              {DOMAINS.find(d => d.id === activeDomain)?.label || 'Overall'} vs {xAxisMode === 'date' ? 'Release Date' : 'Parameters'}
            </h2>
            {isMounted && chartData.length > 0 ? (
              <div className="h-[600px] w-full mt-10">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: -20 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F3F4F6" />
                    <XAxis 
                      dataKey="x" 
                      type="number" 
                      domain={xAxisMode === 'params' ? [1e9, 2e12] : ['dataMin - 2592000000', 'dataMax + 2592000000']} 
                      scale={xAxisMode === 'params' ? 'log' : 'time'}
                      ticks={xAxisMode === 'params' ? [1e9, 3e9, 10e9, 30e9, 100e9, 300e9, 1e12] : dynamicDateTicks}
                      tickCount={xAxisMode === 'date' ? 8 : undefined}
                      tickFormatter={xAxisMode === 'date' ? dateFormatter : paramsFormatter}
                      tick={{ fill: '#6B7280', fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      tickMargin={12}
                      minTickGap={30}
                    />
                    <YAxis 
                      dataKey="y" 
                      type="number" 
                      domain={[0, 100]} 
                      tick={{ fill: '#6B7280', fontSize: 11, fontFamily: 'monospace' }}
                      tickLine={false}
                      axisLine={false}
                      tickMargin={8}
                    />
                    <ZAxis range={[60, 60]} />
                    <RechartsTooltip cursor={{ strokeDasharray: '3 3', stroke: '#D1D5DB' }} content={<CustomTooltip />} />
                    <Scatter name="Models" data={chartData}>
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} fillOpacity={0.8} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            ) : (
               <div className="h-[600px] w-full flex items-center justify-center text-gray-400">
                  No model data available to plot for the selected axis.
               </div>
            )}
          </div>
        </div>

        <div className="w-full xl:w-[360px] flex-shrink-0 space-y-6 pt-6 xl:pt-0">
          <div className="bg-white border border-gray-200 rounded-3xl p-6 shadow-[0_8px_30px_rgb(0,0,0,0.04)] h-full">
            
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-base font-bold text-gray-900">Settings</h3>
              <button className="text-gray-400 hover:text-gray-900 transition-colors">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 21v-5h5"/></svg>
              </button>
            </div>

            <div className="space-y-6 mb-8 pb-8 border-b border-gray-100">
              {/* X Axis Control */}
              <div className="space-y-3">
                <label className="text-xs font-semibold text-gray-900 flex items-center gap-2">
                  X-Axis <span className="text-gray-400 font-normal">Independent Variable</span>
                </label>
                <div className="flex bg-gray-50 p-1 rounded-lg border border-gray-200">
                  <button 
                    onClick={() => setXAxisMode('date')}
                    className={`flex-1 py-1.5 text-xs font-semibold rounded-md transition-all ${xAxisMode === 'date' ? 'bg-white text-gray-900 shadow-sm border border-gray-200/50' : 'text-gray-500 hover:text-gray-900'}`}
                  >
                    Release Date
                  </button>
                  <button 
                    onClick={() => setXAxisMode('params')}
                    className={`flex-1 py-1.5 text-xs font-semibold rounded-md transition-all ${xAxisMode === 'params' ? 'bg-white text-gray-900 shadow-sm border border-gray-200/50' : 'text-gray-500 hover:text-gray-900'}`}
                  >
                    Model Size
                  </button>
                </div>
              </div>

              {/* Y Axis Control */}
              <div className="space-y-3">
                <label className="text-xs font-semibold text-gray-900 flex items-center gap-2">
                  Y-Axis <span className="text-gray-400 font-normal">Evaluation Domain</span>
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {dynamicDomains.map(d => (
                    <button
                      key={d.id}
                      onClick={() => handleDomainChange(d.id)}
                      className={`flex items-center gap-2 px-3 py-2 rounded-md text-xs font-medium transition-all border text-left ${
                        activeDomain === d.id 
                          ? 'bg-blue-50 border-blue-200 text-blue-700' 
                          : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
                      }`}
                    >
                      <div className={`w-2 h-2 rounded-full ${activeDomain === d.id ? 'bg-blue-500' : 'bg-gray-300'}`} />
                      <div className="flex items-center gap-1.5">
                        {d.label}
                        {d.badge ? (
                          <span className="text-[10px] font-mono opacity-50">
                            ({d.badge})
                          </span>
                        ) : null}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <h3 className="text-sm font-bold text-gray-900 mb-4">Organization</h3>
            <div className="space-y-2 text-xs font-medium max-h-[220px] overflow-y-auto pr-2 custom-scrollbar">
              {Object.entries(ORG_COLORS)
                .map(([org, color]) => {
                  const count = chartData.filter(d => (Object.keys(ORG_COLORS).find(k => d.org.includes(k)) || 'Default') === org).length;
                  return { org, color, count };
                })
                .filter(item => item.count > 0)
                .sort((a, b) => b.count - a.count)
                .map(({ org, color, count }) => (
                <div key={org} className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2.5 truncate">
                    <div className="w-3 h-3 rounded-[3px] shrink-0" style={{ backgroundColor: color }}></div>
                    <span className="text-gray-600 truncate" title={org}>{org === 'Default' ? 'Other' : org}</span>
                  </div>
                  <span className="text-gray-400 text-[10px] font-mono bg-gray-100 px-1.5 py-0.5 rounded-full shrink-0">
                    {count}
                  </span>
                </div>
              ))}
            </div>
            
            <div className="mt-8 pt-6 border-t border-gray-100">
               <h3 className="text-sm font-bold text-gray-900 mb-2">Dataset Information</h3>
               <p className="text-xs text-gray-500 leading-relaxed mb-2">
                 Displaying <strong>{chartData.length}</strong> of <strong>{models.length}</strong> evaluated models.
               </p>
               <p className="text-xs text-gray-500 leading-relaxed mb-2">
                 Based on <strong>{models.reduce((acc, m) => acc + (m.prediction?.coverage?.samples_completed || 0), 0).toLocaleString()}</strong> total evaluation data points.
               </p>
               <p className="text-xs text-gray-400 leading-relaxed">
                 {models.length - chartData.length} models are hidden because they lack {xAxisMode === 'date' ? 'a known release date' : 'a known parameter count'}.
               </p>
            </div>
          </div>
        </div>
      </div>
      
      <div className="w-full max-w-[1600px] px-4 pb-32 space-y-8">
        <div className="bg-white border border-gray-200 rounded-3xl p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)]">
          <h3 className="text-xl font-bold text-gray-900 mb-6">Detailed Results</h3>
          <ModelDataTable initialModels={models} />
        </div>
        <CoverageWafflePlot models={models} />
      </div>
    </div>
  );
}
