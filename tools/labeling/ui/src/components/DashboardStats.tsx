"use client";

import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, LabelList } from "recharts";

const getHexColor = (domain: string, variant: 'primary' | 'tag' = 'primary') => {
  if (variant === 'tag') {
    return '#64748b'; // slate-500
  }
  const d = domain.toLowerCase();
  if (d === 'safety') return '#f97316'; // orange-500
  if (d === 'security-privacy') return '#6366f1'; // indigo-500
  if (d === 'fairness-bias') return '#a855f7'; // purple-500
  if (d === 'reliability') return '#14b8a6'; // teal-500
  if (d === 'capability') return '#0ea5e9'; // sky-500
  if (d === 'failed') return '#ef4444'; // red-500
  return '#94a3b8'; // slate-400
};

export function DashboardStats({ datasetData, onFilterClick, isExpanded }: { datasetData?: any[], onFilterClick?: (type: string, value: string) => void, isExpanded?: boolean }) {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(!datasetData);

  useEffect(() => {
    if (datasetData) {
      const localStats = {
        primaryLabels: {} as Record<string, number>,
        secondaryLabels: {} as Record<string, number>,
        confidence: { high: 0, medium: 0, low: 0 } as Record<string, number>,
        needsReview: 0,
        tags: {} as Record<string, number>,
        totalSamples: 0,
      };

      datasetData.forEach(row => {
        localStats.totalSamples++;
        const llm = row.llm_assigned;
        const hasLLMData = llm && Object.keys(llm).length > 0;
        const isApproved = row._human_patch?.human_approved;
        
        const isReviewNeeded = hasLLMData && !isApproved && (!llm.primary_label || llm.label_confidence === 'low' || llm.primary_label === 'failed');
        if (isReviewNeeded) localStats.needsReview++;

        if (llm) {
          const conf = (llm.label_confidence || '').toLowerCase();
          if (conf === 'high' || conf === 'medium' || conf === 'low') {
            localStats.confidence[conf]++;
          } else if (conf === 'mid') {
            localStats.confidence['medium']++;
          }

          if (llm.primary_label) {
            localStats.primaryLabels[llm.primary_label] = (localStats.primaryLabels[llm.primary_label] || 0) + 1;
          }
          if (Array.isArray(llm.secondary_labels)) {
            for (const sl of llm.secondary_labels) {
              localStats.secondaryLabels[sl] = (localStats.secondaryLabels[sl] || 0) + 1;
            }
          }

          if (Array.isArray(llm.tags)) {
            for (const tag of llm.tags) {
              localStats.tags[tag] = (localStats.tags[tag] || 0) + 1;
            }
          }
        }
      });
      setStats(localStats);
      setLoading(false);
    } else {
      setLoading(true);
      fetch('/api/stats')
        .then(res => res.json())
        .then(setStats)
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, [datasetData]);

  if (loading || !stats) {
    return <div className={`bg-slate-50 animate-pulse rounded-xl ${isExpanded ? 'h-full min-h-[500px]' : 'h-[210px]'}`} />;
  }

  const primaryData = Object.entries(stats.primaryLabels || {}).map(([fullName, value]) => {
    const parts = fullName.split(':');
    return { name: parts.length > 1 ? parts.slice(1).join(':') : parts[0], domain: parts[0], fullName, value };
  }).sort((a, b) => (b.value as number) - (a.value as number));
  
  const secondaryData = Object.entries(stats.secondaryLabels || {}).map(([fullName, value]) => {
    const parts = fullName.split(':');
    return { name: parts.length > 1 ? parts.slice(1).join(':') : parts[0], domain: parts[0], fullName, value };
  }).sort((a, b) => (b.value as number) - (a.value as number));
  
  const tagData = Object.entries(stats.tags || {}).map(([fullName, value]) => {
    const parts = fullName.split(':');
    return { name: parts.length > 1 ? parts.slice(1).join(':') : parts[0], domain: parts[0], fullName, value };
  }).sort((a, b) => (b.value as number) - (a.value as number));
  
  const confidenceData = [
    { name: 'High', value: stats.confidence?.high || 0, color: '#10b981' },
    { name: 'Medium', value: stats.confidence?.medium || 0, color: '#f59e0b' },
    { name: 'Low', value: stats.confidence?.low || 0, color: '#f43f5e' },
  ].filter(d => d.value > 0);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-900 text-slate-50 text-xs px-3 py-2 rounded shadow-md z-50">
          <span className="font-semibold">{payload[0].payload.name}:</span> {payload[0].value.toLocaleString()}
        </div>
      );
    }
    return null;
  };

  return (
    <div className={`flex gap-4 p-4 shrink-0 overflow-hidden ${isExpanded ? 'h-full min-h-[600px]' : 'h-[210px]'}`}>
      


      {/* Primary Labels */}
      <div className="flex-1 flex flex-col h-full border-r border-slate-100 pr-2 min-w-0">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1">Top Primary</div>
        <div className="flex-1 min-h-0 w-full overflow-y-auto pr-2 custom-scrollbar">
          <div style={{ height: Math.max(100, primaryData.length * 28) }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={primaryData} layout="vertical" margin={{ top: 0, right: 35, left: 0, bottom: 0 }}>
                <XAxis type="number" hide />
                <YAxis dataKey="name" type="category" width={110} tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: '#f1f5f9' }} />
                <Bar dataKey="value" radius={[0, 2, 2, 0]} barSize={16} minPointSize={3}
                  activeBar={false}
                  background={{ fill: 'transparent' }}
                  onClick={(data: any) => onFilterClick && onFilterClick('primary_label', data.fullName)}
                  style={{ cursor: onFilterClick ? 'pointer' : 'default' }}
                >
                  <LabelList dataKey="value" position="right" fill="#94a3b8" fontSize={10} fontWeight={600} />
                  {primaryData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getHexColor(entry.domain, 'primary')} style={{ outline: 'none' }} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Secondary Labels */}
      <div className="flex-1 flex flex-col h-full border-r border-slate-100 pr-2 min-w-0">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1">Top Secondary</div>
        <div className="flex-1 min-h-0 w-full overflow-y-auto pr-2 custom-scrollbar">
          <div style={{ height: Math.max(100, secondaryData.length * 28) }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={secondaryData} layout="vertical" margin={{ top: 0, right: 35, left: 0, bottom: 0 }}>
                <XAxis type="number" hide />
                <YAxis dataKey="name" type="category" width={140} tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: '#f1f5f9' }} />
                <Bar dataKey="value" radius={[0, 2, 2, 0]} barSize={16} minPointSize={3}
                  activeBar={false}
                  background={{ fill: 'transparent' }}
                  onClick={(data: any) => onFilterClick && onFilterClick('secondary_labels', data.fullName)}
                  style={{ cursor: onFilterClick ? 'pointer' : 'default' }}
                >
                  <LabelList dataKey="value" position="right" fill="#94a3b8" fontSize={10} fontWeight={600} />
                  {secondaryData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getHexColor(entry.domain, 'primary')} style={{ outline: 'none' }} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Tags */}
      <div className="flex-1 flex flex-col h-full border-r border-slate-100 pr-2 min-w-0">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1">Tags</div>
        <div className="flex-1 min-h-0 w-full overflow-y-auto pr-2 custom-scrollbar">
          <div style={{ height: Math.max(100, tagData.length * 28) }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={tagData} layout="vertical" margin={{ top: 0, right: 35, left: 0, bottom: 0 }}>
                <XAxis type="number" hide />
                <YAxis dataKey="name" type="category" width={125} tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: '#f1f5f9' }} />
                <Bar dataKey="value" radius={[0, 2, 2, 0]} barSize={16} minPointSize={3}
                  activeBar={false}
                  background={{ fill: 'transparent' }}
                  onClick={(data: any) => onFilterClick && onFilterClick('tags', data.fullName)}
                  style={{ cursor: onFilterClick ? 'pointer' : 'default' }}
                >
                  <LabelList dataKey="value" position="right" fill="#94a3b8" fontSize={10} fontWeight={600} />
                  {tagData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getHexColor(entry.domain, 'tag')} style={{ outline: 'none' }} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Confidence */}
      <div className="flex flex-col items-center h-full min-w-[100px]">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-2">Confidence</div>
        <div className="w-full flex justify-center">
          {confidenceData.length > 0 ? (
            <div style={{ width: 100, height: 100 }}>
              <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={confidenceData}
                  cx="50%"
                  cy="50%"
                  innerRadius={35}
                  outerRadius={50}
                  paddingAngle={2}
                  dataKey="value"
                  stroke="none"
                >
                  {confidenceData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
            </div>
          ) : (
            <span className="text-[10px] text-slate-400">No Data</span>
          )}
        </div>
      </div>

      {/* Review Needs */}
      <div className="flex flex-col justify-start items-center pl-4 border-l border-slate-100 min-w-[100px] h-full pt-0.5">
        <div 
          className={`flex flex-col justify-center items-center p-3 rounded-xl transition-colors ${onFilterClick ? 'cursor-pointer hover:bg-rose-50' : ''}`}
          onClick={() => onFilterClick?.('human_review', 'needs_review')}
        >
          <div className="text-3xl font-bold text-rose-500">{stats.needsReview?.toLocaleString() || 0}</div>
          <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mt-1 text-center leading-tight">Review<br/>Needed</div>
        </div>
      </div>

    </div>
  );
}
