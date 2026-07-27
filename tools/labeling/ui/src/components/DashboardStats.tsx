"use client";

import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie } from "recharts";

export function DashboardStats({ datasetData }: { datasetData?: any[] }) {
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
        const isApproved = row._human_patch?.human_approved;
        const isReviewNeeded = !isApproved && (!llm || !llm.primary_label || llm.label_confidence === 'low');
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
              if (tag.startsWith('MODALITY:') || tag.startsWith('AGENT:')) {
                localStats.tags[tag] = (localStats.tags[tag] || 0) + 1;
              }
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
    return <div className="h-[210px] bg-slate-50 animate-pulse rounded-xl" />;
  }

  const primaryData = Object.entries(stats.primaryLabels || {}).map(([name, value]) => ({ name: name.split(':').pop(), value })).sort((a, b) => (b.value as number) - (a.value as number)).slice(0, 5);
  const secondaryData = Object.entries(stats.secondaryLabels || {}).map(([name, value]) => ({ name: name.split(':').pop(), value })).sort((a, b) => (b.value as number) - (a.value as number)).slice(0, 5);
  const tagData = Object.entries(stats.tags || {}).map(([name, value]) => ({ name: name.split(':').pop(), value })).sort((a, b) => (b.value as number) - (a.value as number)).slice(0, 5);
  
  const confidenceData = [
    { name: 'High', value: stats.confidence?.high || 0, color: '#10b981' },
    { name: 'Medium', value: stats.confidence?.medium || 0, color: '#f59e0b' },
    { name: 'Low', value: stats.confidence?.low || 0, color: '#f43f5e' },
  ].filter(d => d.value > 0);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-900 text-slate-50 text-[10px] px-2 py-1 rounded shadow-md z-50">
          <span className="font-semibold">{payload[0].payload.name}:</span> {payload[0].value.toLocaleString()}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="flex items-center gap-4 h-[210px] bg-white rounded-xl shadow-sm border border-slate-100 p-4 shrink-0 overflow-hidden">
      
      {/* Review Needs */}
      <div className="flex flex-col justify-center items-center px-2 border-r border-slate-100 min-w-[100px]">
        <div className="text-3xl font-bold text-rose-500">{stats.needsReview?.toLocaleString() || 0}</div>
        <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mt-1">Review Needed</div>
      </div>

      {/* Primary Labels */}
      <div className="flex-1 flex flex-col h-full border-r border-slate-100 pr-2 min-w-0">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1">Top Primary</div>
        <div className="flex-1 min-h-0 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={primaryData} layout="vertical" margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
              <XAxis type="number" hide />
              <YAxis dataKey="name" type="category" width={95} tick={{ fontSize: 9, fill: '#64748b' }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: '#f1f5f9' }} />
              <Bar dataKey="value" fill="#3b82f6" radius={[0, 2, 2, 0]} barSize={14} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Secondary Labels */}
      <div className="flex-1 flex flex-col h-full border-r border-slate-100 pr-2 min-w-0">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1">Top Secondary</div>
        <div className="flex-1 min-h-0 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={secondaryData} layout="vertical" margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
              <XAxis type="number" hide />
              <YAxis dataKey="name" type="category" width={140} tick={{ fontSize: 9, fill: '#64748b' }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: '#f1f5f9' }} />
              <Bar dataKey="value" fill="#0ea5e9" radius={[0, 2, 2, 0]} barSize={14} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Tags */}
      <div className="flex-1 flex flex-col h-full border-r border-slate-100 pr-2 min-w-0">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1">Modality & Agent</div>
        <div className="flex-1 min-h-0 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={tagData} layout="vertical" margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
              <XAxis type="number" hide />
              <YAxis dataKey="name" type="category" width={115} tick={{ fontSize: 9, fill: '#64748b' }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: '#f1f5f9' }} />
              <Bar dataKey="value" fill="#8b5cf6" radius={[0, 2, 2, 0]} barSize={14} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confidence */}
      <div className="flex flex-col items-center h-full min-w-[100px]">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0">Confidence</div>
        <div className="flex-1 w-full min-h-0 relative flex items-center justify-center">
          {confidenceData.length > 0 ? (
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
          ) : (
            <span className="text-[10px] text-slate-400">No Data</span>
          )}
        </div>
      </div>

    </div>
  );
}
