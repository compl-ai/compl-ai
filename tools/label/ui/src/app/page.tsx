"use client";

import { useEffect, useState, Suspense } from "react";
import { DatasetViewer } from "@/components/DatasetViewer";
import { DashboardStats } from "@/components/DashboardStats";
import { Badge } from "@/components/ui/badge";
import { useRouter, useSearchParams } from "next/navigation";

function AppContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [datasets, setDatasets] = useState<{id: string, count: number}[]>([]);
  
  const currentTab = searchParams.get('tab') || 'overview';
  const filterType = searchParams.get('filterType');
  const filterValue = searchParams.get('filterValue');

  useEffect(() => {
    fetch('/api/datasets')
      .then(res => res.json())
      .then(data => {
        if (data.datasets) setDatasets(data.datasets);
      })
      .catch(console.error);
  }, []);

  const handleTabChange = (tab: string) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set('tab', tab);
    params.delete('filterType'); // Clear filter when changing tabs
    params.delete('filterValue');
    router.push(`?${params.toString()}`);
  };

  return (
    <div className="h-screen overflow-hidden bg-slate-50 text-slate-900 font-sans p-4 flex flex-col gap-4">
      <header className="flex items-center justify-between shrink-0 border-b border-slate-200 pb-4">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold tracking-tight">Labeling Inspector</h1>
          <Badge variant="secondary">Live UI</Badge>
          
          <div className="flex items-center gap-2 ml-4">
            <button
              onClick={() => handleTabChange('overview')}
              className={`px-3 py-1.5 text-sm font-semibold rounded-md border transition-colors ${
                currentTab === 'overview' 
                  ? 'border-indigo-200 bg-indigo-50 text-indigo-700 shadow-sm' 
                  : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-slate-900 shadow-sm'
              }`}
            >
              Overview
            </button>
            
            <div className="relative group flex items-center">
              <select
                className={`appearance-none text-sm font-semibold cursor-pointer outline-none pl-3 pr-8 py-1.5 rounded-md border transition-colors ${
                  currentTab !== 'overview' 
                    ? 'border-indigo-200 bg-indigo-50 text-indigo-700 shadow-sm' 
                    : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-slate-900 shadow-sm'
                }`}
                value={currentTab !== 'overview' ? currentTab : ''}
                onChange={(e) => {
                  if (e.target.value) handleTabChange(e.target.value);
                }}
              >
                <option value="" disabled>Select Benchmark...</option>
                {datasets.map(ds => (
                  <option key={ds.id} value={ds.id}>
                    {ds.id} ({ds.count})
                  </option>
                ))}
              </select>
              <div className={`pointer-events-none absolute right-2.5 flex items-center ${
                currentTab !== 'overview' ? 'text-indigo-500' : 'text-slate-400'
              }`}>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
              </div>
            </div>
            
            {filterValue && (
              <Badge variant="outline" className="ml-2 flex items-center gap-1.5 bg-white border-slate-300 shadow-sm py-1">
                <span className="text-slate-500 font-normal uppercase tracking-wider text-[9px]">{filterType === 'primary_label' ? 'Primary' : filterType === 'secondary_labels' ? 'Secondary' : filterType === 'human_review' ? 'Status' : 'Tag'}:</span>
                <span className="font-semibold">{filterValue === 'needs_review' ? 'Needs Review' : filterValue}</span>
                <button 
                  onClick={() => {
                    const params = new URLSearchParams(searchParams.toString());
                    params.delete('filterType');
                    params.delete('filterValue');
                    router.push(`?${params.toString()}`);
                  }} 
                  className="ml-1 -mr-1 text-slate-400 hover:text-rose-500 transition-colors"
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M6 18L18 6M6 6l12 12"></path></svg>
                </button>
              </Badge>
            )}
          </div>
        </div>
      </header>

      <main className="flex flex-col gap-4 flex-1 min-h-0">

        <div className="flex-1 overflow-hidden flex flex-col min-h-0">
          {currentTab === 'overview' ? (
            <div className="flex-1 overflow-y-auto pt-2 flex flex-col">
               <DashboardStats isExpanded />
               <div className="mt-4 pb-8 text-center text-slate-400 text-sm">
                 Select a dataset tab above to view individual samples.
               </div>
            </div>
          ) : (
            <div className="flex-1 flex flex-col min-h-0">
              <DatasetViewer datasetName={currentTab} filterType={filterType} filterValue={filterValue} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default function Home() {
  return (
    <Suspense fallback={<div className="p-8 text-slate-500">Loading...</div>}>
      <AppContent />
    </Suspense>
  )
}
