"use client";

import { useEffect, useState, useRef } from "react";
import { DashboardStats } from "@/components/DashboardStats";
import {
  flexRender,
  getCoreRowModel,
  useReactTable,
  getSortedRowModel,
  SortingState,
} from "@tanstack/react-table";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ChevronDown, ChevronUp, ArrowUpDown, Flag, Search, GripVertical, CheckCircle, RotateCcw } from "lucide-react";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent
} from '@dnd-kit/core';
import { restrictToHorizontalAxis } from '@dnd-kit/modifiers';
import {
  SortableContext,
  arrayMove,
  horizontalListSortingStrategy,
  useSortable
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

const getCategoryColors = (domain: string, isTwoTone: boolean, variant: "primary" | "tag") => {
  if (domain === "safety") {
    return isTwoTone 
      ? { domainColor: "bg-orange-100 text-orange-900", specificColor: "bg-orange-50 text-orange-800", borderColor: "border-orange-200" }
      : { outlineClass: "bg-orange-50 text-orange-900 border-orange-200" };
  }
  if (domain === "security-privacy") {
    return isTwoTone 
      ? { domainColor: "bg-indigo-100 text-indigo-900", specificColor: "bg-indigo-50 text-indigo-800", borderColor: "border-indigo-200" }
      : { outlineClass: "bg-indigo-50 text-indigo-900 border-indigo-200" };
  }
  if (domain === "fairness-bias") {
    return isTwoTone 
      ? { domainColor: "bg-purple-100 text-purple-900", specificColor: "bg-purple-50 text-purple-800", borderColor: "border-purple-200" }
      : { outlineClass: "bg-purple-50 text-purple-900 border-purple-200" };
  }
  if (domain === "reliability") {
    return isTwoTone 
      ? { domainColor: "bg-teal-100 text-teal-900", specificColor: "bg-teal-50 text-teal-800", borderColor: "border-teal-200" }
      : { outlineClass: "bg-teal-50 text-teal-900 border-teal-200" };
  }
  if (domain === "capability") {
    return isTwoTone 
      ? { domainColor: "bg-sky-100 text-sky-900", specificColor: "bg-sky-50 text-sky-800", borderColor: "border-sky-200" }
      : { outlineClass: "bg-sky-50 text-sky-900 border-sky-200" };
  }
  
  if (isTwoTone) {
    if (variant === "tag") {
      return { domainColor: "bg-slate-200 text-slate-700", specificColor: "bg-slate-100 text-slate-800", borderColor: "border-slate-200" };
    }
    return { domainColor: "bg-slate-200 text-slate-800", specificColor: "bg-slate-50 text-slate-700", borderColor: "border-slate-300" };
  }
  
  return { outlineClass: "bg-white text-slate-700 border-slate-200" };
};

const TwoToneBadge = ({ label, variant = "primary" }: { label: string, variant?: "primary" | "tag" }) => {
  const isTwoTone = label && label.includes(':');
  
  if (!isTwoTone) {
    const colors = getCategoryColors(label, false, variant);
    return <Badge variant="outline" className={`font-semibold uppercase tracking-wider text-[10px] ${colors.outlineClass}`}>{label}</Badge>;
  }

  const parts = label.split(':');
  const domain = parts.slice(0, -1).join(':');
  const specific = parts[parts.length - 1];
  
  const colors = getCategoryColors(domain, true, variant);
  
  return (
    <div className={`inline-flex items-center text-[10px] border rounded-md overflow-hidden font-semibold uppercase tracking-wider whitespace-nowrap ${colors.borderColor}`}>
      <span className={`${colors.domainColor} px-2 py-0.5`}>{domain}</span>
      <span className={`${colors.specificColor} px-2 py-0.5`}>{specific}</span>
    </div>
  );
}

const ConfidenceBadge = ({ val }: { val: string }) => {
  if (!val) return null;
  const v = val.toLowerCase();
  let color = "bg-slate-100 text-slate-600";
  if (v === 'high') color = "bg-emerald-100 text-emerald-700 border-emerald-200";
  if (v === 'mid' || v === 'medium') color = "bg-amber-100 text-amber-700 border-amber-200";
  if (v === 'low') color = "bg-rose-100 text-rose-700 border-rose-200";
  return <Badge variant="outline" className={`${color} capitalize font-semibold shadow-sm`}>{val}</Badge>;
}

const DraggableTableHeader = ({ header }: { header: any }) => {
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({
    id: header.column.id,
  });
  
  const style = {
    opacity: isDragging ? 0.8 : 1,
    position: 'relative' as const,
    zIndex: isDragging ? 1 : 0,
    width: header.getSize(),
    transform: CSS.Translate.toString(transform),
    transition,
  };

  return (
    <TableHead 
      ref={setNodeRef}
      style={style}
      className="px-4 py-3 text-xs font-medium text-slate-500 uppercase tracking-wider bg-slate-50/95"
    >
      <div className="flex items-center gap-2">
        <button {...attributes} {...listeners} className="cursor-grab hover:text-slate-900 touch-none flex-shrink-0">
          <GripVertical className="w-3 h-3 opacity-50" />
        </button>
        {header.isPlaceholder ? null : (
          <div
            {...{
              className: header.column.getCanSort()
                ? "cursor-pointer select-none flex items-center gap-1 hover:text-slate-800"
                : "flex items-center gap-1",
              onClick: header.column.getToggleSortingHandler(),
            }}
          >
            {flexRender(header.column.columnDef.header, header.getContext())}
            {{
              asc: <ChevronUp className="w-3 h-3" />,
              desc: <ChevronDown className="w-3 h-3" />,
            }[header.column.getIsSorted() as string] ?? (
              header.column.getCanSort() ? <ArrowUpDown className="w-3 h-3 opacity-50" /> : null
            )}
          </div>
        )}
      </div>
    </TableHead>
  );
};

export function DatasetViewer({ datasetName }: { datasetName: string }) {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedRow, setSelectedRow] = useState<any | null>(null);
  const [sorting, setSorting] = useState<SortingState>([]);
  const [refreshKey, setRefreshKey] = useState(0);
  const tableContainerRef = useRef<HTMLDivElement>(null);
  
  const [taxonomy, setTaxonomy] = useState<{primary_labels: string[], tags: string[]}>({ primary_labels: [], tags: [] });

  useEffect(() => {
    fetch('/api/taxonomy').then(r => r.json()).then(setTaxonomy).catch(console.error);
  }, []);
  
  const [isEditing, setIsEditing] = useState(false);
  const [editLabel, setEditLabel] = useState("");
  const [editSecondaryLabels, setEditSecondaryLabels] = useState("");
  const [editTags, setEditTags] = useState("");
  const [editRationale, setEditRationale] = useState("");

  useEffect(() => {
    if (selectedRow) {
      setEditLabel(selectedRow.llm_assigned?.primary_label || "");
      setEditSecondaryLabels((selectedRow.llm_assigned?.secondary_labels || []).join(", "));
      setEditTags((selectedRow.llm_assigned?.tags || []).join(", "));
      setEditRationale(selectedRow.llm_assigned?.label_rationale || "");
    }
  }, [selectedRow]);

  async function handleSaveEdit() {
    if (!selectedRow) return;
    
    const patchData = {
      sample_id: selectedRow.sample_id,
      human_primary_label: editLabel,
      human_secondary_labels: editSecondaryLabels.split(',').map(t => t.trim()).filter(Boolean),
      human_tags: editTags.split(',').map(t => t.trim()).filter(Boolean),
      human_rationale: editRationale,
      human_approved: selectedRow._human_patch?.human_approved
    };
    
    await fetch(`/api/datasets/${datasetName}/patch`, {
      method: 'POST',
      body: JSON.stringify(patchData)
    });
    
    const newData = data.map(r => {
      if (r.sample_id === selectedRow.sample_id) {
        const nr = { ...r };
        nr._human_patch = patchData;
        if (!nr.llm_assigned) nr.llm_assigned = {};
        nr.llm_assigned.primary_label = patchData.human_primary_label;
        nr.llm_assigned.secondary_labels = patchData.human_secondary_labels;
        nr.llm_assigned.tags = patchData.human_tags;
        nr.llm_assigned.label_rationale = patchData.human_rationale;
        return nr;
      }
      return r;
    });
    
    setData(newData);
    setSelectedRow(newData.find(r => r.sample_id === selectedRow.sample_id));
    setIsEditing(false);
  }

  async function toggleApproval() {
    if (!selectedRow) return;
    
    const existingPatch = selectedRow._human_patch || {};
    const newApprovedState = !existingPatch.human_approved;
    
    const patchData = {
      sample_id: selectedRow.sample_id,
      human_primary_label: existingPatch.human_primary_label || selectedRow.llm_assigned?.primary_label,
      human_secondary_labels: existingPatch.human_secondary_labels || selectedRow.llm_assigned?.secondary_labels || [],
      human_tags: existingPatch.human_tags || selectedRow.llm_assigned?.tags || [],
      human_rationale: existingPatch.human_rationale || selectedRow.llm_assigned?.label_rationale,
      human_approved: newApprovedState
    };
    
    await fetch(`/api/datasets/${datasetName}/patch`, {
      method: 'POST',
      body: JSON.stringify(patchData)
    });
    
    const newData = data.map(r => {
      if (r.sample_id === selectedRow.sample_id) {
        const nr = { ...r };
        nr._human_patch = patchData;
        if (!nr.llm_assigned) nr.llm_assigned = {};
        nr.llm_assigned.primary_label = patchData.human_primary_label;
        nr.llm_assigned.secondary_labels = patchData.human_secondary_labels;
        nr.llm_assigned.tags = patchData.human_tags;
        nr.llm_assigned.label_rationale = patchData.human_rationale;
        return nr;
      }
      return r;
    });
    
    setData(newData);
    setSelectedRow(newData.find(r => r.sample_id === selectedRow.sample_id));
  }

  async function handleReset() {
    if (!selectedRow || !selectedRow._human_patch) return;
    
    const targetId = selectedRow.sample_id;
    await fetch(`/api/datasets/${datasetName}/patch?sample_id=${targetId}`, {
      method: 'DELETE'
    });
    
    setRefreshKey(k => k + 1);
  }

  useEffect(() => {
    if (data.length === 0) setLoading(true);
    Promise.all([
      fetch(`/api/datasets/${datasetName}`).then(res => res.text()),
      fetch(`/api/datasets/${datasetName}/patch`).then(res => res.text())
    ])
      .then(([datasetText, patchText]) => {
        const patchMap = new Map();
        if (patchText) {
          patchText.split('\n').filter(l => l.trim()).forEach(l => {
            try {
              const p = JSON.parse(l);
              patchMap.set(p.sample_id, p);
            } catch (e) {}
          });
        }

        const lines = datasetText
          .split('\n')
          .filter(line => line.trim().length > 0)
          .map(line => {
            try {
              const obj = JSON.parse(line);
              const patch = patchMap.get(obj.sample_id);
              if (patch) {
                obj._human_patch = patch;
                if (!obj.llm_assigned) obj.llm_assigned = {};
                if (patch.human_primary_label) obj.llm_assigned.primary_label = patch.human_primary_label;
                if (patch.human_secondary_labels) obj.llm_assigned.secondary_labels = patch.human_secondary_labels;
                if (patch.human_tags) obj.llm_assigned.tags = patch.human_tags;
                if (patch.human_rationale) obj.llm_assigned.label_rationale = patch.human_rationale;
              }
              return obj;
            } catch (e) {
              return null;
            }
          })
          .filter(Boolean);
        setData(lines);
        
        // If we have a selected row, update its reference to the fresh data so the sheet updates automatically
        setSelectedRow(current => current ? (lines.find(r => r.sample_id === current.sample_id) || null) : null);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [datasetName, refreshKey]);

  const columns = [
    {
      id: "sample_id",
      accessorKey: "sample_id",
      header: "ID",
      cell: (info: any) => <span className="font-mono text-xs">{String(info.getValue() || 'N/A')}</span>,
      size: 60,
    },
    {
      id: "input",
      accessorKey: "input",
      header: "Input",
      cell: (info: any) => {
        const val = info.getValue();
        const displayVal = typeof val === 'object' && val !== null ? JSON.stringify(val) : String(val || '');
        return (
          <div className="line-clamp-2 text-sm text-slate-600 max-w-[400px] whitespace-pre-wrap" title={displayVal}>
            {displayVal}
          </div>
        );
      },
      size: 400,
    },
    {
      id: "primary_label",
      accessorFn: (row: any) => row.llm_assigned?.primary_label || "-",
      header: "Primary Label",
      cell: (info: any) => {
        const val = info.getValue();
        return val && val !== "-" ? <TwoToneBadge label={String(val)} /> : <span className="text-slate-400 text-xs italic">none</span>;
      },
      size: 120,
    },
    {
      id: "secondary_labels",
      accessorFn: (row: any) => row.llm_assigned?.secondary_labels || [],
      header: "Secondary Labels",
      cell: (info: any) => {
        const val = info.getValue();
        if (!val || val.length === 0) return <span className="text-slate-400 text-xs italic">none</span>;
        return (
          <div className="flex gap-1 flex-wrap overflow-hidden">
            {val.slice(0, 2).map((t: string, i: number) => <TwoToneBadge key={i} label={t} variant="tag" />)}
            {val.length > 2 && <Badge variant="secondary" className="text-[10px] h-5 py-0 px-1.5 flex items-center shrink-0">+{val.length - 2}</Badge>}
          </div>
        );
      },
      size: 240,
    },
    {
      id: "tags",
      accessorKey: "llm_assigned.tags",
      header: "Tags",
      enableSorting: false,
      cell: (info: any) => {
        const val = info.getValue();
        if (!Array.isArray(val) || val.length === 0) return null;
        return (
          <div className="flex gap-1 flex-wrap overflow-hidden">
            {val.slice(0, 2).map((t: string, i: number) => <TwoToneBadge key={i} label={t} variant="tag" />)}
            {val.length > 2 && <Badge variant="secondary" className="text-[10px] h-5 py-0 px-1.5 flex items-center shrink-0">+{val.length - 2}</Badge>}
          </div>
        );
      },
      size: 240,
    },
    {
      id: "rationale",
      accessorKey: "llm_assigned.label_rationale",
      header: "Rationale",
      enableSorting: false,
      cell: (info: any) => (
        <div className="line-clamp-2 text-xs italic text-slate-500 max-w-[300px] whitespace-pre-wrap" title={String(info.getValue() || '')}>
          {String(info.getValue() || '')}
        </div>
      ),
      size: 300,
    },
    {
      id: "confidence",
      accessorKey: "llm_assigned.label_confidence",
      header: "Confidence",
      cell: (info: any) => <ConfidenceBadge val={info.getValue()} />,
      size: 120,
    },
    {
      id: "human_review",
      header: "Rev",
      accessorFn: (row: any) => {
        if (row._human_patch?.human_approved) return 'approved';
        const llm = row.llm_assigned;
        return (!llm || !llm.primary_label || llm.label_confidence === 'low') ? 'needs_review' : 'ok';
      },
      cell: (info: any) => {
        const status = info.getValue();
        if (status === 'approved') return <CheckCircle className="w-4 h-4 text-emerald-500 mx-auto" />;
        if (status === 'needs_review') return <Flag className="w-4 h-4 text-rose-500 mx-auto" />;
        return null;
      },
      size: 60,
    }
  ];

  const [columnOrder, setColumnOrder] = useState<string[]>(columns.map(c => c.id));

  const table = useReactTable({
    data,
    columns,
    state: { sorting, columnOrder },
    onSortingChange: setSorting,
    onColumnOrderChange: setColumnOrder,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const { rows } = table.getRowModel();
  
  const selectedIndex = selectedRow 
    ? rows.findIndex(r => r.original === selectedRow)
    : -1;

  const rowVirtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => tableContainerRef.current,
    estimateSize: () => 45,
    overscan: 10,
  });

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor)
  );

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (active && over && active.id !== over.id) {
      setColumnOrder((order) => {
        const oldIndex = order.indexOf(active.id as string);
        const newIndex = order.indexOf(over.id as string);
        return arrayMove(order, oldIndex, newIndex);
      });
    }
  }

  if (loading) {
    return <div className="p-8 text-center text-slate-500 flex-1 flex items-center justify-center">Loading dataset...</div>;
  }

  return (
    <div className="flex flex-col h-full overflow-hidden gap-4">
      {data.length > 0 && (
        <div className="shrink-0">
          <DashboardStats datasetData={data} />
        </div>
      )}
      
      <div 
        ref={tableContainerRef} 
        className="flex-1 overflow-auto bg-white rounded-xl shadow-sm border border-slate-100 relative"
      >
        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragEnd={handleDragEnd}
          modifiers={[restrictToHorizontalAxis]}
        >
          <div style={{ height: `${rowVirtualizer.getTotalSize()}px`, width: '100%', position: 'relative' }}>
            <Table className="w-max min-w-full text-left table-fixed">
              <TableHeader className="sticky top-0 bg-slate-50/95 backdrop-blur z-10 shadow-sm">
                {table.getHeaderGroups().map(headerGroup => (
                  <TableRow key={headerGroup.id}>
                    <SortableContext
                      items={columnOrder}
                      strategy={horizontalListSortingStrategy}
                    >
                      {headerGroup.headers.map(header => (
                        <DraggableTableHeader key={header.id} header={header} />
                      ))}
                    </SortableContext>
                  </TableRow>
                ))}
              </TableHeader>
            <TableBody>
              {rowVirtualizer.getVirtualItems().length > 0 && (
                <>
                  <TableRow className="border-0 hover:bg-transparent">
                    <TableCell 
                      colSpan={columns.length} 
                      className="p-0 border-0" 
                      style={{ height: `${rowVirtualizer.getVirtualItems()[0].start}px` }} 
                    />
                  </TableRow>
                  {rowVirtualizer.getVirtualItems().map(virtualRow => {
                    const row = rows[virtualRow.index];
                    return (
                      <TableRow 
                        key={row.id}
                        ref={rowVirtualizer.measureElement}
                        data-index={virtualRow.index}
                        className="hover:bg-slate-50/80 transition-colors cursor-pointer group"
                        onClick={() => setSelectedRow(row.original)}
                      >
                        {row.getVisibleCells().map(cell => (
                          <TableCell 
                            key={cell.id} 
                            className="px-4 py-2 align-middle border-b border-slate-100"
                            style={{ width: cell.column.getSize() }}
                          >
                            {flexRender(cell.column.columnDef.cell, cell.getContext())}
                          </TableCell>
                        ))}
                      </TableRow>
                    );
                  })}
                  <TableRow className="border-0 hover:bg-transparent">
                    <TableCell 
                      colSpan={columns.length} 
                      className="p-0 border-0" 
                      style={{ height: `${rowVirtualizer.getTotalSize() - rowVirtualizer.getVirtualItems()[rowVirtualizer.getVirtualItems().length - 1].end}px` }} 
                    />
                  </TableRow>
                </>
              )}
            </TableBody>
          </Table>
        </div>
        </DndContext>
      </div>

      <Sheet open={!!selectedRow} onOpenChange={(open) => !open && setSelectedRow(null)}>
        <SheetContent 
          className="w-[800px] sm:max-w-3xl flex flex-col gap-0 p-0 border-l border-slate-200" 
          side="right"
          onKeyDown={(e) => {
            if (e.key === 'ArrowLeft') {
              if (selectedIndex > 0) {
                setSelectedRow(rows[selectedIndex - 1].original);
              }
            } else if (e.key === 'ArrowRight') {
              if (selectedIndex >= 0 && selectedIndex < rows.length - 1) {
                setSelectedRow(rows[selectedIndex + 1].original);
              }
            }
          }}
        >
          <div className="p-6 border-b border-slate-100 bg-slate-50/50 shrink-0 outline-none" tabIndex={0} autoFocus>
            <SheetHeader>
              <div className="flex items-center justify-between">
                <SheetTitle className="flex items-center gap-2">
                  Sample <span className="font-mono text-slate-500 text-sm">#{selectedRow?.sample_id}</span>
                </SheetTitle>
                {selectedIndex >= 0 && (
                  <Badge variant="outline" className="text-slate-500 font-mono shadow-sm">
                    {selectedIndex + 1} / {rows.length}
                  </Badge>
                )}
              </div>
              <SheetDescription>
                Detailed metadata and label breakdown.
              </SheetDescription>
            </SheetHeader>
          </div>
          
          <div className="flex-1 overflow-y-auto min-h-0">
            {selectedRow && (
              <div className="flex flex-col gap-8 p-6 pb-10">
                {selectedRow.llm_assigned && (
                  <div className="flex flex-col gap-4 p-5 bg-slate-50 rounded-lg border border-slate-100">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold text-sm text-slate-900 uppercase tracking-wider flex items-center gap-2">
                        {selectedRow._human_patch ? (
                          <><Badge className="bg-indigo-600 hover:bg-indigo-700">Human Checked</Badge> Labeling</>
                        ) : (
                          "AI Labeling"
                        )}
                      </h3>
                      {!isEditing && (
                        <div className="flex items-center gap-3">
                          <button 
                            onClick={toggleApproval} 
                            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-md border transition-colors shadow-sm ${
                              selectedRow._human_patch?.human_approved 
                                ? "bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100" 
                                : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50"
                            }`}
                          >
                            <CheckCircle className="w-3.5 h-3.5" />
                            {selectedRow._human_patch?.human_approved ? "Approved" : "Approve"}
                          </button>
                          <button onClick={() => setIsEditing(true)} className="text-xs font-semibold text-indigo-600 hover:text-indigo-800">
                            Edit Labels
                          </button>
                          {selectedRow._human_patch && (
                            <button onClick={handleReset} className="text-xs font-semibold text-rose-500 hover:text-rose-700 flex items-center gap-1 ml-2 border-l border-slate-200 pl-4">
                              <RotateCcw className="w-3.5 h-3.5" />
                              Reset
                            </button>
                          )}
                        </div>
                      )}
                    </div>

                    {isEditing ? (
                      <div className="flex flex-col gap-4">
                        <div>
                          <label className="text-xs text-slate-500 mb-1 block">Primary Label</label>
                          <select 
                            value={editLabel} 
                            onChange={e => setEditLabel(e.target.value)} 
                            className="w-full flex h-9 rounded-md border border-slate-200 bg-white px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-indigo-500" 
                          >
                            <option value="">Select a label...</option>
                            {taxonomy.primary_labels?.map((l: string) => (
                              <option key={l} value={l}>{l}</option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="text-xs text-slate-500 mb-1 block">Secondary Labels</label>
                          <div className="max-h-64 overflow-y-auto border border-slate-200 rounded-md p-2 bg-white grid grid-cols-2 gap-x-2 gap-y-1 shadow-sm">
                            {taxonomy.secondary_labels?.map((t: string) => {
                               const isSelected = editSecondaryLabels.split(',').map(s => s.trim()).includes(t);
                               return (
                                 <label key={t} className="flex items-center gap-2 text-sm text-slate-700 cursor-pointer p-1.5 hover:bg-slate-50 rounded">
                                   <input 
                                     type="checkbox" 
                                     checked={isSelected}
                                     onChange={(e) => {
                                        const tags = editSecondaryLabels.split(',').map(s => s.trim()).filter(Boolean);
                                        if (e.target.checked) setEditSecondaryLabels([...tags, t].join(', '));
                                        else setEditSecondaryLabels(tags.filter(x => x !== t).join(', '));
                                     }}
                                     className="rounded border-slate-300 text-sky-600 focus:ring-sky-600 w-4 h-4"
                                   />
                                   {t}
                                 </label>
                               )
                            })}
                          </div>
                        </div>
                        <div>
                          <label className="text-xs text-slate-500 mb-1 block">Tags</label>
                          <div className="max-h-64 overflow-y-auto border border-slate-200 rounded-md p-2 bg-white grid grid-cols-2 gap-x-2 gap-y-1 shadow-sm">
                            {taxonomy.tags?.map((t: string) => {
                               const isSelected = editTags.split(',').map(s => s.trim()).includes(t);
                               return (
                                 <label key={t} className="flex items-center gap-2 text-sm text-slate-700 cursor-pointer p-1.5 hover:bg-slate-50 rounded">
                                   <input 
                                     type="checkbox" 
                                     checked={isSelected}
                                     onChange={(e) => {
                                        const tags = editTags.split(',').map(s => s.trim()).filter(Boolean);
                                        if (e.target.checked) setEditTags([...tags, t].join(', '));
                                        else setEditTags(tags.filter(x => x !== t).join(', '));
                                     }}
                                     className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-600 w-4 h-4"
                                   />
                                   {t}
                                 </label>
                               )
                            })}
                          </div>
                        </div>
                        <div>
                          <label className="text-xs text-slate-500 mb-1 block">Rationale</label>
                          <textarea 
                            value={editRationale} 
                            onChange={e => setEditRationale(e.target.value)} 
                            className="w-full flex min-h-[80px] rounded-md border border-slate-200 bg-white px-3 py-2 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-indigo-500" 
                          />
                        </div>
                        <div className="flex items-center gap-2 mt-2">
                          <button onClick={handleSaveEdit} className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold rounded-md shadow-sm">Save Edits</button>
                          <button onClick={() => setIsEditing(false)} className="px-4 py-2 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-sm font-semibold rounded-md shadow-sm">Cancel</button>
                        </div>
                      </div>
                    ) : (
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <div className="text-xs text-slate-500 mb-1">Primary Label</div>
                          <TwoToneBadge label={selectedRow.llm_assigned.primary_label} />
                        </div>
                        <div>
                          <div className="text-xs text-slate-500 mb-1">Confidence</div>
                          <ConfidenceBadge val={selectedRow.llm_assigned.label_confidence} />
                        </div>
                        <div className="col-span-2">
                          <div className="text-xs text-slate-500 mb-1">Rationale</div>
                          <div className="text-sm text-slate-700">{selectedRow.llm_assigned.label_rationale}</div>
                        </div>
                        <div className="col-span-2">
                          <div className="text-xs text-slate-500 mb-1">Tags</div>
                          <div className="flex flex-wrap gap-2">
                            {(selectedRow.llm_assigned.tags || []).map((t: string, i: number) => (
                              <TwoToneBadge key={i} label={t} variant="tag" />
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
                
                <div className="flex flex-col gap-2">
                  <h3 className="font-semibold text-sm text-slate-900 uppercase tracking-wider">Input Prompt</h3>
                  <div className="bg-slate-900 text-slate-50 rounded-lg p-4 text-xs font-mono overflow-x-auto whitespace-pre-wrap leading-relaxed shadow-inner">
                    {typeof selectedRow.input === 'object' && selectedRow.input !== null 
                      ? JSON.stringify(selectedRow.input, null, 2) 
                      : String(selectedRow.input || '')}
                  </div>
                </div>

                {selectedRow.metadata && (
                  <div className="flex flex-col gap-2">
                    <h3 className="font-semibold text-sm text-slate-900 uppercase tracking-wider">Metadata</h3>
                    <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                      <pre className="text-xs font-mono overflow-x-auto whitespace-pre-wrap text-slate-700">
                        {JSON.stringify(selectedRow.metadata, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </SheetContent>
      </Sheet>
    </div>
  );
}
