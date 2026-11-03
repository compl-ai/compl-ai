'use client';
import { useState } from 'react';
import { Input } from '@/components/ui/input';
import { INDICES, formatTokens } from '@/lib/scores';

export interface BenchmarkArea {
  key: string;
  slug: string;
  name: string;
}

export interface BenchmarkRow {
  name: string;
  index: string;
  // Evaluation areas its tasks are labelled with, most tasks first.
  areas: BenchmarkArea[];
  modalities: string[];
  populationItems: number;
  coreItems: number | null;
  tokensPerItem: number | null;
}

const TAG_LABELS: Record<string, string> = {
  'modality:static-mcq': 'Multiple choice',
  'modality:static-generation': 'Generation',
  'modality:multimodal': 'Multimodal',
  'modality:coding': 'Code',
  'agent:untrusted-context': 'Untrusted context',
  'agent:multi-step': 'Multi-step agent',
};

const indexLabel = (id: string) => INDICES.find(i => i.id === id)?.label ?? id;

export function BenchmarkList({ benchmarks }: { benchmarks: BenchmarkRow[] }) {
  const [search, setSearch] = useState('');
  const query = search.trim().toLowerCase();
  const filtered = benchmarks.filter(b =>
    !query || [b.name, indexLabel(b.index), ...b.areas.map(a => a.name)].some(s => s.toLowerCase().includes(query))
  );

  return (
    <div className="space-y-4">
      <Input
        type="search"
        placeholder="Search benchmarks or evaluation areas…"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="max-w-sm bg-white"
      />
      <div className="rounded-2xl border border-border bg-white overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              <th className="px-4 py-3 text-left">Benchmark</th>
              <th className="px-3 py-3 text-left">Index</th>
              <th className="px-3 py-3 text-left">Evaluation areas</th>
              <th className="px-3 py-3 text-left">Format</th>
              <th className="px-3 py-3 text-right whitespace-nowrap">Full-suite tasks</th>
              <th className="px-3 py-3 text-right whitespace-nowrap">CORE tasks</th>
              <th className="px-4 py-3 text-right" title="Typical tokens per item for the evaluated model: average input plus median output across models. Judge models are not included.">Tokens / item</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(b => (
              <tr key={b.name} id={`benchmark-${b.name}`} className="scroll-mt-24 border-b border-border last:border-0 align-top transition-colors target:bg-muted/60">
                <td className="px-4 py-3 font-semibold text-ink whitespace-nowrap">{b.name}</td>
                <td className="px-3 py-3 text-ink/80">{indexLabel(b.index)}</td>
                <td className="px-3 py-3">
                  <div className="flex flex-wrap gap-1">
                    {b.areas.map(a => (
                      <a
                        key={a.key}
                        href={`#${a.slug}`}
                        className="rounded-full border border-border bg-white px-2 py-0.5 text-[11px] font-semibold text-ink/80 whitespace-nowrap hover:border-ink/40 hover:text-ink"
                      >
                        {a.name}
                      </a>
                    ))}
                  </div>
                </td>
                <td className="px-3 py-3">
                  <div className="flex flex-wrap gap-1">
                    {b.modalities.map(m => (
                      <span key={m} className="rounded-full border border-border bg-muted px-2 py-0.5 text-[11px] font-medium text-muted-foreground whitespace-nowrap">{TAG_LABELS[m] ?? m}</span>
                    ))}
                  </div>
                </td>
                <td className="px-3 py-3 text-right tabular text-xs text-muted-foreground">{b.populationItems.toLocaleString()}</td>
                <td className="px-3 py-3 text-right tabular text-xs font-semibold text-ink">{b.coreItems?.toLocaleString() ?? '—'}</td>
                <td className="px-4 py-3 text-right tabular text-xs text-muted-foreground">{b.tokensPerItem ? formatTokens(Math.round(b.tokensPerItem)) : '—'}</td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-muted-foreground">No benchmarks match “{search}”.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
