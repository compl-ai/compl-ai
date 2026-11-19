'use client';
import { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import type { CostOverview, ExperimentsData, JoinedModelData } from '@/lib/data';
import { QUESTIONS, type QuestionId, buildFindings, contextOf, lookupModels } from './explorer';
import { VIEWS } from './explorerViews';

// Space between the floating question card and the first data on wide screens.
const CARD_GAP = 28;

export function ResultsExplorer({ models, explorer, costs }: { models: JoinedModelData[]; explorer: ExperimentsData; costs: CostOverview }) {
  const [question, setQuestion] = useState<QuestionId>(QUESTIONS[0].id);
  const findings = useMemo(() => buildFindings(explorer, costs, models), [explorer, costs, models]);
  const byId = useMemo(() => lookupModels(models), [models]);

  const columnRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  // Where the data area starts, so it clears the question card that floats over the chart column on wide screens.
  const [inset, setInset] = useState(0);
  useEffect(() => {
    const col = columnRef.current;
    const card = cardRef.current;
    if (!col || !card) return;
    const wide = window.matchMedia('(min-width: 1280px)');
    const measure = () => setInset(wide.matches ? Math.max(0, Math.round(card.getBoundingClientRect().right - col.getBoundingClientRect().left + CARD_GAP)) : 0);
    const ro = new ResizeObserver(measure);
    ro.observe(col);
    ro.observe(card);
    wide.addEventListener('change', measure);
    return () => { ro.disconnect(); wide.removeEventListener('change', measure); };
  }, []);

  const View = VIEWS[question];
  const current = QUESTIONS.find((q) => q.id === question)!;

  return (
    <div className="grid gap-6 rounded-panel border border-border bg-card p-5 md:p-8 xl:relative xl:left-1/2 xl:w-[max(calc(100%+48px),80vw)] xl:-translate-x-1/2 xl:gap-0 xl:min-h-[760px]">
      <div
        ref={cardRef}
        className="relative z-10 hidden self-start overflow-hidden rounded-2xl xl:block border border-border bg-card/90 shadow-[0_8px_30px_-12px_rgba(5,28,44,0.18)] backdrop-blur-sm xl:[grid-area:1/1] xl:w-[390px]"
      >
        <ol className="divide-y divide-border">
          {QUESTIONS.map((q, i) => {
            const open = q.id === question;
            return (
              <li key={q.id} className={open ? 'bg-brand-soft/50' : undefined}>
                <button
                  onClick={() => setQuestion(q.id)}
                  aria-expanded={open}
                  aria-controls="results-explorer-chart"
                  className={`group flex w-full cursor-pointer items-baseline gap-3 px-5 py-3.5 text-left transition-colors ${open ? '' : 'hover:bg-muted/50'}`}
                >
                  <span className={`w-5 shrink-0 font-plex text-xs font-semibold tabular ${open ? 'text-brand-strong' : 'text-muted-foreground'}`}>
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <span className={`flex-1 font-serif text-[16px] leading-snug ${open ? 'font-bold text-ink' : 'font-semibold text-ink/80 group-hover:text-ink'}`}>
                    {q.title}
                  </span>
                </button>
                {open && (
                  <div className="px-5 pb-5 pl-[52px] text-sm leading-relaxed">
                    <p className="text-muted-foreground">{contextOf(q, explorer)}</p>
                    <div className="mt-3 border-l-2 border-brand pl-3">
                      <p className="font-plex text-[11px] font-semibold uppercase tracking-[0.1em] text-brand-strong">Finding</p>
                      <p className="mt-1 text-ink">{findings[q.id]}</p>
                    </div>
                  </div>
                )}
              </li>
            );
          })}
        </ol>
      </div>

      <div className="xl:hidden">
        <label htmlFor="results-explorer-question" className="font-plex text-[11px] font-semibold uppercase tracking-[0.1em] text-muted-foreground">
          Question {QUESTIONS.findIndex((q) => q.id === question) + 1} of {QUESTIONS.length}
        </label>
        {/* The native select stays on top (transparent) so phones open their own picker; the label underneath can wrap. */}
        <div className="relative mt-1.5 rounded-xl border border-border bg-brand-soft/50 py-3 pl-4 pr-10 has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-brand">
          <span aria-hidden className="block font-serif text-[14px] font-bold leading-snug text-ink">{current.title}</span>
          <ChevronDown aria-hidden className="pointer-events-none absolute right-3 top-1/2 size-5 -translate-y-1/2 text-ink" />
          <select
            id="results-explorer-question"
            value={question}
            onChange={(ev) => setQuestion(ev.target.value as QuestionId)}
            aria-controls="results-explorer-chart"
            className="absolute inset-0 size-full cursor-pointer appearance-none opacity-0 focus:outline-none"
          >
            {QUESTIONS.map((q) => (
              <option key={q.id} value={q.id}>{q.title}</option>
            ))}
          </select>
        </div>
        <div className="mt-4 text-sm leading-relaxed">
          <p className="text-muted-foreground">{contextOf(current, explorer)}</p>
          <div className="mt-3 border-l-2 border-brand pl-3">
            <p className="font-plex text-[11px] font-semibold uppercase tracking-[0.1em] text-brand-strong">Finding</p>
            <p className="mt-1 text-ink">{findings[question]}</p>
          </div>
        </div>
      </div>

      <div ref={columnRef} id="results-explorer-chart" className="font-plex min-w-0 xl:flex xl:flex-col xl:[grid-area:1/1]">
        <View key={question} e={explorer} costs={costs} models={models} byId={byId} inset={inset} />
      </div>
    </div>
  );
}
