'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { GroundingLink, GroundingRef } from '@/lib/data';
import { EYEBROW_CLASS } from '@/components/layout/Eyebrow';
import { FRAMEWORK_LABEL } from '@/components/CoverageOverview';
import { INDEX_TONE, type IndexId } from '@/lib/scores';

export interface ExplorerIndex {
  id: string;
  label: string;
}

export interface ExplorerSubcategory {
  key: string;
  slug: string;
  name: string;
  description: string;
  index: string;
  fullItems: number;
  coreItems: number;
  gap: string | null;
  // Ordered by number of contributing tasks, largest first.
  benchmarks: string[];
  lineage: string[];
  links: GroundingLink[];
}

function gapNote(gap: string, coreItems: number, indexLabel: string) {
  if (coreItems === 0) return 'No CORE tasks fall in this area yet, so it has no separate score.';
  const needed = gap.match(/needs (\d+)/)?.[1];
  return `CORE runs only ${coreItems} of these tasks, too few for a separate score${needed ? ` (at least ${needed} are needed)` : ''}. They still count toward the ${indexLabel} index.`;
}

const SECTION_TITLE = 'text-[11px] font-semibold uppercase tracking-wider text-muted-foreground';

function RefCard({ item: r, link }: { item: GroundingRef; link: GroundingLink }) {
  const direct = link.strength === 'direct' && link.coverage !== 'none';
  return (
    <li
      className={`flex flex-col rounded-xl border p-4 ${
        direct ? 'border-border border-l-4 border-l-ink/70 bg-white' : 'border-dashed border-ink/20 bg-muted/30'
      }`}
    >
      <div className="text-xs text-muted-foreground">
        <span className="font-semibold text-ink/80">{FRAMEWORK_LABEL[r.framework]}</span> · {r.citation}
      </div>
      <div className="mt-1.5 font-semibold leading-snug text-ink">{r.title}</div>
      <p className="mt-1.5 line-clamp-3 text-sm leading-relaxed text-ink/70" title={r.excerpt}>“{r.excerpt}”</p>
      {link.coverage === 'none' && (
        <p className="mt-2 text-xs italic text-muted-foreground">CORE’s current benchmarks don’t test this part yet.</p>
      )}
      <a
        href={r.url}
        target="_blank"
        rel="noopener noreferrer"
        className="mt-auto pt-3 text-xs font-semibold text-ink/80 underline decoration-ink/25 underline-offset-2 hover:decoration-ink"
      >
        Read the full text ↗
      </a>
    </li>
  );
}

export function MeasurementExplorer({
  indices, subcategories, refs, paperUrl,
}: {
  indices: ExplorerIndex[];
  subcategories: ExplorerSubcategory[];
  refs: Record<string, GroundingRef>;
  paperUrl: string;
}) {
  const [selected, setSelected] = useState(subcategories[0].key);
  const rootRef = useRef<HTMLDivElement>(null);
  const detailRef = useRef<HTMLElement>(null);

  const reveal = useCallback(() => {
    // On narrow screens the detail sits below the list; on wide screens it sits beside it.
    const target = window.matchMedia('(min-width: 1024px)').matches ? rootRef.current : detailRef.current;
    if (target && target.getBoundingClientRect().top < 0) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    else if (target && !window.matchMedia('(min-width: 1024px)').matches) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, []);

  useEffect(() => {
    // Arriving from elsewhere on the page: show the list beside the detail when there is room, else the detail itself.
    const jump = (behavior: ScrollBehavior) => {
      const target = window.matchMedia('(min-width: 1024px)').matches ? rootRef.current : detailRef.current;
      target?.scrollIntoView({ behavior, block: 'start' });
    };
    // An area slug selects that area; `#areas-<index>` selects the index's first scored area.
    const resolve = (hash: string) => {
      const index = hash.match(/^#areas-(.+)$/)?.[1];
      if (index) {
        const inIndex = subcategories.filter(s => s.index === index);
        return inIndex.find(s => !s.gap && s.coreItems > 0) ?? inIndex[0];
      }
      return subcategories.find(s => `#${s.slug}` === hash);
    };
    const show = (match: ExplorerSubcategory | undefined, behavior: ScrollBehavior) => {
      if (!match) return false;
      setSelected(match.key);
      jump(behavior);
      return true;
    };
    show(resolve(window.location.hash), 'auto');
    const onHash = () => show(resolve(window.location.hash), 'smooth');
    // Links elsewhere on the page (index cards, coverage, catalog) select an area even when the hash is unchanged.
    const onClick = (e: MouseEvent) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const anchor = (e.target as Element | null)?.closest?.('a[href^="#"]');
      const match = anchor ? resolve(anchor.getAttribute('href') ?? '') : undefined;
      if (!show(match, 'smooth')) return;
      e.preventDefault();
      window.history.replaceState(null, '', `#${match!.slug}`);
    };
    window.addEventListener('hashchange', onHash);
    document.addEventListener('click', onClick);
    return () => {
      window.removeEventListener('hashchange', onHash);
      document.removeEventListener('click', onClick);
    };
  }, [subcategories]);

  const select = (s: ExplorerSubcategory) => {
    setSelected(s.key);
    window.history.replaceState(null, '', `#${s.slug}`);
    reveal();
  };

  const current = subcategories.find(s => s.key === selected)!;
  const indexLabel = indices.find(i => i.id === current.index)?.label ?? current.index;
  const direct = current.links.filter(l => l.strength === 'direct' && l.coverage !== 'none');
  const related = current.links.filter(l => !(l.strength === 'direct' && l.coverage !== 'none'));
  const tone = INDEX_TONE[current.index as IndexId];

  return (
    <div ref={rootRef} className="scroll-mt-24 grid grid-cols-1 gap-6 lg:grid-cols-[300px_minmax(0,1fr)] lg:gap-8">
      <nav aria-label="Evaluation areas" className="lg:sticky lg:top-24 lg:self-start space-y-5">
        {indices.map(index => {
          const items = subcategories.filter(s => s.index === index.id);
          return (
            <div key={index.id}>
              <div className="flex items-baseline justify-between border-b border-border pb-1.5">
                <h3 className="text-sm font-extrabold uppercase tracking-wider text-ink">{index.label}</h3>
                <span className="hidden text-[11px] font-semibold uppercase tracking-wider text-muted-foreground lg:inline">CORE tasks</span>
              </div>
              <ul className="mt-2 flex flex-wrap gap-1.5 lg:mt-1 lg:block lg:space-y-0.5">
                {items.map(s => {
                  const active = s.key === selected;
                  const itemTone = INDEX_TONE[s.index as IndexId];
                  return (
                    <li key={s.key}>
                      <button
                        type="button"
                        onClick={() => select(s)}
                        aria-pressed={active}
                        className={`cursor-pointer rounded-full border px-3 py-1.5 text-left text-sm transition-colors lg:flex lg:w-full lg:items-center lg:gap-2 lg:rounded-none lg:rounded-r-lg lg:border-0 lg:border-l-2 lg:py-2 lg:text-[15px] ${
                          active
                            ? `${itemTone.line} ${itemTone.soft} font-semibold text-ink`
                            : 'border-border text-ink/80 hover:bg-muted/60 hover:text-ink lg:border-transparent'
                        }`}
                      >
                        <span className={active ? '' : 'font-medium'}>{s.name}</span>
                        {s.gap && (
                          <span className="ml-1.5 rounded-full border border-border bg-white px-1.5 py-0.5 text-[10px] font-semibold text-muted-foreground lg:ml-0">
                            Not scored
                          </span>
                        )}
                        <span className="hidden tabular text-xs font-semibold text-ink/60 lg:ml-auto lg:inline">
                          {s.coreItems.toLocaleString()}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          );
        })}
      </nav>

      <article ref={detailRef} className="scroll-mt-24 rounded-2xl border border-border bg-white p-5 md:p-8" aria-live="polite">
        <p className={`${EYEBROW_CLASS} mb-1.5`}>{indexLabel} index</p>
        <h3 className="text-2xl md:text-3xl font-extrabold tracking-tight text-ink">{current.name}</h3>
        <p className="mt-2 max-w-2xl text-base leading-relaxed text-ink/80">{current.description}</p>

        <dl className="mt-6 grid max-w-xl grid-cols-3 gap-3 text-center">
          <div className="rounded-xl bg-muted/60 px-2 py-2.5">
            <dt className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Benchmarks</dt>
            <dd className="text-xl font-bold tabular text-ink">{current.benchmarks.length}</dd>
          </div>
          <div className="rounded-xl bg-muted/60 px-2 py-2.5">
            <dt className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Full-suite tasks</dt>
            <dd className="text-xl font-bold tabular text-ink">{current.fullItems.toLocaleString()}</dd>
          </div>
          <div className={`rounded-xl px-2 py-2.5 ${tone.soft}`}>
            <dt className="text-[11px] font-semibold uppercase tracking-wider text-ink">CORE tasks</dt>
            <dd className="text-xl font-bold tabular text-ink">{current.coreItems.toLocaleString()}</dd>
          </div>
        </dl>
        {current.gap && (
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted-foreground">
            <span className="mr-1.5 rounded-full border border-border bg-muted px-1.5 py-0.5 text-[10px] font-semibold text-muted-foreground">
              Not scored
            </span>
            {gapNote(current.gap, current.coreItems, indexLabel)}
          </p>
        )}

        <div className="mt-8 grid grid-cols-1 gap-8 md:grid-cols-2">
          <section>
            <h4 className={SECTION_TITLE}>Source benchmarks</h4>
            <ul className="mt-2 flex flex-wrap gap-1.5">
              {current.benchmarks.map(b => (
                <li key={b}>
                  <a
                    href={`#benchmark-${b}`}
                    className="block rounded-md border border-border bg-muted/40 px-2 py-0.5 font-mono text-xs text-ink/80 hover:border-ink/40 hover:text-ink"
                  >
                    {b}
                  </a>
                </li>
              ))}
            </ul>
          </section>
          <section>
            <h4 className={SECTION_TITLE}>Original COMPL-AI requirement</h4>
            <ul className="mt-2 space-y-1">
              {current.lineage.map(l => (
                <li key={l} className="text-sm font-semibold text-ink">{l}</li>
              ))}
            </ul>
            <a href={paperUrl} target="_blank" rel="noopener noreferrer" className="mt-2 inline-block text-xs font-semibold text-ink/80 underline decoration-ink/25 underline-offset-2 hover:decoration-ink">
              Read the 2024 COMPL-AI paper ↗
            </a>
          </section>
        </div>

        <section className="mt-8">
          <h4 className={SECTION_TITLE}>Regulatory references</h4>
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
            EU AI Act provisions and Code of Practice measures that relate to what this area tests. Relevance is not compliance.
          </p>
          {direct.length > 0 && (
            <>
              <p className="mt-5 text-sm text-ink/80"><span className="font-semibold text-ink">Directly relevant.</span> The reference names the behaviour this area tests.</p>
              <ul className="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-2">
                {direct.map(l => <RefCard key={l.id} item={refs[l.id]} link={l} />)}
              </ul>
            </>
          )}
          {related.length > 0 && (
            <>
              <p className="mt-6 text-sm text-ink/80"><span className="font-semibold text-ink">Related.</span> The results are relevant evidence but don’t test the reference directly.</p>
              <ul className="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-2">
                {related.map(l => <RefCard key={l.id} item={refs[l.id]} link={l} />)}
              </ul>
            </>
          )}
        </section>
      </article>
    </div>
  );
}
