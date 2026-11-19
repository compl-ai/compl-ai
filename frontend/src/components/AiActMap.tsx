'use client';
import { useEffect, useMemo, useState } from 'react';
import { cn } from 'cn';
import type { GroundingRef } from '@/lib/data';
import { FRAMEWORK_LABEL, FRAMEWORKS, citeLabel } from '@/lib/frameworks';
import { INDEX_TONE, INDICES, type IndexId } from '@/lib/scores';
import { useIndexFilter, type IndexFilter } from '@/components/IndexStrip';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card';

export interface MapArea {
  key: string;
  slug: string;
  name: string;
  index: IndexId;
  description: string;
  fullItems: number;
  coreItems: number;
  gap: string | null;
  // Benchmarks its questions come from, most questions first.
  benchmarks: { label: string; href: string }[];
}

// direct: the provision names the behaviour the area tests. Otherwise the area informs a related risk.
export interface MapLink {
  area: string;
  ref: string;
  direct: boolean;
}

export interface WholeEvaluationRef {
  id: string;
  scopes: ('all' | IndexId)[];
}

type Framework = GroundingRef['framework'];
type FrameworkChoice = Framework | 'both';
type Focus = { kind: 'area'; key: string } | { kind: 'ref'; id: string };

// Diagram geometry in px. Areas share the column height equally; provisions get one row each.
const HEAD = 30;
const GAP = 18;
const ROW = 26;
const ROW_MAX = 44;
const AREA_MIN = 42;
const AREA_MAX = 88;

const INDEX_LABEL = Object.fromEntries(INDICES.map(i => [i.id, i.label])) as Record<IndexId, string>;
const sameFocus = (a: Focus | null, b: Focus | null) =>
  !!a && !!b && a.kind === b.kind && (a.kind === 'area' ? a.key === (b as typeof a).key : a.id === (b as typeof a).id);
const focusHash = (f: Focus, areas: MapArea[]) =>
  f.kind === 'area' ? areas.find(a => a.key === f.key)!.slug : `ref-${f.id}`;

function LegendLine({ dashed = false }: { dashed?: boolean }) {
  return (
    <svg width="22" height="4" aria-hidden className="text-ink/60">
      <line x1="0" y1="2" x2="22" y2="2" stroke="currentColor" strokeWidth={dashed ? 1 : 2} strokeDasharray={dashed ? '5 4' : undefined} />
    </svg>
  );
}

function gapNote(gap: string, coreItems: number, indexLabel: string) {
  if (coreItems === 0) return 'No CORE questions fall in this area yet, so it has no separate score.';
  const needed = gap.match(/needs (\d+)/)?.[1];
  return `CORE includes only ${coreItems} questions from this area.${needed ? ` We only score an area separately when CORE includes at least ${needed} of its questions.` : ' That is too few for a separate score.'} These ${coreItems} still count toward the ${indexLabel} index.`;
}

interface Group {
  id: string;
  label: string;
  items: string[];
}
interface Node {
  id: string;
  top: number;
  height: number;
}
interface Layout {
  height: number;
  leftHeads: { id: IndexId; top: number }[];
  rightHeads: { id: Framework; top: number; count: number }[];
  areas: Node[];
  refs: Node[];
  paths: (MapLink & { y1: number; y2: number })[];
}

// Orders both columns to reduce crossings (a few rounds of barycentre sorting within each group), then places
// the nodes so both columns span the same height, and spreads each node's link ends in the order of their other end.
function computeLayout(areas: MapArea[], refs: Record<string, GroundingRef>, links: MapLink[]): Layout {
  const refsOf: Record<string, string[]> = {};
  const areasOf: Record<string, string[]> = {};
  for (const l of links) {
    (refsOf[l.area] ??= []).push(l.ref);
    (areasOf[l.ref] ??= []).push(l.area);
  }
  const fullItems = Object.fromEntries(areas.map(a => [a.key, a.fullItems]));
  let left: Group[] = INDICES.map(i => ({ id: i.id, label: i.label, items: areas.filter(a => a.index === i.id).map(a => a.key) })).filter(g => g.items.length);
  const refIds = Object.keys(areasOf);
  let right: Group[] = FRAMEWORKS.map(f => ({ id: f, label: FRAMEWORK_LABEL[f], items: refIds.filter(id => refs[id].framework === f) })).filter(g => g.items.length);

  const positions = (groups: Group[]) => new Map(groups.flatMap(g => g.items).map((id, i) => [id, i]));
  const mean = (xs: number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : Infinity);
  const order = (groups: Group[], other: Group[], neighbours: Record<string, string[]>, tie: (a: string, b: string) => number) => {
    const pos = positions(other);
    const centre = (id: string) => mean((neighbours[id] ?? []).map(n => pos.get(n)!));
    return groups.map(g => ({ ...g, items: [...g.items].sort((a, b) => centre(a) - centre(b) || tie(a, b)) }));
  };
  const byCitation = (a: string, b: string) => refs[a].citation.localeCompare(refs[b].citation, undefined, { numeric: true });
  const bySize = (a: string, b: string) => fullItems[b] - fullItems[a];
  for (let round = 0; round < 4; round++) {
    right = order(right, left, areasOf, byCitation);
    left = order(left, right, refsOf, bySize);
  }
  right = order(right, left, areasOf, byCitation);

  const measure = (groups: Group[]) => {
    const n = groups.reduce((s, g) => s + g.items.length, 0);
    return { n, fixed: groups.length * HEAD + Math.max(0, groups.length - 1) * GAP };
  };
  const lm = measure(left);
  const rm = measure(right);
  const height = Math.max(lm.fixed + lm.n * AREA_MIN, rm.fixed + rm.n * ROW);
  const place = (groups: Group[], m: { n: number; fixed: number }, min: number, max: number) => {
    const size = m.n ? Math.min(max, Math.max(min, (height - m.fixed) / m.n)) : 0;
    let y = (height - (m.fixed + m.n * size)) / 2;
    const heads: { id: string; top: number; count: number }[] = [];
    const nodes: Node[] = [];
    for (const g of groups) {
      heads.push({ id: g.id, top: y, count: g.items.length });
      y += HEAD;
      for (const id of g.items) {
        nodes.push({ id, top: y, height: size });
        y += size;
      }
      y += GAP;
    }
    return { heads, nodes };
  };
  const L = place(left, lm, AREA_MIN, AREA_MAX);
  const R = place(right, rm, ROW, ROW_MAX);

  const areaNode = new Map(L.nodes.map(n => [n.id, n]));
  const refNode = new Map(R.nodes.map(n => [n.id, n]));
  const centre = (n: Node) => n.top + n.height / 2;
  // y of each link end on a node, spread around its centre in the order of the far ends.
  const spread = (node: Node, ends: { key: string; far: number }[], step: number, pad: number) => {
    const sorted = [...ends].sort((a, b) => a.far - b.far);
    const span = Math.min(node.height - pad * 2, (sorted.length - 1) * step);
    return new Map(sorted.map((e, i) => [e.key, centre(node) - span / 2 + (sorted.length > 1 ? (i * span) / (sorted.length - 1) : 0)]));
  };
  const linkKey = (l: MapLink) => `${l.area}|${l.ref}`;
  const y1 = new Map<string, number>();
  const y2 = new Map<string, number>();
  for (const [key, node] of areaNode) {
    const ends = links.filter(l => l.area === key).map(l => ({ key: linkKey(l), far: centre(refNode.get(l.ref)!) }));
    for (const [k, y] of spread(node, ends, 6, 10)) y1.set(k, y);
  }
  for (const [id, node] of refNode) {
    const ends = links.filter(l => l.ref === id).map(l => ({ key: linkKey(l), far: centre(areaNode.get(l.area)!) }));
    for (const [k, y] of spread(node, ends, 3, 6)) y2.set(k, y);
  }

  return {
    height,
    leftHeads: L.heads.map(h => ({ id: h.id as IndexId, top: h.top })),
    rightHeads: R.heads.map(h => ({ id: h.id as Framework, top: h.top, count: h.count })),
    areas: L.nodes,
    refs: R.nodes,
    paths: links.map(l => ({ ...l, y1: y1.get(linkKey(l))!, y2: y2.get(linkKey(l))! })),
  };
}

function RefCardBody({ item: r, note }: { item: GroundingRef; note?: string }) {
  return (
    <>
      <p className="text-[11px] text-muted-foreground">
        <span className="font-semibold text-ink/80">{FRAMEWORK_LABEL[r.framework]}</span> · {r.citation}
      </p>
      <p className="text-sm font-semibold leading-snug text-ink">{r.title}</p>
      <p className="line-clamp-5 text-xs leading-relaxed text-ink/70">“{r.excerpt}”</p>
      {note && <p className="text-xs text-muted-foreground">{note}</p>}
      <a
        href={r.url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-block pt-0.5 text-xs font-semibold text-ink/80 underline decoration-ink/25 underline-offset-2 hover:decoration-ink"
      >
        Read the full text ↗
      </a>
    </>
  );
}

function AreaCardBody({ area: a, refCount }: { area: MapArea; refCount: number }) {
  const shown = a.benchmarks.slice(0, 5);
  return (
    <>
      <p className="text-[11px] text-muted-foreground">
        <span className="font-semibold text-ink/80">{INDEX_LABEL[a.index]} index</span> · {a.gap ? 'Not scored separately' : 'Scored separately'}
      </p>
      <p className="text-sm font-semibold leading-snug text-ink">{a.name}</p>
      <p className="line-clamp-4 text-xs leading-relaxed text-ink/70">{a.description}</p>
      <p className="text-xs tabular text-muted-foreground">
        {a.fullItems.toLocaleString()} questions in the full suite, {a.coreItems.toLocaleString()} in CORE. Linked to{' '}
        {refCount} {refCount === 1 ? 'provision' : 'provisions'}.
      </p>
      {a.gap && <p className="text-xs italic text-muted-foreground">{gapNote(a.gap, a.coreItems, INDEX_LABEL[a.index])}</p>}
      {shown.length > 0 && (
        <p className="text-xs leading-relaxed text-muted-foreground">
          From{' '}
          {shown.map((b, i) => (
            <span key={b.href}>
              {i > 0 && (i === shown.length - 1 && a.benchmarks.length === shown.length ? ' and ' : ', ')}
              <a href={b.href} className="font-semibold text-ink/80 hover:text-link hover:underline">{b.label}</a>
            </span>
          ))}
          {a.benchmarks.length > shown.length && ` and ${a.benchmarks.length - shown.length} more`}.
        </p>
      )}
    </>
  );
}

function FrameworkSwitch({ value, onChange }: { value: FrameworkChoice; onChange: (v: FrameworkChoice) => void }) {
  const options: { id: FrameworkChoice; label: string }[] = [{ id: 'both', label: 'Both' }, ...FRAMEWORKS.map(f => ({ id: f, label: FRAMEWORK_LABEL[f] }))];
  return (
    <div role="group" aria-label="Provisions to show" className="inline-flex rounded-full border border-border bg-card p-0.5">
      {options.map(o => (
        <button
          key={o.id}
          type="button"
          aria-pressed={value === o.id}
          onClick={() => onChange(o.id)}
          className={cn(
            'cursor-pointer rounded-full px-3 py-1 text-xs font-semibold transition-colors',
            value === o.id ? 'bg-ink text-background' : 'text-muted-foreground hover:text-ink',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function RefChip({ item: r, note }: { item: GroundingRef; note?: string }) {
  return (
    <HoverCard>
      <HoverCardTrigger
        href={r.url}
        target="_blank"
        rel="noopener noreferrer"
        delay={300}
        closeDelay={150}
        className="flex max-w-full items-baseline gap-1.5 rounded-full border border-border bg-card px-2.5 py-1 text-xs hover:border-ink/30"
      >
        <span className="shrink-0 font-semibold tabular text-ink">{citeLabel(r)}</span>
        <span className="max-w-[16rem] truncate text-ink/70">{r.title}</span>
      </HoverCardTrigger>
      <HoverCardContent side="top" className="w-80 space-y-1.5 p-3">
        <RefCardBody item={r} note={note} />
      </HoverCardContent>
    </HoverCard>
  );
}

function ChipRow({ title, description, children }: { title: string; description: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-x-8 gap-y-2 md:grid-cols-[minmax(12rem,16rem)_1fr]">
      <div>
        <h3 className="text-sm font-semibold text-ink">{title}</h3>
        <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">{description}</p>
      </div>
      <div className="flex flex-wrap content-start gap-1.5">{children}</div>
    </div>
  );
}

export function AiActMap({
  areas,
  refs,
  links,
  wholeEvaluation,
  untested,
}: {
  areas: MapArea[];
  refs: Record<string, GroundingRef>;
  links: MapLink[];
  wholeEvaluation: WholeEvaluationRef[];
  untested: string[];
}) {
  const { filter, setFilter } = useIndexFilter();
  const [framework, setFramework] = useState<FrameworkChoice>('both');
  const [hover, setHover] = useState<Focus | null>(null);
  const [card, setCard] = useState<Focus | null>(null);
  const [pinned, setPinned] = useState<Focus | null>(null);

  const visibleAreas = useMemo(() => areas.filter(a => filter === 'all' || a.index === filter), [areas, filter]);
  const visibleLinks = useMemo(() => {
    const keys = new Set(visibleAreas.map(a => a.key));
    return links.filter(l => keys.has(l.area) && (framework === 'both' || refs[l.ref].framework === framework));
  }, [links, refs, visibleAreas, framework]);
  const layout = useMemo(() => computeLayout(visibleAreas, refs, visibleLinks), [visibleAreas, refs, visibleLinks]);

  // A `#<area>` or `#ref-<id>` hash pins that node, switching the index or framework if it is hidden.
  useEffect(() => {
    const read = () => {
      const id = decodeURIComponent(window.location.hash.slice(1));
      const area = areas.find(a => a.slug === id);
      const refId = id.startsWith('ref-') && links.some(l => l.ref === id.slice(4)) ? id.slice(4) : null;
      if (!area && !refId) return;
      if (area) {
        setPinned({ kind: 'area', key: area.key });
        const current = document.querySelector('[data-index-filter]')?.getAttribute('data-index-filter') as IndexFilter | null;
        if (current && current !== 'all' && current !== area.index) setFilter(area.index, false);
      } else {
        setPinned({ kind: 'ref', id: refId! });
        setFramework(f => (f === 'both' || f === refs[refId!].framework ? f : 'both'));
      }
      const selector = area ? `[data-area="${area.slug}"]` : `[data-ref="${refId}"]`;
      setTimeout(() => {
        const target = [...document.querySelectorAll<HTMLElement>(selector)].find(el => el.offsetParent);
        target?.scrollIntoView({ block: 'center' });
      }, 60);
    };
    read();
    window.addEventListener('hashchange', read);
    return () => window.removeEventListener('hashchange', read);
  }, [areas, links, refs, setFilter]);

  // A pin hidden by the index or framework filter no longer applies.
  const pin =
    pinned && (pinned.kind === 'area' ? layout.areas.some(n => n.id === pinned.key) : layout.refs.some(n => n.id === pinned.id))
      ? pinned
      : null;
  const focus = hover ?? card ?? pin;
  const touches = (l: MapLink) => !!focus && (focus.kind === 'area' ? l.area === focus.key : l.ref === focus.id);
  const lit = new Set<string>();
  if (focus) {
    for (const l of visibleLinks) {
      if (!touches(l)) continue;
      lit.add(`a:${l.area}`);
      lit.add(`r:${l.ref}`);
    }
    lit.add(focus.kind === 'area' ? `a:${focus.key}` : `r:${focus.id}`);
  }
  const dim = (id: string) => !!focus && !lit.has(id);

  const togglePin = (f: Focus) => {
    const next = sameFocus(pinned, f) ? null : f;
    setPinned(next);
    const { pathname, search } = window.location;
    const hash = next ? `#${focusHash(next, areas)}` : filter === 'all' ? '' : `#${filter}`;
    window.history.replaceState(null, '', `${pathname}${search}${hash}`);
  };
  const handlers = (f: Focus) => ({
    onMouseEnter: () => setHover(f),
    onMouseLeave: () => setHover(h => (sameFocus(h, f) ? null : h)),
    onFocus: () => setHover(f),
    onBlur: () => setHover(h => (sameFocus(h, f) ? null : h)),
    onClick: (e: React.MouseEvent) => {
      e.preventDefault();
      togglePin(f);
    },
  });
  const onCardChange = (f: Focus) => (open: boolean) => setCard(c => (open ? f : sameFocus(c, f) ? null : c));

  const areaByKey = useMemo(() => Object.fromEntries(areas.map(a => [a.key, a])), [areas]);
  const refCount = (key: string) => visibleLinks.filter(l => l.area === key).length;
  const paths = [...layout.paths].sort((a, b) => Number(touches(a)) - Number(touches(b)) || Number(a.direct) - Number(b.direct));
  const showFramework = (r: GroundingRef) => framework === 'both' || r.framework === framework;
  const whole = wholeEvaluation.filter(w => showFramework(refs[w.id]) && (filter === 'all' || w.scopes.includes('all') || w.scopes.includes(filter)));
  const notTested = untested.filter(id => showFramework(refs[id]));
  const scopeNote = (w: WholeEvaluationRef) =>
    w.scopes.includes('all') ? 'Cited for all three indices.' : `Cited for the ${w.scopes.map(s => INDEX_LABEL[s as IndexId]).join(' and ')} ${w.scopes.length > 1 ? 'indices' : 'index'}.`;

  return (
    <div className="space-y-10">
      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
          <FrameworkSwitch value={framework} onChange={setFramework} />
          <p className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-1.5">
              <LegendLine /> Names the behaviour the area tests
            </span>
            <span className="inline-flex items-center gap-1.5">
              <LegendLine dashed /> Related
            </span>
            <span className="hidden md:inline">Hover for details; click to pin.</span>
          </p>
        </div>

        {/* Diagram, from md up. */}
        <div
          className="relative hidden grid-cols-[minmax(12rem,16rem)_minmax(4rem,1fr)_fit-content(26rem)] md:grid"
          style={{ height: layout.height }}
        >
          <div className="relative">
            {layout.leftHeads.map(h => (
              <p
                key={h.id}
                className="absolute inset-x-0 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.06em] text-muted-foreground"
                style={{ top: h.top, height: HEAD }}
              >
                <span className={cn('size-2 rounded-full', INDEX_TONE[h.id].bar)} aria-hidden />
                {INDEX_LABEL[h.id]}
              </p>
            ))}
            {layout.areas.map(n => {
              const a = areaByKey[n.id];
              const f: Focus = { kind: 'area', key: a.key };
              const pinnedHere = sameFocus(pin, f);
              return (
                <HoverCard key={a.key} onOpenChange={onCardChange(f)}>
                  <HoverCardTrigger
                    href={`#${a.slug}`}
                    data-area={a.slug}
                    delay={450}
                    closeDelay={150}
                    {...handlers(f)}
                    className={cn(
                      'absolute inset-x-0 flex min-w-0 flex-col justify-center rounded-md border border-l-[3px] bg-card pr-2 pl-2.5 transition-opacity outline-none focus-visible:ring-2 focus-visible:ring-ink/40',
                      INDEX_TONE[a.index].line,
                      pinnedHere ? 'border-y-ink/50 border-r-ink/50 bg-muted' : 'border-y-border border-r-border',
                      a.gap && !pinnedHere && 'border-dashed',
                      dim(`a:${a.key}`) && 'opacity-30',
                    )}
                    style={{ top: n.top + 3, height: n.height - 6 }}
                  >
                    <span className="truncate text-[13px] font-semibold leading-tight text-ink">{a.name}</span>
                    <span className="truncate text-[11px] tabular leading-tight text-muted-foreground">
                      {a.fullItems.toLocaleString()} questions{a.gap ? ' · not scored' : ''}
                    </span>
                  </HoverCardTrigger>
                  <HoverCardContent side="left" align="center" sideOffset={8} className="w-80 space-y-1.5 p-3">
                    <AreaCardBody area={a} refCount={refCount(a.key)} />
                  </HoverCardContent>
                </HoverCard>
              );
            })}
          </div>

          <svg viewBox={`0 0 100 ${layout.height}`} preserveAspectRatio="none" className="h-full w-full" aria-hidden>
            {paths.map(p => {
              const on = touches(p);
              const opacity = focus ? (on ? 0.9 : 0.06) : 0.45;
              return (
                <path
                  key={`${p.area}|${p.ref}`}
                  d={`M0,${p.y1} C50,${p.y1} 50,${p.y2} 100,${p.y2}`}
                  fill="none"
                  vectorEffect="non-scaling-stroke"
                  style={{ stroke: `var(--color-${areaByKey[p.area].index})`, strokeOpacity: opacity, transition: 'stroke-opacity 120ms' }}
                  strokeWidth={p.direct ? (on ? 2.25 : 1.5) : on ? 1.5 : 1}
                  strokeDasharray={p.direct ? undefined : '5 4'}
                />
              );
            })}
          </svg>

          <div className="relative">
            {/* Rows are absolutely positioned; this invisible copy sizes the column to the longest label. */}
            <div className="invisible h-0 overflow-hidden text-xs" aria-hidden>
              {layout.refs.map(n => (
                <div key={n.id} className="flex gap-2 whitespace-nowrap border-l-2 pr-2 pl-2.5">
                  <span className="font-semibold tabular">{refs[n.id].short}</span>
                  <span>{refs[n.id].title}</span>
                </div>
              ))}
            </div>
            {layout.rightHeads.map(h => (
              <p
                key={h.id}
                className="absolute inset-x-0 flex items-center pl-2.5 text-[11px] font-semibold uppercase tracking-[0.06em] text-muted-foreground"
                style={{ top: h.top, height: HEAD }}
              >
                {h.id === 'EU AI Act' ? 'EU AI Act' : 'GPAI Code of Practice'}
                <span className="ml-1.5 font-normal tabular">· {h.count}</span>
              </p>
            ))}
            {layout.refs.map(n => {
              const r = refs[n.id];
              const f: Focus = { kind: 'ref', id: n.id };
              const pinnedHere = sameFocus(pin, f);
              const on = !!focus && lit.has(`r:${n.id}`);
              return (
                <HoverCard key={n.id} onOpenChange={onCardChange(f)}>
                  <HoverCardTrigger
                    href={`#ref-${n.id}`}
                    data-ref={n.id}
                    delay={450}
                    closeDelay={150}
                    {...handlers(f)}
                    className={cn(
                      'absolute inset-x-0 flex min-w-0 items-center gap-2 rounded-r-md border-l-2 pr-2 pl-2.5 text-xs transition-opacity outline-none focus-visible:ring-2 focus-visible:ring-ink/40',
                      pinnedHere ? 'border-ink bg-muted' : on ? 'border-ink/60 bg-muted/50' : 'border-ink/20',
                      dim(`r:${n.id}`) && 'opacity-30',
                    )}
                    style={{ top: n.top + 1, height: n.height - 2 }}
                  >
                    <span className="shrink-0 font-semibold tabular text-ink">{r.short}</span>
                    <span className="truncate text-ink/70">{r.title}</span>
                  </HoverCardTrigger>
                  <HoverCardContent side="right" align="center" sideOffset={8} className="w-80 space-y-1.5 p-3">
                    <RefCardBody item={r} />
                  </HoverCardContent>
                </HoverCard>
              );
            })}
          </div>
        </div>

        {/* List, below md. */}
        <div className="space-y-6 md:hidden">
          {INDICES.filter(i => visibleAreas.some(a => a.index === i.id)).map(i => (
            <div key={i.id} className="space-y-2">
              <p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.06em] text-muted-foreground">
                <span className={cn('size-2 rounded-full', INDEX_TONE[i.id].bar)} aria-hidden />
                {i.label}
              </p>
              {visibleAreas.filter(a => a.index === i.id).map(a => {
                const own = visibleLinks.filter(l => l.area === a.key).sort((x, y) => Number(y.direct) - Number(x.direct));
                return (
                  <div
                    key={a.key}
                    data-area={a.slug}
                    className={cn('rounded-md border border-l-[3px] border-border bg-card p-3', INDEX_TONE[a.index].line, sameFocus(pin, { kind: 'area', key: a.key }) && 'bg-muted')}
                  >
                    <p className="text-sm font-semibold text-ink">{a.name}</p>
                    <p className="text-[11px] tabular text-muted-foreground">
                      {a.fullItems.toLocaleString()} questions{a.gap ? ' · not scored separately' : ''}
                    </p>
                    {own.length > 0 && (
                      <ul className="mt-2 space-y-1">
                        {own.map(l => (
                          <li key={l.ref} data-ref={l.ref}>
                            <a
                              href={refs[l.ref].url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-baseline gap-2 text-xs leading-snug hover:underline"
                            >
                              <span className={cn('shrink-0 font-semibold tabular', l.direct ? 'text-ink' : 'text-muted-foreground')}>{citeLabel(refs[l.ref])}</span>
                              <span className={l.direct ? 'text-ink/80' : 'text-muted-foreground'}>{refs[l.ref].title}</span>
                            </a>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      <div className="space-y-6 border-t border-border pt-8">
        {whole.length > 0 && (
          <ChipRow
            title="Apply to the whole evaluation"
            description="Evaluation duties that every ComplAI CORE result speaks to, rather than one evaluation area."
          >
            {whole.map(w => <RefChip key={w.id} item={refs[w.id]} note={scopeNote(w)} />)}
          </ChipRow>
        )}
        {notTested.length > 0 && (
          <ChipRow
            title="Not tested by ComplAI CORE yet"
            description="Provisions that need tests CORE doesn’t run yet, such as agentic and dangerous capabilities, transparency, content marking and copyright."
          >
            {notTested.map(id => <RefChip key={id} item={refs[id]} />)}
          </ChipRow>
        )}
      </div>
    </div>
  );
}
