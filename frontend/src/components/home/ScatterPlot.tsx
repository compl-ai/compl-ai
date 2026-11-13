'use client';
import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { CartesianGrid, ReferenceDot, Scatter, ScatterChart, Tooltip as RechartsTooltip, XAxis, YAxis } from 'recharts';
import type { JoinedModelData } from '@/lib/data';
import { INDICES, THETA_MEAN, formatInterval, formatScore, getIndexTheta } from '@/lib/scores';
import { fmtMonth, releaseOf } from './explorer';

export const AXIS_TICK = { fill: 'var(--muted-foreground)', fontSize: 11 };
const AXIS_LABEL = { fill: 'var(--color-ink)', fontSize: 12, fontWeight: 600 };

export type LabelPos = { dx: number; dy: number; anchor: 'start' | 'middle' | 'end' };

export type Point = {
  id: string;
  name: string;
  x: number;
  y: number;
  r: number;
  color: string;
  hollow?: boolean;
  shape?: 'circle' | 'square';
  opacity?: number;
  label?: string;
  labelPos?: LabelPos;
  // Higher-priority labels are placed first; muted ones are drawn lighter, for context models.
  labelPriority?: number;
  labelMuted?: boolean;
  // Absent for points that summarise several models (e.g. a median); those get a plain tooltip.
  model?: JoinedModelData;
  subtitle?: string;
  extra: [string, string][];
  // 90% intervals in axis units, and the resulting whiskers as pixel offsets from the dot.
  xCi?: [number, number];
  yCi?: [number, number];
  bar?: { l: number; r: number; t: number; b: number };
};

export type Axis = {
  label: string;
  domain: [number, number];
  log?: boolean;
  ticks?: number[];
  format?: (v: number) => string;
  width?: number;
  strong?: boolean;
};

// Text annotations drawn on the plot; label placement avoids them.
export type Note = { x: number; y: number; text: string; anchor?: LabelPos['anchor']; dx?: number; dy?: number };

export type ChartSpec = {
  points: Point[];
  x: Axis;
  y: Axis;
  refs: React.ReactNode[];
  notes?: Note[];
  verticalGrid?: boolean;
};

export function scoreDomain(values: number[]): [number, number] {
  if (!values.length) return [40, 160];
  return [Math.floor(Math.min(...values, THETA_MEAN) / 10) * 10 - 5, Math.ceil(Math.max(...values, THETA_MEAN) / 10) * 10 + 5];
}

// Ticks every 20 points, aligned so the panel average (100) is always one of them.
export function scoreTicks([lo, hi]: [number, number], every = 20): number[] {
  const ticks: number[] = [];
  for (let t = THETA_MEAN + Math.ceil((lo - THETA_MEAN) / every) * every; t <= hi; t += every) ticks.push(t);
  return ticks;
}

export const scoreAxis = (label: string, values: number[]): Axis => {
  const domain = scoreDomain(values);
  return { label, domain, ticks: scoreTicks(domain) };
};

export function logDomain(values: number[]): [number, number] {
  return [10 ** Math.floor(Math.log10(Math.min(...values))), 10 ** Math.ceil(Math.log10(Math.max(...values)))];
}

export const logTicks = ([lo, hi]: [number, number], steps = [1, 3]) => {
  const ticks: number[] = [];
  for (let d = lo; d <= hi; d *= 10) steps.forEach(s => ticks.push(d * s));
  return ticks.filter(t => t <= hi);
};

// Date axis padded by ~6 weeks, with ticks every `months` months.
export function dateAxis(dates: number[], months = 6, end?: number): Axis {
  const pad = 45 * 864e5;
  const domain: [number, number] = dates.length ? [Math.min(...dates) - pad, Math.max(end ?? -Infinity, ...dates) + pad] : [0, 1];
  const ticks: number[] = [];
  const d = new Date(domain[0]);
  for (let t = Date.UTC(d.getUTCFullYear(), Math.ceil((d.getUTCMonth() + 1) / months) * months, 1); t <= domain[1];) {
    ticks.push(t);
    const n = new Date(t);
    t = Date.UTC(n.getUTCFullYear(), n.getUTCMonth() + months, 1);
  }
  return { label: 'Release date', domain, ticks, format: v => new Date(v).toLocaleDateString('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' }) };
}

const relPos = (v: number, [lo, hi]: [number, number], log = false) =>
  log ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo);

// Deterministic jitter so strip plots don't reshuffle between renders.
export function jitter(id: string): number {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
  return ((h >>> 0) % 1000) / 1000 - 0.5;
}

export const MARGIN = { top: 12, right: 16, bottom: 22, left: 0 };
const X_AXIS_HEIGHT = 30;
const yAxisWidth = (y: Axis) => y.width ?? (y.log ? 64 : 58);
const LABEL_CHAR_W = 6.3;
const LABEL_H = 13;
const LABEL_PAD = 2;

type Box = { l: number; r: number; t: number; b: number };
const overlaps = (a: Box, b: Box) => a.l < b.r && b.l < a.r && a.t < b.b && b.t < a.b;

function textBox(x: number, y: number, text: string, anchor: LabelPos['anchor']): Box {
  const w = text.length * LABEL_CHAR_W;
  const l = anchor === 'start' ? x : anchor === 'end' ? x - w : x - w / 2;
  return { l, r: l + w, t: y - LABEL_H + 3, b: y + 3 };
}

const noteOffset = (n: Note) => ({ dx: n.dx ?? -4, dy: n.dy ?? 13, anchor: n.anchor ?? 'end' as const });

// Greedy label placement in pixel space: try positions around each labelled dot and take the
// one that stays inside the plot, doesn't hit an earlier label or note, and covers the fewest dots.
function placeLabels(chart: ChartSpec, width: number, height: number, inset: number): { points: Point[]; notes: Note[] } {
  const plot = { l: inset + yAxisWidth(chart.y), r: width - MARGIN.right, t: MARGIN.top, b: height - MARGIN.bottom - X_AXIS_HEIGHT };
  const px = (v: number) => plot.l + relPos(v, chart.x.domain, chart.x.log) * (plot.r - plot.l);
  const py = (v: number) => plot.b - relPos(v, chart.y.domain, chart.y.log) * (plot.b - plot.t);
  const dots = chart.points.map(p => ({ cx: px(p.x), cy: py(p.y), r: p.r }));
  const taken: Box[] = [];
  const notes = (chart.notes ?? []).filter(n => {
    const o = noteOffset(n);
    const box = textBox(px(n.x) + o.dx, py(n.y) + o.dy, n.text, o.anchor);
    if (taken.some(t => overlaps(t, box))) return false;
    taken.push(box);
    return true;
  });
  const placed = new Map<number, LabelPos>();
  const order = chart.points.map((_, i) => i).sort((a, b) => (chart.points[b].labelPriority ?? 0) - (chart.points[a].labelPriority ?? 0));
  order.forEach(i => {
    const p = chart.points[i];
    if (!p.label) return;
    const { cx, cy } = dots[i];
    const g = p.r + 4;
    const candidates: LabelPos[] = [
      { dx: g, dy: 4, anchor: 'start' },
      { dx: -g, dy: 4, anchor: 'end' },
      { dx: 0, dy: -g, anchor: 'middle' },
      { dx: 0, dy: g + LABEL_H - 3, anchor: 'middle' },
      { dx: g - 2, dy: -g + 2, anchor: 'start' },
      { dx: -g + 2, dy: -g + 2, anchor: 'end' },
      { dx: g - 2, dy: g + 8, anchor: 'start' },
      { dx: -g + 2, dy: g + 8, anchor: 'end' },
    ];
    let best: { pos: LabelPos; cost: number } | null = null;
    for (const pos of candidates) {
      const box = textBox(cx + pos.dx, cy + pos.dy, p.label, pos.anchor);
      if (box.l < plot.l || box.r > plot.r || box.t < plot.t || box.b > plot.b) continue;
      if (taken.some(t => overlaps(t, box))) continue;
      const cost = dots.filter((d, j) => j !== i && overlaps(box, { l: d.cx - d.r, r: d.cx + d.r, t: d.cy - d.r, b: d.cy + d.r })).length;
      if (!best || cost < best.cost) best = { pos, cost };
      if (cost === 0) break;
    }
    if (best) {
      placed.set(i, best.pos);
      const b = textBox(cx + best.pos.dx, cy + best.pos.dy, p.label, best.pos.anchor);
      taken.push({ l: b.l - LABEL_PAD, r: b.r + LABEL_PAD, t: b.t - LABEL_PAD, b: b.b + LABEL_PAD });
    }
  });
  const clampX = (v: number) => Math.min(plot.r, Math.max(plot.l, v));
  const clampY = (v: number) => Math.min(plot.b, Math.max(plot.t, v));
  return {
    points: chart.points.map((p, i) => {
      const { cx, cy } = dots[i];
      const bar = p.xCi || p.yCi
        ? {
          l: p.xCi ? clampX(px(p.xCi[0])) - cx : 0,
          r: p.xCi ? clampX(px(p.xCi[1])) - cx : 0,
          t: p.yCi ? clampY(py(p.yCi[1])) - cy : 0,
          b: p.yCi ? clampY(py(p.yCi[0])) - cy : 0,
        }
        : undefined;
      return { ...p, bar, label: placed.has(i) ? p.label : undefined, labelPos: placed.get(i) };
    }),
    notes,
  };
}

function Dot(props: { cx?: number; cy?: number; payload?: Point }) {
  const { cx, cy, payload: p } = props;
  if (cx === undefined || cy === undefined || !p) return null;
  const common = {
    fill: p.hollow ? 'var(--card)' : p.color,
    fillOpacity: p.hollow ? 1 : (p.opacity ?? 0.85),
    stroke: p.hollow ? p.color : 'var(--card)',
    strokeOpacity: p.hollow ? (p.opacity ?? 1) : 1,
    strokeWidth: 1.5,
  };
  return (
    <g>
      {p.shape === 'square'
        ? <rect x={cx - p.r * 0.9} y={cy - p.r * 0.9} width={p.r * 1.8} height={p.r * 1.8} {...common} />
        : <circle cx={cx} cy={cy} r={p.r} {...common} />}
      {p.label && p.labelPos && (
        <text
          x={cx + p.labelPos.dx}
          y={cy + p.labelPos.dy}
          textAnchor={p.labelPos.anchor}
          className="pointer-events-none"
          style={{ fontSize: 11, fontWeight: p.labelMuted ? 500 : 600, fill: p.labelMuted ? 'var(--muted-foreground)' : 'var(--color-ink)', paintOrder: 'stroke', stroke: 'var(--card)', strokeWidth: 3 }}
        >
          {p.label}
        </text>
      )}
    </g>
  );
}

const CAP = 3;

function CiBar(props: { cx?: number; cy?: number; payload?: Point }) {
  const { cx, cy, payload: p } = props;
  if (cx === undefined || cy === undefined || !p?.bar) return null;
  const { l, r, t, b } = p.bar;
  return (
    <g stroke={p.color} strokeOpacity={0.45} strokeWidth={1.25}>
      {(l !== 0 || r !== 0) && (
        <>
          <line x1={cx + l} y1={cy} x2={cx + r} y2={cy} />
          <line x1={cx + l} y1={cy - CAP} x2={cx + l} y2={cy + CAP} />
          <line x1={cx + r} y1={cy - CAP} x2={cx + r} y2={cy + CAP} />
        </>
      )}
      {(t !== 0 || b !== 0) && (
        <>
          <line x1={cx} y1={cy + t} x2={cx} y2={cy + b} />
          <line x1={cx - CAP} y1={cy + t} x2={cx + CAP} y2={cy + t} />
          <line x1={cx - CAP} y1={cy + b} x2={cx + CAP} y2={cy + b} />
        </>
      )}
    </g>
  );
}

function ChartTooltip({ active, payload }: { active?: boolean; payload?: { payload: Point }[] }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const m = d.model;
  const released = m ? releaseOf(m) : null;
  const rows: [string, string][] = [...d.extra, ...(released !== null ? [['Released', fmtMonth(released)] as [string, string]] : [])];
  return (
    <div className="bg-card border border-border shadow-lg p-3 text-xs rounded-xl min-w-[240px] font-plex">
      <div className="flex items-center gap-2 mb-0.5">
        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d.color }} />
        <span className="font-bold text-ink text-sm">{d.name}</span>
      </div>
      <div className="text-muted-foreground mb-3 pl-[18px]">{m ? m.metadata?.organization || 'Organization not published' : d.subtitle}</div>
      <div className="flex flex-col gap-1.5">
        {m && INDICES.map(idx => {
          const t = getIndexTheta(m, idx.id);
          return (
            <div key={idx.id} className="flex justify-between gap-6">
              <span className="text-muted-foreground">{idx.label}</span>
              <span className="tabular text-ink">
                {t ? <><span className="font-semibold">{formatScore(t.theta)}</span> <span className="text-muted-foreground">{formatInterval(t.interval)}</span></> : '—'}
              </span>
            </div>
          );
        })}
        {m && rows.length > 0 && <div className="my-1 border-t border-border" />}
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-6">
            <span className="text-muted-foreground">{k}</span>
            <span className="tabular font-medium text-ink">{v}</span>
          </div>
        ))}
      </div>
      {m && <p className="mt-2.5 text-[10px] text-muted-foreground">Ranges are 90% intervals. Click for the full report.</p>}
    </div>
  );
}

// Measures its own box; `inset` keeps the plotting area clear of the question card on wide screens.
export function ScatterPlot({ chart, inset = 0, showCi = false, className }: { chart: ChartSpec; inset?: number; showCi?: boolean; className?: string }) {
  const router = useRouter();
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSize(s => (s.width === r.width && s.height === r.height ? s : { width: r.width, height: r.height }));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const ready = size.width > 0 && size.height > 0;
  const { points, notes } = ready ? placeLabels(chart, size.width, size.height, inset) : { points: [], notes: [] };

  return (
    <div ref={ref} className={className}>
      {ready && (
        <ScatterChart width={size.width} height={size.height} margin={{ ...MARGIN, left: inset }}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="var(--color-grid)"
            vertical={chart.verticalGrid ?? true}
            horizontal={({ key, x1, x2, y1, y2 }) => <line key={key ?? y1} x1={inset ? 0 : x1} y1={y1} x2={x2} y2={y2} stroke="var(--color-grid)" strokeDasharray="3 3" />}
          />
          <XAxis
            dataKey="x"
            type="number"
            domain={chart.x.domain}
            scale={chart.x.log ? 'log' : 'linear'}
            allowDataOverflow
            ticks={chart.x.ticks}
            tickFormatter={chart.x.format}
            tick={chart.x.strong ? { ...AXIS_TICK, fontSize: 12, fontWeight: 600, fill: 'var(--color-ink)' } : AXIS_TICK}
            tickLine={false}
            axisLine={false}
            tickMargin={10}
            height={X_AXIS_HEIGHT}
            label={chart.x.label ? { value: chart.x.label, position: 'insideBottom', offset: -14, ...AXIS_LABEL } : undefined}
          />
          <YAxis
            dataKey="y"
            type="number"
            domain={chart.y.domain}
            scale={chart.y.log ? 'log' : 'linear'}
            allowDataOverflow
            ticks={chart.y.ticks}
            tickFormatter={chart.y.format}
            tick={chart.y.strong ? { ...AXIS_TICK, fontSize: 12, fontWeight: 600, fill: 'var(--color-ink)' } : AXIS_TICK}
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            width={yAxisWidth(chart.y)}
            label={chart.y.label ? { value: chart.y.label, angle: -90, position: 'insideLeft', offset: 6, ...AXIS_LABEL, style: { textAnchor: 'middle' } } : undefined}
          />
          {chart.refs}
          {notes.map(n => {
            const o = noteOffset(n);
            return (
              <ReferenceDot
                key={n.text}
                x={n.x}
                y={n.y}
                ifOverflow="visible"
                shape={(d: { cx?: number; cy?: number }) => (
                  <text x={(d.cx ?? 0) + o.dx} y={(d.cy ?? 0) + o.dy} textAnchor={o.anchor} style={{ fontSize: 10.5, fill: 'var(--muted-foreground)' }}>
                    {n.text}
                  </text>
                )}
              />
            );
          })}
          {showCi && <Scatter data={points.filter(p => p.bar)} shape={CiBar} isAnimationActive={false} className="pointer-events-none" />}
          <RechartsTooltip cursor={false} isAnimationActive={false} content={<ChartTooltip />} />
          <Scatter
            data={points}
            shape={Dot}
            isAnimationActive={false}
            className="cursor-pointer"
            onClick={(p: { payload?: Point }) => p.payload?.model && router.push(`/models/${p.payload.model.yamlId}`)}
          />
        </ScatterChart>
      )}
    </div>
  );
}
