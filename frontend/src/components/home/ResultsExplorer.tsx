'use client';
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { useRouter } from 'next/navigation';
import { CartesianGrid, ReferenceDot, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip as RechartsTooltip, XAxis, YAxis } from 'recharts';
import type { JoinedModelData } from '@/lib/data';
import { INDICES, type IndexId, RUN_TYPES, THETA_MEAN, formatInterval, formatScore, formatTokens, formatPercent, getIndexTheta, getRunType, scaleTheta } from '@/lib/scores';
import {
  LARGE_PARAMS, QUESTIONS, SIZE_BUCKETS, type QuestionId, type SizeBucket, type Weights, capabilityFrontier, fmtParams, indexScore, median, nameOf, paramsOf, releaseOf, sizeBucketOf, tokensOf, weightsOf,
} from './explorer';

const WEIGHT_STYLE: Record<Weights, { color: string; label: string }> = {
  open: { color: 'var(--color-teal-strong)', label: 'Open weights' },
  proprietary: { color: 'var(--color-lavender-strong)', label: 'Proprietary' },
  unknown: { color: '#AEB4BA', label: 'Not published' },
};
const FRONTIER_COLOR = 'var(--color-brand)';
const AXIS_TICK = { fill: '#5B6670', fontSize: 11 };
const AXIS_LABEL = { fill: '#051C2C', fontSize: 13, fontWeight: 600 };

type Point = {
  id: string;
  name: string;
  x: number;
  y: number;
  r: number;
  hollow?: boolean;
  label?: string;
  labelPos?: LabelPos;
  color: string;
  model: JoinedModelData;
  extra: [string, string][];
  // 90% intervals in axis units, and the resulting whiskers as pixel offsets from the dot.
  xCi?: [number, number];
  yCi?: [number, number];
  bar?: { l: number; r: number; t: number; b: number };
};

type Axis = {
  label: string;
  domain: [number, number];
  log?: boolean;
  ticks?: number[];
  format?: (v: number) => string;
};

type LabelPos = { dx: number; dy: number; anchor: 'start' | 'middle' | 'end' };

// Text annotations drawn on the plot (e.g. the cost diagonals); label placement avoids them.
type Note = { x: number; y: number; text: string };

type KeyItem = { kind: 'line' | 'dash' | 'hollow' | 'bar'; label: string };

type ChartSpec = {
  points: Point[];
  x: Axis;
  y: Axis;
  refs: React.ReactNode[];
  notes?: Note[];
  excluded: number;
  excludedReason: string;
  keys: KeyItem[];
};

const noopSubscribe = () => () => {};

function scoreDomain(values: number[]): [number, number] {
  if (!values.length) return [40, 160];
  return [Math.floor(Math.min(...values, THETA_MEAN) / 10) * 10 - 5, Math.ceil(Math.max(...values, THETA_MEAN) / 10) * 10 + 5];
}

// Ticks every 20 points, aligned so the panel average (100) is always one of them.
function scoreTicks([lo, hi]: [number, number]): number[] {
  const ticks: number[] = [];
  for (let t = THETA_MEAN + Math.ceil((lo - THETA_MEAN) / 20) * 20; t <= hi; t += 20) ticks.push(t);
  return ticks;
}

const scoreAxis = (label: string, values: number[]): Axis => {
  const domain = scoreDomain(values);
  return { label, domain, ticks: scoreTicks(domain) };
};

const tokenTick = (v: number) => (v >= 1e9 ? `${v / 1e9}B` : `${v / 1e6}M`);

function logDomain(values: number[]): [number, number] {
  return [10 ** Math.floor(Math.log10(Math.min(...values))), 10 ** Math.ceil(Math.log10(Math.max(...values)))];
}

const logTicks = ([lo, hi]: [number, number]) => {
  const ticks: number[] = [];
  for (let d = lo; d <= hi; d *= 10) ticks.push(d, d * 3);
  return ticks.filter(t => t <= hi);
};

const relPos = (v: number, [lo, hi]: [number, number], log = false) =>
  log ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo);

// Deterministic jitter so strip plots don't reshuffle between renders.
function jitter(id: string): number {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
  return ((h >>> 0) % 1000) / 1000 - 0.5;
}

const indexLabel = (id: IndexId) => INDICES.find(i => i.id === id)!.label;

function labelTop(points: Point[], by: (p: Point) => number, n: number) {
  [...points].sort((a, b) => by(b) - by(a)).slice(0, n).forEach(p => { p.label = p.name; });
}

// Label one point per model name at most (the group chart shows each model three times).
function labelOnce(points: Point[], picks: (Point | undefined)[]) {
  const named = new Set(points.filter(p => p.label).map(p => p.label));
  picks.forEach(p => {
    if (p && !named.has(p.name)) { p.label = p.name; named.add(p.name); }
  });
}

const maxBy = (points: Point[], by: (p: Point) => number) => points.reduce<Point | undefined>((m, p) => (!m || by(p) > by(m) ? p : m), undefined);

// Chart geometry shared by the ScatterChart props and the label layout below.
const MARGIN = { top: 12, right: 16, bottom: 22, left: 0 };
const X_AXIS_HEIGHT = 30;
const CARD_GAP = 28;
const yAxisWidth = (y: Axis) => (y.log ? 76 : 62);
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

const NOTE_OFFSET = { dx: -4, dy: 13 };

// Greedy label placement in pixel space: try positions around each labelled dot and take the
// one that stays inside the plot, doesn't hit an earlier label or note, and covers the fewest dots.
function placeLabels(chart: ChartSpec, width: number, height: number, inset: number): { points: Point[]; notes: Note[] } {
  const plot = { l: inset + yAxisWidth(chart.y), r: width - MARGIN.right, t: MARGIN.top, b: height - MARGIN.bottom - X_AXIS_HEIGHT };
  const px = (v: number) => plot.l + relPos(v, chart.x.domain, chart.x.log) * (plot.r - plot.l);
  const py = (v: number) => plot.b - relPos(v, chart.y.domain, chart.y.log) * (plot.b - plot.t);
  const dots = chart.points.map(p => ({ cx: px(p.x), cy: py(p.y), r: p.r }));
  const taken: Box[] = [];
  // Notes go first; on narrow charts the diagonals converge, so drop any note that would collide.
  const notes = (chart.notes ?? []).filter(n => {
    const box = textBox(px(n.x) + NOTE_OFFSET.dx, py(n.y) + NOTE_OFFSET.dy, n.text, 'end');
    if (taken.some(t => overlaps(t, box))) return false;
    taken.push(box);
    return true;
  });
  const order = chart.points.map((p, i) => i).filter(i => chart.points[i].label);
  const placed = new Map<number, LabelPos>();
  for (const i of order) {
    const p = chart.points[i];
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
      const box = textBox(cx + pos.dx, cy + pos.dy, p.label!, pos.anchor);
      if (box.l < plot.l || box.r > plot.r || box.t < plot.t || box.b > plot.b) continue;
      if (taken.some(t => overlaps(t, box))) continue;
      const cost = dots.filter((d, j) => j !== i && overlaps(box, { l: d.cx - d.r, r: d.cx + d.r, t: d.cy - d.r, b: d.cy + d.r })).length;
      if (!best || cost < best.cost) best = { pos, cost };
      if (cost === 0) break;
    }
    if (best) {
      placed.set(i, best.pos);
      const b = textBox(cx + best.pos.dx, cy + best.pos.dy, p.label!, best.pos.anchor);
      taken.push({ l: b.l - LABEL_PAD, r: b.r + LABEL_PAD, t: b.t - LABEL_PAD, b: b.b + LABEL_PAD });
    }
  }
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
      return { ...p, bar, ...(p.label ? { label: placed.has(i) ? p.label : undefined, labelPos: placed.get(i) } : {}) };
    }),
    notes,
  };
}

function indexCi(m: JoinedModelData, idx: IndexId): [number, number] | undefined {
  const t = getIndexTheta(m, idx);
  return t ? [scaleTheta(t.interval[0]), scaleTheta(t.interval[1])] : undefined;
}

function basePoint(m: JoinedModelData, x: number, y: number): Point {
  return { id: m.yamlId, name: nameOf(m), x, y, r: 6, color: WEIGHT_STYLE[weightsOf(m)].color, model: m, extra: [] };
}

function indexScatter(models: JoinedModelData[], a: IndexId, b: IndexId): ChartSpec {
  const points = models.flatMap(m => {
    const x = indexScore(m, a);
    const y = indexScore(m, b);
    return x !== null && y !== null ? [{ ...basePoint(m, x, y), xCi: indexCi(m, a), yCi: indexCi(m, b) }] : [];
  });
  const x = scoreAxis(indexLabel(a), points.map(p => p.x));
  const y = scoreAxis(indexLabel(b), points.map(p => p.y));
  labelTop(points, p => p.x + p.y, 3);
  labelOnce(points, [maxBy(points, p => p.x), maxBy(points, p => p.y), maxBy(points, p => -p.y), maxBy(points, p => p.y - p.x), maxBy(points, p => p.x - p.y)]);
  return {
    points, x, y,
    refs: [
      <ReferenceLine key="x" x={THETA_MEAN} stroke="#C9CDD2" strokeDasharray="4 4" />,
      <ReferenceLine key="y" y={THETA_MEAN} stroke="#C9CDD2" strokeDasharray="4 4" />,
    ],
    excluded: models.length - points.length,
    excludedReason: `too few benchmarks run to score ${indexLabel(a)} or ${indexLabel(b)}`,
    keys: [{ kind: 'dash', label: `Panel average (${THETA_MEAN})` }],
  };
}

function frontierChart(models: JoinedModelData[]): ChartSpec {
  const points = models.flatMap(m => {
    const date = releaseOf(m);
    const c = indexScore(m, 'capability');
    if (date === null || c === null) return [];
    const p = basePoint(m, date, c);
    p.yCi = indexCi(m, 'capability');
    const params = paramsOf(m);
    p.r = params ? Math.min(16, 3.5 + Math.sqrt(params / 1e9) * 0.38) : 5;
    p.hollow = !params;
    p.extra = [
      ['Released', new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' })],
      ['Parameters', params ? fmtParams(params) : 'Not published'],
    ];
    return [p];
  });
  const steps = capabilityFrontier(models);
  const pad = 45 * 864e5;
  const dates = points.map(p => p.x);
  const domain: [number, number] = dates.length ? [Math.min(...dates) - pad, Math.max(...dates) + pad] : [0, 1];
  const ticks: number[] = [];
  if (dates.length) {
    const d = new Date(domain[0]);
    let t = Date.UTC(d.getUTCFullYear(), d.getUTCMonth() < 6 ? 6 : 12, 1);
    for (; t <= domain[1]; t = Date.UTC(new Date(t).getUTCFullYear(), new Date(t).getUTCMonth() + 6, 1)) ticks.push(t);
  }
  const x: Axis = {
    label: 'Release date',
    domain,
    ticks,
    format: v => new Date(v).toLocaleDateString('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' }),
  };
  const ids = new Set(steps.slice(-5).map(s => s.id));
  points.filter(p => ids.has(p.id)).forEach(p => { p.label = p.name; });
  labelOnce(points, [maxBy(points, p => p.r)]);
  const refs: React.ReactNode[] = [];
  steps.forEach((s, i) => {
    const next = steps[i + 1];
    const end = next ? next.date : domain[1];
    refs.push(<ReferenceLine key={`h${i}`} segment={[{ x: s.date, y: s.score }, { x: end, y: s.score }]} stroke={FRONTIER_COLOR} strokeWidth={2} ifOverflow="hidden" />);
    if (next) refs.push(<ReferenceLine key={`v${i}`} segment={[{ x: next.date, y: s.score }, { x: next.date, y: next.score }]} stroke={FRONTIER_COLOR} strokeWidth={2} ifOverflow="hidden" />);
  });
  return {
    points, x,
    y: scoreAxis('Capability', points.map(p => p.y)),
    refs,
    excluded: models.length - points.length,
    excludedReason: 'no Capability score or no published release date',
    keys: [{ kind: 'line', label: 'Best released so far' }, { kind: 'hollow', label: 'Size not published' }],
  };
}

function sizeChart(models: JoinedModelData[]): ChartSpec {
  const points = models.flatMap(m => {
    const params = paramsOf(m);
    const c = indexScore(m, 'capability');
    if (params === null || c === null) return [];
    const p = basePoint(m, params / 1e9, c);
    p.yCi = indexCi(m, 'capability');
    const active = m.metadata?.specs?.active_params;
    p.extra = [['Parameters', `${fmtParams(params)}${active && active < params ? ` (${fmtParams(active)} active)` : ''}`]];
    return [p];
  });
  const domain = points.length ? logDomain(points.map(p => p.x)) : [1, 1000] as [number, number];
  const x: Axis = { label: 'Total parameters (billions, log scale)', domain, log: true, ticks: logTicks(domain), format: v => (v >= 1000 ? `${v / 1000}T` : `${v}B`) };
  labelTop(points.filter(p => p.x * 1e9 < LARGE_PARAMS), p => p.y, 4);
  labelTop(points.filter(p => p.x * 1e9 >= LARGE_PARAMS), p => p.y, 2);
  labelOnce(points, [maxBy(points, p => -p.x)]);
  const large = points.filter(p => p.x * 1e9 >= LARGE_PARAMS).map(p => p.y);
  const refs: React.ReactNode[] = [
    <ReferenceLine key="split" x={LARGE_PARAMS / 1e9} stroke="#C9CDD2" strokeDasharray="4 4" />,
  ];
  if (large.length) {
    refs.push(<ReferenceLine key="bar" segment={[{ x: LARGE_PARAMS / 1e9, y: median(large) }, { x: domain[1], y: median(large) }]} stroke={FRONTIER_COLOR} strokeWidth={2} />);
  }
  return {
    points, x,
    y: scoreAxis('Capability', points.map(p => p.y)),
    refs,
    excluded: models.length - points.length,
    excludedReason: 'no published parameter count or no Capability score',
    keys: [{ kind: 'line', label: 'Median of shown 100B+ models' }, { kind: 'dash', label: '100B parameters' }],
  };
}

const GROUP_OFFSET: Record<'open' | 'proprietary', number> = { open: -0.18, proprietary: 0.18 };

function groupChart(models: JoinedModelData[]): ChartSpec {
  const points: Point[] = [];
  const refs: React.ReactNode[] = [];
  const known = models.filter(m => weightsOf(m) !== 'unknown');
  INDICES.forEach((idx, i) => {
    (['open', 'proprietary'] as const).forEach(w => {
      const group = known.filter(m => weightsOf(m) === w).flatMap(m => {
        const s = indexScore(m, idx.id);
        if (s === null) return [];
        const p = basePoint(m, i + GROUP_OFFSET[w] + jitter(m.yamlId + idx.id) * 0.16, s);
        p.id = `${m.yamlId}:${idx.id}`;
        p.r = 5;
        p.yCi = indexCi(m, idx.id);
        return [p];
      });
      points.push(...group);
      if (group.length) {
        const c = i + GROUP_OFFSET[w];
        const med = median(group.map(p => p.y));
        refs.push(<ReferenceLine key={`${idx.id}-${w}`} segment={[{ x: c - 0.13, y: med }, { x: c + 0.13, y: med }]} stroke="#051C2C" strokeWidth={2.5} />);
      }
    });
  });
  INDICES.forEach(idx => (['open', 'proprietary'] as const).forEach(w => {
    labelOnce(points, [maxBy(points.filter(p => p.id.endsWith(`:${idx.id}`) && weightsOf(p.model) === w), p => p.y)]);
  }));
  return {
    points,
    x: { label: '', domain: [-0.5, 2.5], ticks: [0, 1, 2], format: v => INDICES[Math.round(v)]?.label ?? '' },
    y: scoreAxis('Index score', points.map(p => p.y)),
    refs: [<ReferenceLine key="avg" y={THETA_MEAN} stroke="#C9CDD2" strokeDasharray="4 4" />, ...refs],
    excluded: models.length - known.length,
    excludedReason: 'open-weight status not published',
    keys: [{ kind: 'bar', label: 'Group median' }, { kind: 'dash', label: `Panel average (${THETA_MEAN})` }],
  };
}

function costChart(models: JoinedModelData[]): ChartSpec {
  const points = models.flatMap(m => {
    const t = tokensOf(m);
    if (!t) return [];
    const p = basePoint(m, t.full, t.core);
    p.extra = [
      ['Full suite', `${formatTokens(t.full)} tokens${t.measured === 'full' ? ' (measured)' : ' (estimated)'}`],
      ['CORE', `${formatTokens(t.core)} tokens${t.measured === 'core' ? ' (measured)' : ' (estimated)'}`],
      ['Reduction', formatPercent(t.reduction)],
    ];
    return [p];
  });
  const xd = points.length ? logDomain(points.map(p => p.x)) : [1e6, 1e10] as [number, number];
  const yd = points.length ? logDomain(points.map(p => p.y)) : [1e6, 1e10] as [number, number];
  const x: Axis = { label: 'Tokens, full suite (log scale)', domain: xd, log: true, ticks: logTicks(xd), format: tokenTick };
  const sorted = [...points].sort((a, b) => a.y / a.x - b.y / b.x);
  labelOnce(points, [sorted[0], sorted[sorted.length - 1], maxBy(points, p => p.x), maxBy(points, p => -p.x)]);
  const ratios = [2, 4, 8];
  const refs = ratios.map(k => (
    <ReferenceLine
      key={k}
      segment={[{ x: xd[0], y: xd[0] / k }, { x: xd[1], y: xd[1] / k }]}
      stroke="#C9CDD2"
      strokeDasharray="4 4"
      ifOverflow="hidden"
    />
  ));
  // Label each diagonal where it leaves the plot (right edge, or top edge if it exits there first).
  const notes = ratios.flatMap(k => {
    const atRight = xd[1] / k;
    const n = atRight <= yd[1] ? { x: xd[1], y: atRight } : { x: yd[1] * k, y: yd[1] };
    return n.y >= yd[0] ? [{ ...n, text: `${k}× fewer` }] : [];
  });
  return {
    points, x,
    y: { label: 'Tokens, CORE (log scale)', domain: yd, log: true, ticks: logTicks(yd), format: tokenTick },
    refs,
    notes,
    excluded: models.length - points.length,
    excludedReason: 'no token usage recorded',
    keys: [{ kind: 'dash', label: '2×, 4× and 8× fewer tokens' }],
  };
}

function buildChart(q: QuestionId, models: JoinedModelData[]): ChartSpec {
  switch (q) {
    case 'capability-safety': return indexScatter(models, 'capability', 'safety');
    case 'reliability-safety': return indexScatter(models, 'reliability', 'safety');
    case 'frontier': return frontierChart(models);
    case 'small-vs-large': return sizeChart(models);
    case 'open-vs-proprietary': return groupChart(models);
    case 'cost': return costChart(models);
  }
}

function Dot(props: { cx?: number; cy?: number; payload?: Point }) {
  const { cx, cy, payload: p } = props;
  if (cx === undefined || cy === undefined || !p) return null;
  return (
    <g>
      <circle
        cx={cx}
        cy={cy}
        r={p.r}
        fill={p.hollow ? '#FFFFFF' : p.color}
        fillOpacity={p.hollow ? 1 : 0.85}
        stroke={p.hollow ? p.color : '#FFFFFF'}
        strokeWidth={1.5}
      />
      {p.label && p.labelPos && (
        <text
          x={cx + p.labelPos.dx}
          y={cy + p.labelPos.dy}
          textAnchor={p.labelPos.anchor}
          className="pointer-events-none"
          style={{ fontSize: 11, fontWeight: 600, fill: '#051C2C', paintOrder: 'stroke', stroke: '#FFFFFF', strokeWidth: 3 }}
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
  const cov = d.model.prediction?.coverage;
  const run = getRunType(d.model);
  return (
    <div className="bg-white border border-border shadow-lg p-3 text-xs rounded-xl min-w-[240px]">
      <div className="flex items-center gap-2 mb-0.5">
        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d.color }} />
        <span className="font-bold text-ink text-sm">{d.name}</span>
      </div>
      <div className="text-muted-foreground mb-3 pl-[18px]">{d.model.metadata?.organization || 'Organization not published'}</div>
      <div className="flex flex-col gap-1.5">
        {INDICES.map(idx => {
          const t = getIndexTheta(d.model, idx.id);
          return (
            <div key={idx.id} className="flex justify-between gap-6">
              <span className="text-muted-foreground">{idx.label}</span>
              <span className="tabular text-ink">
                {t ? <><span className="font-semibold">{formatScore(t.theta)}</span> <span className="text-muted-foreground">{formatInterval(t.interval)}</span></> : '—'}
              </span>
            </div>
          );
        })}
        <div className="my-1 border-t border-border" />
        {d.extra.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-6">
            <span className="text-muted-foreground">{k}</span>
            <span className="tabular font-medium text-ink">{v}</span>
          </div>
        ))}
        {cov && (
          <div className="flex justify-between gap-6">
            <span className="text-muted-foreground">Benchmarks run</span>
            <span className="tabular font-medium text-ink">{cov.tasks_completed}/{cov.tasks_total}</span>
          </div>
        )}
        {run && (
          <div className="flex justify-between gap-6">
            <span className="text-muted-foreground">Run</span>
            <span className="font-medium text-ink">{RUN_TYPES[run].label}</span>
          </div>
        )}
      </div>
      <p className="mt-2.5 text-[10px] text-muted-foreground">Ranges are 90% intervals. Click for the full report.</p>
    </div>
  );
}

function SelectFilter({ label, value, onChange, children }: { label: string; value: string; onChange: (v: string) => void; children: React.ReactNode }) {
  return (
    <label className="flex items-center gap-2">
      <span className="sr-only">{label}</span>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        aria-label={label}
        className="h-[34px] rounded-full border border-border bg-white px-3 text-xs font-semibold text-ink focus:outline-none focus:ring-2 focus:ring-brand/40"
      >
        {children}
      </select>
    </label>
  );
}

function ProviderFilter({ providers, value, onChange }: { providers: [string, number][]; value: string[]; onChange: (v: string[]) => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDown); document.removeEventListener('keydown', onKey); };
  }, [open]);
  const toggle = (org: string) => onChange(value.includes(org) ? value.filter(o => o !== org) : [...value, org]);
  const text = value.length === 0 ? 'All providers' : value.length === 1 ? value[0] : `${value.length} providers`;
  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        aria-haspopup="true"
        aria-expanded={open}
        className="inline-flex h-[34px] items-center gap-2 rounded-full border border-border bg-white pl-3 pr-2.5 text-xs font-semibold text-ink focus:outline-none focus:ring-2 focus:ring-brand/40"
      >
        {text}
        <svg aria-hidden width="10" height="10" viewBox="0 0 10 10" className={`transition-transform ${open ? 'rotate-180' : ''}`}>
          <path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open && (
        <div className="absolute left-0 top-[40px] z-30 w-60 rounded-xl border border-border bg-white py-2 shadow-lg">
          <div className="flex items-center justify-between px-3 pb-1.5 text-[11px] text-muted-foreground">
            <span>Providers</span>
            <button type="button" onClick={() => onChange([])} className={`font-semibold text-brand-strong hover:underline ${value.length ? '' : 'invisible'}`}>
              Clear
            </button>
          </div>
          <ul className="max-h-72 overflow-y-auto">
            {providers.map(([org, n]) => (
              <li key={org}>
                <label className="flex cursor-pointer items-center gap-2.5 px-3 py-1.5 text-xs text-ink hover:bg-muted/60">
                  <input type="checkbox" checked={value.includes(org)} onChange={() => toggle(org)} className="h-3.5 w-3.5 accent-[var(--color-brand)]" />
                  <span className="flex-1 truncate">{org}</span>
                  <span className="tabular text-muted-foreground">{n}</span>
                </label>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function Switch({ label, checked, disabled, onChange }: { label: string; checked: boolean; disabled?: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className="inline-flex h-[34px] items-center gap-2 px-1 text-xs font-semibold text-ink disabled:cursor-not-allowed disabled:opacity-40"
    >
      <span className={`relative h-[18px] w-8 rounded-full transition-colors ${checked && !disabled ? 'bg-brand' : 'bg-[#D5D9DD]'}`}>
        <span className={`absolute top-[2px] h-[14px] w-[14px] rounded-full bg-white shadow transition-[left] ${checked && !disabled ? 'left-[16px]' : 'left-[2px]'}`} />
      </span>
      {label}
    </button>
  );
}

function KeySwatch({ kind }: { kind: KeyItem['kind'] }) {
  if (kind === 'line') return <span className="inline-block h-0.5 w-4 rounded" style={{ backgroundColor: FRONTIER_COLOR }} />;
  if (kind === 'bar') return <span className="inline-block h-[3px] w-4 rounded bg-ink" />;
  if (kind === 'hollow') return <span className="inline-block h-2.5 w-2.5 rounded-full border-[1.5px] border-muted-foreground bg-white" />;
  return <span className="inline-block w-4 border-t-[1.5px] border-dashed border-[#AEB4BA]" />;
}

export function ResultsExplorer({ models, findings }: { models: JoinedModelData[]; findings: Record<QuestionId, string | null> }) {
  const router = useRouter();
  const [question, setQuestion] = useState<QuestionId>('capability-safety');
  const [provider, setProvider] = useState<string[]>([]);
  const [showCi, setShowCi] = useState(false);
  const [weights, setWeights] = useState<'all' | 'open' | 'proprietary'>('all');
  const [size, setSize] = useState<'all' | SizeBucket>('all');
  // Recharts measures the DOM, so the chart renders on the client only.
  const isMounted = useSyncExternalStore(noopSubscribe, () => true, () => false);
  const plotRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  // inset: where the plotting area starts, so data clears the question card that floats over the plot on wide screens.
  const [plotSize, setPlotSize] = useState({ width: 0, height: 0, inset: 0 });
  useEffect(() => {
    const el = plotRef.current;
    const card = cardRef.current;
    if (!el || !card) return;
    const wide = window.matchMedia('(min-width: 1024px)');
    const measure = () => {
      const r = el.getBoundingClientRect();
      const inset = wide.matches ? Math.max(0, Math.round(card.getBoundingClientRect().right - r.left + CARD_GAP)) : MARGIN.left;
      setPlotSize(s => (s.width === r.width && s.height === r.height && s.inset === inset ? s : { width: r.width, height: r.height, inset }));
    };
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    ro.observe(card);
    wide.addEventListener('change', measure);
    return () => { ro.disconnect(); wide.removeEventListener('change', measure); };
  }, []);

  const providers = useMemo(() => {
    const counts = new Map<string, number>();
    models.forEach(m => {
      const org = m.metadata?.organization;
      if (org) counts.set(org, (counts.get(org) ?? 0) + 1);
    });
    return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [models]);

  const filtered = useMemo(() => models.filter(m => {
    if (provider.length && !provider.includes(m.metadata?.organization ?? '')) return false;
    if (weights !== 'all' && weightsOf(m) !== weights) return false;
    if (size !== 'all' && sizeBucketOf(m.metadata?.specs?.total_params) !== size) return false;
    return true;
  }), [models, provider, weights, size]);

  const chart = useMemo(() => buildChart(question, filtered), [question, filtered]);
  const { points, notes } = useMemo(
    () => (plotSize.width ? placeLabels(chart, plotSize.width, plotSize.height, plotSize.inset) : { points: [], notes: [] }),
    [chart, plotSize.width, plotSize.height, plotSize.inset],
  );
  const shownModels = new Set(chart.points.map(p => p.model.id)).size;
  const filtersActive = provider.length > 0 || weights !== 'all' || size !== 'all';
  const ciAvailable = question !== 'cost';

  return (
    <div className="grid gap-6 rounded-panel border border-border bg-white p-5 md:p-8 lg:relative lg:left-1/2 lg:w-[max(calc(100%+48px),80vw)] lg:-translate-x-1/2 lg:gap-0 lg:min-h-[790px] xl:min-h-[740px]">
      <div
        ref={cardRef}
        className="relative z-10 self-start overflow-hidden rounded-2xl border border-border bg-white/90 shadow-[0_8px_30px_-12px_rgba(5,28,44,0.18)] backdrop-blur-sm lg:[grid-area:1/1] lg:w-[360px] xl:w-[390px]"
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
                  className={`group flex w-full cursor-pointer items-baseline gap-3 px-5 py-4 text-left transition-colors ${open ? '' : 'hover:bg-muted/50'}`}
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
                    <p className="text-muted-foreground">{q.context}</p>
                    <div className="mt-3 border-l-2 border-brand pl-3">
                      <p className="font-plex text-[11px] font-semibold uppercase tracking-[0.1em] text-brand-strong">Finding</p>
                      <p className="mt-1 text-ink">{findings[q.id] ?? 'Not enough data yet to summarise this question.'}</p>
                    </div>
                  </div>
                )}
              </li>
            );
          })}
        </ol>
        <p className="border-t border-border px-5 py-3 text-xs text-muted-foreground leading-relaxed">
          Findings are computed from all evaluated models and ignore the filters.
        </p>
      </div>

      <div
        id="results-explorer-chart"
        className="font-plex min-w-0 lg:flex lg:flex-col lg:[grid-area:1/1]"
      >
        <div className="flex flex-wrap items-center gap-2 lg:justify-end lg:pl-[388px] xl:pl-[418px]">
          <ProviderFilter providers={providers} value={provider} onChange={setProvider} />
          <SelectFilter label="Weights" value={weights} onChange={v => setWeights(v as typeof weights)}>
            <option value="all">Open and proprietary</option>
            <option value="open">Open weights</option>
            <option value="proprietary">Proprietary</option>
          </SelectFilter>
          <SelectFilter label="Size" value={size} onChange={v => setSize(v as 'all' | SizeBucket)}>
            <option value="all">All sizes</option>
            {SIZE_BUCKETS.map(b => <option key={b.id} value={b.id}>{b.label}</option>)}
          </SelectFilter>
          <span className="mx-1 hidden h-5 w-px bg-border sm:block" />
          <Switch label="Show 90% intervals" checked={showCi} disabled={!ciAvailable} onChange={setShowCi} />
          <button
            onClick={() => { setProvider([]); setWeights('all'); setSize('all'); }}
            className={`h-[34px] px-2 text-xs font-semibold text-brand-strong hover:underline lg:order-first ${filtersActive ? '' : 'invisible'}`}
          >
            Clear filters
          </button>
        </div>

        <p className="mt-3 min-h-[2.5em] text-xs text-muted-foreground leading-snug lg:ml-auto lg:max-w-[560px] lg:pl-[28px] lg:text-right">
          Showing {shownModels} of {filtered.length}{filtersActive ? ` filtered` : ''} models
          {chart.excluded > 0 ? `; ${chart.excluded} excluded: ${chart.excludedReason}.` : '.'}
        </p>

        <div ref={plotRef} className="mt-1 h-[400px] md:h-[520px] lg:h-auto lg:min-h-[520px] lg:flex-1">
          {isMounted && points.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ ...MARGIN, left: plotSize.inset }}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#EEF0F2"
                  vertical={question !== 'open-vs-proprietary'}
                  horizontal={({ key, x2, y1, y2 }) => <line key={key ?? y1} x1={0} y1={y1} x2={x2} y2={y2} stroke="#EEF0F2" strokeDasharray="3 3" />}
                />
                <XAxis
                  dataKey="x"
                  type="number"
                  domain={chart.x.domain}
                  scale={chart.x.log ? 'log' : 'linear'}
                  allowDataOverflow
                  ticks={chart.x.ticks}
                  tickFormatter={chart.x.format}
                  tick={question === 'open-vs-proprietary' ? { ...AXIS_TICK, fontSize: 13, fontWeight: 600, fill: '#051C2C' } : AXIS_TICK}
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
                  tick={AXIS_TICK}
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                  width={yAxisWidth(chart.y)}
                  label={{ value: chart.y.label, angle: -90, position: 'insideLeft', offset: 6, ...AXIS_LABEL, style: { textAnchor: 'middle' } }}
                />
                {chart.refs}
                {notes.map(n => (
                  <ReferenceDot
                    key={n.text}
                    x={n.x}
                    y={n.y}
                    ifOverflow="visible"
                    shape={(d: { cx?: number; cy?: number }) => (
                      <text x={(d.cx ?? 0) + NOTE_OFFSET.dx} y={(d.cy ?? 0) + NOTE_OFFSET.dy} textAnchor="end" style={{ fontSize: 10, fill: '#8A939B' }}>
                        {n.text}
                      </text>
                    )}
                  />
                ))}
                {showCi && ciAvailable && (
                  <Scatter data={points.filter(p => p.bar)} shape={CiBar} isAnimationActive={false} className="pointer-events-none" />
                )}
                <RechartsTooltip cursor={false} isAnimationActive={false} content={<ChartTooltip />} />
                <Scatter
                  data={points}
                  shape={Dot}
                  isAnimationActive={false}
                  className="cursor-pointer"
                  onClick={(p: { payload?: Point }) => p.payload && router.push(`/models/${p.payload.model.yamlId}`)}
                />
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full w-full flex items-center justify-center text-muted-foreground text-sm">
              {isMounted && chart.points.length === 0 ? 'No models match these filters.' : ''}
            </div>
          )}
        </div>

        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-xs text-muted-foreground lg:justify-end lg:pl-[388px] xl:pl-[418px]">
          {(Object.keys(WEIGHT_STYLE) as Weights[]).map(w => (
            <span key={w} className="inline-flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: WEIGHT_STYLE[w].color }} />
              {WEIGHT_STYLE[w].label}
            </span>
          ))}
          <span className="hidden sm:block w-px self-stretch bg-border" />
          {chart.keys.map(k => (
            <span key={k.label} className="inline-flex items-center gap-1.5">
              <KeySwatch kind={k.kind} />
              {k.label}
            </span>
          ))}
          {showCi && ciAvailable && (
            <span className="inline-flex items-center gap-1.5">
              <span aria-hidden className="relative inline-block h-2.5 w-4">
                <span className="absolute inset-x-0 top-1/2 h-px bg-muted-foreground/70" />
                <span className="absolute inset-y-0 left-0 w-px bg-muted-foreground/70" />
                <span className="absolute inset-y-0 right-0 w-px bg-muted-foreground/70" />
              </span>
              90% interval
            </span>
          )}
          {question !== 'cost' && (
            <span
              className="inline-flex items-center gap-1.5 cursor-help"
              title="Index scores are relative to the calibration panel: 100 is the panel average and 15 points is one standard deviation. They are not percentages."
            >
              <span aria-hidden className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full border border-muted-foreground/60 text-[9px] font-semibold leading-none">i</span>
              Index scores: 100 = calibration-panel average
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
