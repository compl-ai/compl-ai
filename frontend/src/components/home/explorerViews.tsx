'use client';
import { useMemo, useState } from 'react';
import { ReferenceLine } from 'recharts';
import type { ExplorerSafetyGroup, JoinedModelData } from '@/lib/data';
import { type IndexId, INDICES, THETA_MEAN, formatDate, formatTokens, getIndexTheta, scaleTheta } from '@/lib/scores';
import {
  type QuestionId, fmtMonth, fmtParams, fmtUsd,
  indexLabel, ols, paramsFit, indexScore, nameFor, nameOf, releaseOf, shortName, weightsOf,
} from './explorer';
import { type ChartSpec, type Point, ScatterPlot, dateAxis, jitter, logDomain, logTicks, scoreAxis, scoreDomain, scoreTicks } from './ScatterPlot';
import {
  INDEX_COLOR, Key, KeySep, ScaleKey, Segmented, Switch, type ViewProps, ViewFrame, WeightKeys,
} from './explorerUi';

const R = 5;
const f1 = (v: number) => v.toFixed(1);
const f2 = (v: number) => v.toFixed(2);
const signed = (v: number) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${f1(Math.abs(v))}`;

const INDEX_OPTIONS = INDICES.map(i => ({ id: i.id as IndexId, label: i.label }));

function point(m: JoinedModelData, x: number, y: number, color: string, extra: [string, string][] = []): Point {
  return { id: m.id, name: nameOf(m), x, y, r: R, color, hollow: weightsOf(m) === 'proprietary', model: m, extra };
}

const scaledInterval = (m: JoinedModelData, idx: IndexId): [number, number] | undefined => {
  const t = getIndexTheta(m, idx);
  return t ? [scaleTheta(t.interval[0]), scaleTheta(t.interval[1])] : undefined;
};


const trendLine = (key: string, xs: number[], ys: number[], [x0, x1]: [number, number], color = 'var(--color-subtle)') => {
  const { a, b } = ols(xs, ys);
  return <ReferenceLine key={key} segment={[{ x: x0, y: a + b * x0 }, { x: x1, y: a + b * x1 }]} stroke={color} strokeWidth={1.5} strokeDasharray="6 4" ifOverflow="hidden" />;
};

const avgLine = (key: string, axis: 'x' | 'y') => (
  <ReferenceLine key={key} {...{ [axis]: THETA_MEAN }} stroke="var(--color-subtle)" strokeDasharray="2 4" ifOverflow="hidden" />
);

// Horizontal-then-vertical segments through frontier steps, extended to `end`.
function stepLines(key: string, steps: { date: number; score: number }[], end: number, color: string, dashed = false) {
  const lines: React.ReactNode[] = [];
  const props = { stroke: color, strokeWidth: 2, strokeDasharray: dashed ? '5 4' : undefined, ifOverflow: 'hidden' as const };
  steps.forEach((s, i) => {
    const next = steps[i + 1];
    const x1 = next ? next.date : end;
    lines.push(<ReferenceLine key={`${key}h${i}`} segment={[{ x: s.date, y: s.score }, { x: x1, y: s.score }]} {...props} />);
    if (next) lines.push(<ReferenceLine key={`${key}v${i}`} segment={[{ x: next.date, y: s.score }, { x: next.date, y: next.score }]} {...props} />);
  });
  return lines;
}

// 2. Capability against Safety or Reliability.
function CapabilitySafetyView({ e, models, inset }: ViewProps) {
  const [y, setY] = useState<'safety' | 'reliability'>('safety');
  const [ci, setCi] = useState(false);
  const color = INDEX_COLOR[y];
  const stats = e.capability_safety[y];

  const chart = useMemo<ChartSpec>(() => {
    const rows = models.flatMap(m => {
      const c = indexScore(m, 'capability');
      const s = indexScore(m, y);
      return c !== null && s !== null ? [{ m, c, s }] : [];
    });
    // Safety labels come from the generator; for Reliability pick the same mix of leaders and outliers here.
    let labels = new Set(e.capability_safety.labels);
    if (y === 'reliability') {
      const { a, b } = ols(rows.map(r => r.c), rows.map(r => r.s));
      const ids = [
        ...[...rows].sort((p, q) => q.c - p.c).slice(0, 3),
        ...[...rows].sort((p, q) => q.s - p.s).slice(0, 3),
        ...[...rows].sort((p, q) => Math.abs(q.s - a - b * q.c) - Math.abs(p.s - a - b * p.c)),
      ].map(r => r.m.id);
      labels = new Set([...new Set(ids)].slice(0, 10));
    }
    const points = rows.map(({ m, c, s }) => ({
      ...point(m, s, c, color),
      label: labels.has(m.id) ? nameOf(m) : undefined,
      xCi: scaledInterval(m, y),
      yCi: scaledInterval(m, 'capability'),
    }));
    // Labelled points go last so they draw on top.
    points.sort((p, q) => Number(!!p.label) - Number(!!q.label));
    const x = scoreAxis(indexLabel(y), rows.map(r => r.s));
    const yAxis = scoreAxis('Capability', rows.map(r => r.c));
    return {
      points,
      x,
      y: yAxis,
      refs: [avgLine('ax', 'x'), avgLine('ay', 'y'), trendLine('trend', rows.map(r => r.s), rows.map(r => r.c), x.domain)],
    };
  }, [e, models, y, color]);

  return (
    <ViewFrame
      controls={<>
        <Segmented label="Horizontal axis" value={y} onChange={setY} options={[{ id: 'safety', label: 'Safety' }, { id: 'reliability', label: 'Reliability' }]} />
        <Switch label="90% intervals" checked={ci} onChange={setCi} />
      </>}
      status={<>Rank correlation ρ = {f2(stats.rho)} across {stats.n} models, {f2(stats.rho_release)} with release date held fixed. {stats.top10_overlap} of the 10 most capable models are in the {indexLabel(y)} top 10.</>}
      legend={<>
        <WeightKeys color={color} />
        <Key kind="dash">Trend</Key>
        {ci && <Key kind="ci">90% interval</Key>}
        <KeySep />
        <ScaleKey />
      </>}
    >
      <ScatterPlot chart={chart} inset={inset} showCi={ci} className="absolute inset-0" />
    </ViewFrame>
  );
}

// 3. The scored safety areas in two groups by release date: one group's models, every group's trend.
const GROUP_STYLE: Record<ExplorerSafetyGroup['id'], { color: string; dash?: string }> = {
  injection: { color: 'var(--color-lavender-strong)' },
  refusal: { color: 'var(--color-lavender)', dash: '5 4' },
};
const pp = (v: number) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${f1(Math.abs(v))}`;

function SafetyAreasView({ e, byId, inset }: ViewProps) {
  const groups = e.safety_areas.groups;
  const [sel, setSel] = useState(groups[0].id);
  const group = groups.find(g => g.id === sel)!;
  const style = GROUP_STYLE[group.id];

  const chart = useMemo<ChartSpec>(() => {
    const points = group.points.flatMap(p => {
      const m = byId.get(p.id);
      return m ? [{ ...point(m, Date.parse(p.date), p.y, style.color, [[group.name, `${f1(p.y)}%`]]), opacity: 0.75 }] : [];
    });
    const line = (g: ExplorerSafetyGroup, selected: boolean) => {
      const [a, b] = g.trend;
      const st = GROUP_STYLE[g.id];
      return (
        <ReferenceLine
          key={g.id}
          segment={[{ x: Date.parse(a.date), y: a.y }, { x: Date.parse(b.date), y: b.y }]}
          stroke={st.color}
          strokeWidth={selected ? 2.5 : 1.5}
          strokeOpacity={selected ? 1 : 0.45}
          strokeDasharray={st.dash}
          ifOverflow="hidden"
        />
      );
    };
    // Groups share one 0–100% axis so their trends are comparable.
    return {
      points,
      x: dateAxis(points.map(p => p.x), 6),
      y: { label: `${group.name} (%)`, domain: [0, 100], ticks: [0, 20, 40, 60, 80, 100], format: (v: number) => `${v}%` },
      refs: [...groups.filter(g => g.id !== sel).map(g => line(g, false)), line(group, true)],
    };
  }, [groups, sel, group, style, byId]);

  const [t0, t1] = group.trend;
  return (
    <ViewFrame
      controls={<Segmented label="Safety group" value={sel} onChange={setSel} options={groups.map(g => ({ id: g.id, label: g.name }))} />}
      summary={<>
        <span className="font-semibold text-ink">{pp(group.pp_per_year)} pp/yr</span>,
        {' '}{f1(t0.y)}% in {fmtMonth(t0.date)} to {f1(t1.y)}% in {fmtMonth(t1.date)} on trend, {group.n} models
      </>}
      status={<>
        Each group averages its areas: {groups.filter(g => g.members.length > 1).map((g, k) => (
          <span key={g.id}>{k > 0 && '; '}<span className="text-ink">{g.name.toLowerCase()}</span> = {g.members.map(m => m.name.toLowerCase()).join(', ')}</span>
        ))}. Fairness is left out because many older models skipped one of its benchmarks, so their scores are not comparable.
        {' '}Accuracy on our benchmarks, not real-world safety; jailbreak resistance is not scored yet.
      </>}
      legend={<>
        <WeightKeys color={style.color} />
        <Key kind={style.dash ? 'dash' : 'line'} color={style.color}>Trend</Key>
        <Key kind="line" color="var(--color-subtle)">Other group’s trend</Key>
      </>}
    >
      <ScatterPlot chart={chart} inset={inset} className="absolute inset-0" />
    </ViewFrame>
  );
}

// 1. Every model on one index by release date, with the best open-weight and best proprietary score so far.
function OverTimeView({ e, models, byId, inset }: ViewProps) {
  const [ix, setIx] = useState<IndexId>('safety');
  const [ci, setCi] = useState(false);
  const ol = e.open_lag;
  const lag = ol[ix];
  const asOf = Date.parse(ol.as_of);
  const color = INDEX_COLOR[ix];

  const chart = useMemo<ChartSpec>(() => {
    const toSteps = (s: { date: string; score: number }[]) => s.map(p => ({ date: Date.parse(p.date), score: p.score }));
    const stepIds = new Set([...lag.open_steps, ...lag.closed_steps].map(s => s.id));
    const keyIds = new Set([lag.best_open.id, lag.best_closed.id, lag.oldest_unmatched?.id].filter(Boolean));
    const points = models.flatMap(m => {
      const s = indexScore(m, ix);
      const d = releaseOf(m);
      const w = weightsOf(m);
      if (s === null || d === null || w === 'unknown') return [];
      const onStep = stepIds.has(m.id);
      // Every model is labelled; records first, then open models, then the rest, so collisions drop the least informative.
      const priority = keyIds.has(m.id) ? 4 : onStep ? 3 : w === 'open' ? 2 : 1;
      return [{
        ...point(m, d, s, color),
        r: onStep ? R : 4,
        opacity: onStep ? 0.95 : 0.4,
        label: shortName(m),
        labelPriority: priority + s / 1000,
        labelMuted: !onStep,
        yCi: scaledInterval(m, ix),
      }];
    });
    points.sort((p, q) => (p.opacity ?? 1) - (q.opacity ?? 1));
    const yDomain = scoreDomain(points.map(p => p.y));
    return {
      points,
      x: dateAxis(points.map(p => p.x), 6, asOf),
      y: { label: indexLabel(ix), domain: yDomain, ticks: scoreTicks(yDomain) },
      refs: [
        avgLine('ay', 'y'),
        ...stepLines('o', toSteps(lag.open_steps), asOf, color),
        ...stepLines('c', toSteps(lag.closed_steps), asOf, color, true),
        <ReferenceLine key="gap" segment={[{ x: asOf, y: lag.best_open.score }, { x: asOf, y: lag.best_closed.score }]} stroke="var(--color-ink)" strokeWidth={2} ifOverflow="visible" />,
      ],
      notes: [{ x: asOf, y: (lag.best_open.score + lag.best_closed.score) / 2, text: `${f1(lag.gap)} pts`, dy: 4 }],
    };
  }, [models, ix, lag, asOf, color]);

  return (
    <ViewFrame
      controls={<>
        <Segmented label="Index" value={ix} onChange={setIx} options={INDEX_OPTIONS} />
        <Switch label="90% intervals" checked={ci} onChange={setCi} />
      </>}
      status={<>
        Lag: today’s gap ÷ how fast the best open score rose over the past year. Best open: {nameFor(byId, lag.best_open.id)}; best proprietary: {nameFor(byId, lag.best_closed.id)}.
      </>}
      legend={<>
        <WeightKeys color={color} />
        <Key kind="line" color={color}>Best open-weight so far</Key>
        <Key kind="dash" color={color}>Best proprietary so far</Key>
        <Key kind="tick" color="var(--color-ink)">Gap today</Key>
        {ci && <Key kind="ci">90% interval</Key>}
        <KeySep />
        <ScaleKey />
      </>}
    >
      <ScatterPlot chart={chart} inset={inset} showCi={ci} className="absolute inset-0" />
    </ViewFrame>
  );
}

// 4. Capability against total parameters (open-weight models), MoE and dense coloured apart.
const DENSE_COLOR = 'var(--color-ink)';

// Capability ~ log10(total) + years, both regressors centred; returns the per-year slope.

function ParametersView({ e, byId, inset }: ViewProps) {
  const pf = e.params.fit;
  const moeColor = INDEX_COLOR.capability;

  const { chart, perMonth, refDate } = useMemo(() => {
    const { rows, refDate, perYear, dMin, dMax, trends } = paramsFit(e, byId);
    const byCap = [...rows].sort((a, b) => b.capability - a.capability).slice(0, 3);
    const bySize = [...rows].sort((a, b) => b.x - a.x).slice(0, 2);
    const labels = new Set([...byCap, ...bySize].map(r => r.id));
    const domain = logDomain(rows.map(r => r.x));
    const groupTrend = (moe: boolean, color: string) => {
      const t = trends[String(moe) as 'true' | 'false'];
      if (!t) return [];
      const at = (x: number) => t.a + t.b * Math.log10(x);
      return [<ReferenceLine key={`t${moe}`} segment={[{ x: t.x0, y: at(t.x0) }, { x: t.x1, y: at(t.x1) }]} stroke={color} strokeWidth={2} strokeOpacity={0.75} strokeDasharray="6 4" ifOverflow="hidden" />];
    };
    const chart = {
      points: rows.map(r => ({
        ...point(r.m, r.x, r.capability, r.moe ? moeColor : DENSE_COLOR, [
          ['Capability', f1(r.raw)],
          ['At ' + fmtMonth(refDate), f1(r.capability)],
          ['Released', fmtMonth(r.date)],
          ['Total parameters', fmtParams(r.total)],
          ['Active parameters', fmtParams(r.active)],
          ['Architecture', r.moe ? 'Mixture of experts' : 'Dense'],
        ]),
        hollow: false,
        // Paler = older release.
        opacity: dMax > dMin ? 0.35 + (0.65 * (r.date - dMin)) / (dMax - dMin) : 1,
        label: labels.has(r.id) ? nameOf(r.m) : undefined,
      })),
      x: { label: 'Total parameters (log scale)', domain, log: true, ticks: logTicks(domain), format: fmtParams },
      y: scoreAxis(`Capability at ${fmtMonth(refDate)} release`, rows.map(r => r.capability)),
      refs: [avgLine('ay', 'y'), ...groupTrend(false, DENSE_COLOR), ...groupTrend(true, moeColor)],
    } as ChartSpec;
    return { chart, perMonth: perYear / 12, refDate };
  }, [e, byId, moeColor]);

  return (
    <ViewFrame
      status={<>Scores are shifted to a common {fmtMonth(refDate)} release using the overall date trend ({signed(perMonth)} per month at the same size); paler dots are older models. Trends fitted per architecture on {pf.n} open models ({pf.n_moe} MoE).</>}
      legend={<>
        <Key kind="dot" color={moeColor}>Mixture of experts ({e.params.n_points_moe})</Key>
        <Key kind="dot" color={DENSE_COLOR}>Dense ({e.params.n_points - e.params.n_points_moe})</Key>
        <Key kind="dash" color="var(--muted-foreground)">Trend per architecture</Key>
        <Key kind="dot" color="color-mix(in srgb, var(--color-ink) 35%, transparent)">Paler = older</Key>
        <KeySep />
        <ScaleKey />
      </>}
    >
      <ScatterPlot chart={chart} inset={inset} className="absolute inset-0" />
    </ViewFrame>
  );
}

const fmtUsdTick = (v: number) => (v < 10 ? `$${+v.toPrecision(1)}` : fmtUsd(v));

// 6. What one evaluation of each model costs, full suite against CORE, by release date.
const fmtCost = (unit: 'usd' | 'tokens') => (v: number) =>
  v === 0 ? (unit === 'usd' ? '$0' : '0') : unit === 'usd' ? fmtUsd(v) : formatTokens(v);

function EvalCostView({ costs: ec, byId, inset }: ViewProps) {
  const [unit, setUnit] = useState<'usd' | 'tokens'>('usd');
  const brand = INDEX_COLOR.capability;
  const full = 'var(--color-ink)';

  const { chart, shown } = useMemo(() => {
    const fmt = fmtCost(unit);
    const kc = `core_${unit}` as const;
    const kf = `full_${unit}` as const;
    // Priced models only, the same ones the totals and trends cover.
    const rows = ec.rows.flatMap(r => {
      const m = byId.get(r.id);
      return m && r.release && r.core_usd && r.full_usd && r.core_tokens && r.full_tokens
        ? [{ m, date: Date.parse(r.release), r: { id: r.id, core_usd: r.core_usd, full_usd: r.full_usd, core_tokens: r.core_tokens, full_tokens: r.full_tokens } }]
        : [];
    });
    // Label the most expensive full-suite runs; every CORE run sits far below them.
    const labelled = new Set([...rows].sort((a, b) => b.r[kf] - a.r[kf]).slice(0, 10).map(d => d.r.id));
    const extra = (r: (typeof rows)[number]['r']): [string, string][] => [
      ['Full suite', fmt(r[kf])],
      ['CORE run', fmt(r[kc])],
      ['CORE is', `${f1(r[kf] / r[kc])}× cheaper`],
    ];
    // Log scale: a constant ×/year growth is a straight line, and cheap CORE runs stay visible.
    const vals = rows.flatMap(d => [d.r[kc], d.r[kf]]).filter(v => v > 0);
    const floor = Math.min(...vals);
    const clip = (v: number) => Math.max(v, floor);
    const points: Point[] = rows.flatMap(({ m, r, date }) => [
      { ...point(m, date, clip(r[kf]), full, extra(r)), id: `${r.id}:full`, hollow: false, label: labelled.has(r.id) ? shortName(m) : undefined, labelPriority: r[kf] },
      { ...point(m, date, clip(r[kc]), brand, extra(r)), id: `${r.id}:core`, hollow: false },
    ]);
    // Fixed ranges that fit the runs; full decades would leave empty space.
    const domain: [number, number] = unit === 'usd' ? [0.3, 5000] : [3e6, 6e8];
    const ticks = unit === 'usd' ? [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 5000] : [3e6, 1e7, 3e7, 1e8, 3e8, 6e8];
    // The fitted exponential trends behind the "×/year" figures, drawn as short segments so they curve.
    const epoch = Date.parse(ec.trend.epoch);
    const x0 = Math.min(...rows.map(d => d.date));
    const x1 = Math.max(...rows.map(d => d.date));
    const trend = (key: typeof kc | typeof kf, color: string) => {
      const { log10_intercept: a, log10_per_year: b } = ec.trend[key];
      const at = (x: number) => 10 ** (a + (b * (x - epoch)) / (365.25 * 864e5));
      const n = 60;
      const xs = Array.from({ length: n + 1 }, (_, i) => x0 + ((x1 - x0) * i) / n);
      return xs.slice(1).map((x, i) => (
        <ReferenceLine key={`trend:${key}:${i}`} segment={[{ x: xs[i], y: at(xs[i]) }, { x, y: at(x) }]} stroke={color} strokeWidth={2} strokeOpacity={0.75} ifOverflow="hidden" />
      ));
    };
    return {
      chart: {
        points,
        x: dateAxis(rows.map(d => d.date)),
        y: { label: unit === 'usd' ? 'Cost per model (USD, log scale)' : 'Tokens per model (log scale)', domain, log: true, ticks, format: unit === 'usd' ? fmtUsdTick : formatTokens, width: 64 },
        refs: [
          ...rows.map(({ r, date }) => (
            <ReferenceLine key={r.id} segment={[{ x: date, y: clip(r[kc]) }, { x: date, y: clip(r[kf]) }]} stroke="var(--color-subtle)" strokeOpacity={0.4} strokeWidth={1.5} strokeDasharray="1.5 3" strokeLinecap="round" ifOverflow="hidden" />
          )),
          ...trend(kf, full),
          ...trend(kc, brand),
        ],
      } as ChartSpec,
      shown: rows.length,
    };
  }, [ec, byId, unit, brand, full]);

  const t = ec.summary.totals;
  return (
    <ViewFrame
      controls={<Segmented label="Measure" value={unit} onChange={setUnit} options={[{ id: 'usd', label: 'Cost' }, { id: 'tokens', label: 'Tokens' }]} />}
      status={<>
        {shown} models. All runs together: {unit === 'usd' ? `${fmtUsd(t.full_usd)} for the full suite, ${fmtUsd(t.core_usd)} for CORE.` : `${formatTokens(t.full_tokens)} tokens for the full suite, ${formatTokens(t.core_tokens)} for CORE.`}
        {unit === 'usd' ? ` With model size held fixed (${ec.size_control.n} models of known size), the full-suite cost still rises about ${Math.round(ec.size_control.x_per_year)}× a year.${ec.rates_date ? ` List prices as of ${formatDate(ec.rates_date)}.` : ''}` : ''}
      </>}
      legend={<>
        <Key kind="dot" color={full}>Full suite</Key>
        <Key kind="dot" color={brand}>CORE</Key>
        <Key kind="line" color="var(--muted-foreground)">Exponential trend</Key>
      </>}
    >
      <ScatterPlot chart={chart} inset={inset} className="absolute inset-0" />
    </ViewFrame>
  );
}

// 5. Score above or below what Capability and release date predict, by lab.
// Fixed per lab so a lab keeps its colour when the index changes; mid-tones that read in light and dark mode.
const LAB_COLOR: Record<string, string> = {
  OpenAI: '#2F9E6E',
  Anthropic: '#D9773A',
  Google: '#3B82D6',
  Meta: '#6C63D9',
  NVIDIA: '#7BAF2A',
  'Alibaba Cloud': '#E0A400',
  'Mistral AI': '#D9485F',
  DeepSeek: '#2BA3B8',
  Xiaomi: '#C25FB0',
  MiniMax: '#8C6A4F',
};
const labColor = (lab: string) => LAB_COLOR[lab] ?? 'var(--color-subtle)';

function LabsView({ e, byId, inset }: ViewProps) {
  const [ix, setIx] = useState<'safety' | 'reliability'>('safety');
  const st = e.labs[ix];

  const chart = useMemo<ChartSpec>(() => {
    const labs = [...st.labs].sort((a, b) => b.median - a.median);
    const pos = new Map(labs.map((l, i) => [l.lab, labs.length - 1 - i]));
    const rows = st.rows.flatMap(r => {
      const m = byId.get(r.id);
      return m ? [{ m, ...r }] : [];
    });
    const sorted = [...rows].sort((a, b) => b.resid - a.resid);
    const labels = new Set([...sorted.slice(0, 2), ...sorted.slice(-2)].map(r => r.id));
    const lim = Math.ceil(Math.max(...rows.map(r => Math.abs(r.resid))) / 5) * 5;
    const ticksY = labs.map((_, i) => i);
    const labOf = new Map(labs.map(l => [pos.get(l.lab)!, l]));
    return {
      points: rows.map(r => ({
        ...point(r.m, r.resid, pos.get(r.lab)! + jitter(r.id) * 0.5, labColor(r.lab), [
          ['Lab', r.lab === 'Other' ? `Other (${r.m.metadata?.organization ?? 'unknown'})` : r.lab],
          ['Above or below prediction', signed(r.resid)],
        ]),
        label: labels.has(r.id) ? nameOf(r.m) : undefined,
      })),
      x: { label: `${indexLabel(ix)} above or below prediction (points)`, domain: [-lim, lim], ticks: Array.from({ length: (2 * lim) / 5 + 1 }, (_, i) => -lim + i * 5), format: v => signed(v).replace('.0', '') },
      y: {
        label: '',
        domain: [-0.7, labs.length - 0.3],
        ticks: ticksY,
        format: v => { const l = labOf.get(v); return l ? `${l.lab} (${l.n})` : ''; },
        width: 128,
        strong: true,
      },
      refs: [
        <ReferenceLine key="zero" x={0} stroke="var(--color-subtle)" strokeDasharray="2 4" />,
        ...labs.map(l => (
          <ReferenceLine key={l.lab} segment={[{ x: l.median, y: pos.get(l.lab)! - 0.36 }, { x: l.median, y: pos.get(l.lab)! + 0.36 }]} stroke={labColor(l.lab)} strokeWidth={3} />
        )),
      ],
      verticalGrid: true,
    };
  }, [st, byId, ix]);

  return (
    <ViewFrame
      controls={<Segmented label="Index" value={ix} onChange={setIx} options={[{ id: 'safety', label: 'Safety' }, { id: 'reliability', label: 'Reliability' }]} />}
      status={<>Prediction from Capability ({f2(st.per_capability_point)} {indexLabel(ix)} points per Capability point) and release date ({signed(st.per_month)} per month), fitted on {st.n} models. Labs with one model are pooled as Other. Most labs have only 2–8 models: treat as exploratory.</>}
      legend={<>
        <WeightKeys color="var(--color-ink)" />
        <Key kind="tick" color="var(--color-ink)">Lab median, in the lab’s colour</Key>
      </>}
    >
      <ScatterPlot chart={chart} inset={inset} className="absolute inset-0" />
    </ViewFrame>
  );
}

export const VIEWS: Record<QuestionId, (p: ViewProps) => React.ReactNode> = {
  'capability-safety': CapabilitySafetyView,
  'safety-areas': SafetyAreasView,
  'over-time': OverTimeView,
  parameters: ParametersView,
  labs: LabsView,
  'eval-cost': EvalCostView,
};
