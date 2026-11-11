'use client';
import { useMemo, useState } from 'react';
import { ReferenceLine } from 'recharts';
import type { JoinedModelData } from '@/lib/data';
import { type IndexId, INDICES, THETA_MEAN, formatTokens, getIndexTheta, scaleTheta } from '@/lib/scores';
import {
  type QuestionId, fmtParams, fmtUsd,
  indexLabel, indexScore, nameFor, nameOf, releaseOf, shortName, weightsOf,
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

function ols(xs: number[], ys: number[]) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0;
  let sxx = 0;
  xs.forEach((x, i) => { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) ** 2; });
  const b = sxx ? sxy / sxx : 0;
  return { a: my - b * mx, b };
}

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

// 1. Capability against Safety or Reliability.
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
      ...point(m, c, s, color),
      label: labels.has(m.id) ? nameOf(m) : undefined,
      xCi: scaledInterval(m, 'capability'),
      yCi: scaledInterval(m, y),
    }));
    // Labelled points go last so they draw on top.
    points.sort((p, q) => Number(!!p.label) - Number(!!q.label));
    const x = scoreAxis('Capability', rows.map(r => r.c));
    const yAxis = scoreAxis(indexLabel(y), rows.map(r => r.s));
    return {
      points,
      x,
      y: yAxis,
      refs: [avgLine('ax', 'x'), avgLine('ay', 'y'), trendLine('trend', rows.map(r => r.c), rows.map(r => r.s), x.domain)],
    };
  }, [e, models, y, color]);

  return (
    <ViewFrame
      controls={<>
        <Segmented label="Vertical axis" value={y} onChange={setY} options={[{ id: 'safety', label: 'Safety' }, { id: 'reliability', label: 'Reliability' }]} />
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

// 2. Which safety areas rise with Capability.
function ManipulationView({ e, byId, inset }: ViewProps) {
  const mp = e.manipulation;
  const [sel, setSel] = useState(mp.bars[0].id);
  const bar = mp.bars.find(b => b.id === sel)!;
  const color = INDEX_COLOR.safety;
  const maxRho = Math.max(0.8, ...mp.bars.map(b => b.rho_release));

  const chart = useMemo<ChartSpec>(() => {
    const rows = (mp.points[sel] ?? []).flatMap(p => {
      const m = byId.get(p.id);
      return m ? [{ m, ...p }] : [];
    });
    const top = [...rows].sort((a, b) => b.y - a.y);
    const labels = new Set([...top.slice(0, 2), ...top.slice(-2)].map(r => r.id));
    const x = scoreAxis('Capability', rows.map(r => r.x));
    return {
      points: rows.map(r => ({ ...point(r.m, r.x, r.y, color, [[bar.name, f1(r.y)]]), label: labels.has(r.id) ? nameOf(r.m) : undefined })),
      x,
      y: scoreAxis(bar.name, rows.map(r => r.y)),
      refs: [avgLine('ay', 'y'), trendLine('trend', rows.map(r => r.x), rows.map(r => r.y), x.domain)],
    };
  }, [mp, sel, byId, bar.name, color]);

  const group = (g: 'attack' | 'behaviour', title: string) => (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
        <span>{title}</span>
        <span className="tabular normal-case tracking-normal font-normal">mean ρ {f2(mp.mean[g])}</span>
      </div>
      <ul className="grid gap-1">
        {mp.bars.filter(b => b.group === g).map(b => (
          <li key={b.id}>
            <button
              type="button"
              onClick={() => setSel(b.id)}
              aria-pressed={b.id === sel}
              className={`w-full cursor-pointer rounded-lg px-2 py-1.5 text-left transition-colors ${b.id === sel ? 'bg-lavender-soft' : 'hover:bg-hover'}`}
            >
              <div className="flex justify-between gap-3 text-xs">
                <span className={b.id === sel ? 'font-semibold text-ink' : 'text-ink'}>{b.name}</span>
                <span className="tabular font-semibold text-ink">{f2(b.rho_release)}</span>
              </div>
              <div className="mt-1 h-2 rounded-full bg-track">
                <div
                  className="h-full rounded-full"
                  style={{ width: `${(Math.max(0, b.rho_release) / maxRho) * 100}%`, backgroundColor: color, opacity: g === 'attack' ? 1 : 0.5 }}
                />
              </div>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );

  return (
    <ViewFrame
      status={<>{bar.name}: ρ = {f2(bar.rho_release)} with Capability, release date held fixed ({f2(bar.rho)} without), across {bar.n} models.</>}
      legend={<>
        <Key kind="bar" color={color}>Resisting attacks</Key>
        <Key kind="bar" color="color-mix(in srgb, var(--color-lavender-strong) 50%, transparent)">Behaving well</Key>
        <Key kind="dash">Trend</Key>
        <KeySep />
        <ScaleKey />
      </>}
      plotClassName="xl:flex-1 xl:min-h-[520px]"
    >
      <div className="flex flex-col gap-4 xl:absolute xl:inset-0" style={{ paddingLeft: inset }}>
        <div>
          <div className="grid gap-x-6 gap-y-4 sm:grid-cols-2">
            {group('attack', 'Resisting attacks')}
            {group('behaviour', 'Behaving well')}
          </div>
          <p className="mt-2 text-[11px] leading-snug text-muted-foreground">Bars: rank correlation with Capability, release date held fixed. Pick a bar to plot its scores.</p>
        </div>
        <div className="relative h-[360px] xl:h-auto xl:min-h-0 xl:flex-1">
          <ScatterPlot chart={chart} className="absolute inset-0" />
        </div>
      </div>
    </ViewFrame>
  );
}

// 3. Every model on one index by release date, with the best open-weight and best proprietary score so far.
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
        <div className="flex basis-full flex-wrap items-baseline gap-x-5 gap-y-1 pt-2 text-xs text-muted-foreground">
          <span className="font-semibold uppercase tracking-[0.08em] text-ink">Open-weight lag</span>
          {INDEX_OPTIONS.map(({ id }) => (
            <button key={id} type="button" onClick={() => setIx(id)} className={`inline-flex items-baseline gap-1.5 ${id === ix ? 'text-ink' : ''}`}>
              {indexLabel(id)}
              <span className={`font-serif leading-none ${id === ix ? 'text-2xl' : 'text-lg'}`} style={{ color: INDEX_COLOR[id] }}>
                ≈ {f1(ol[id].lag_months)}
              </span>
              months
            </button>
          ))}
        </div>
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

// 4. Capability against active or total parameters (open-weight models), MoE and dense coloured apart.
const DENSE_COLOR = 'var(--color-ink)';

function ParametersView({ e, byId, inset }: ViewProps) {
  const [by, setBy] = useState<'active' | 'total'>('active');
  const pf = e.params.fit;
  const moeColor = INDEX_COLOR.capability;

  const chart = useMemo<ChartSpec>(() => {
    const rows = e.params.points.flatMap(p => {
      const m = byId.get(p.id);
      return m ? [{ m, ...p, x: by === 'total' ? p.total : p.active }] : [];
    });
    const byCap = [...rows].sort((a, b) => b.capability - a.capability).slice(0, 3);
    const bySize = [...rows].sort((a, b) => b.x - a.x).slice(0, 2);
    const labels = new Set([...byCap, ...bySize].map(r => r.id));
    const domain = logDomain(rows.map(r => r.x));
    // Trend per architecture, fitted on log10 size so it is straight on the log axis.
    const groupTrend = (moe: boolean, color: string) => {
      const g = rows.filter(r => r.moe === moe);
      if (g.length < 3) return [];
      const { a, b } = ols(g.map(r => Math.log10(r.x)), g.map(r => r.capability));
      const x0 = Math.min(...g.map(r => r.x));
      const x1 = Math.max(...g.map(r => r.x));
      const at = (x: number) => a + b * Math.log10(x);
      return [<ReferenceLine key={`t${moe}`} segment={[{ x: x0, y: at(x0) }, { x: x1, y: at(x1) }]} stroke={color} strokeWidth={2} strokeOpacity={0.75} strokeDasharray="6 4" ifOverflow="hidden" />];
    };
    return {
      points: rows.map(r => ({
        ...point(r.m, r.x, r.capability, r.moe ? moeColor : DENSE_COLOR, [
          ['Total parameters', fmtParams(r.total)],
          ['Active parameters', fmtParams(r.active)],
          ['Architecture', r.moe ? 'Mixture of experts' : 'Dense'],
        ]),
        hollow: false,
        label: labels.has(r.id) ? nameOf(r.m) : undefined,
      })),
      x: { label: by === 'total' ? 'Total parameters (log scale)' : 'Active parameters per token (log scale)', domain, log: true, ticks: logTicks(domain), format: fmtParams },
      y: scoreAxis('Capability', rows.map(r => r.capability)),
      refs: [avgLine('ay', 'y'), ...groupTrend(false, DENSE_COLOR), ...groupTrend(true, moeColor)],
    };
  }, [e, byId, by, moeColor]);

  return (
    <ViewFrame
      controls={<Segmented label="Horizontal axis" value={by} onChange={setBy} options={[{ id: 'active', label: 'Active parameters' }, { id: 'total', label: 'Total parameters' }]} />}
      status={by === 'active'
        ? <>At the same active size, MoE models score higher than dense ones (they are also newer on average). With release date held fixed, 10× more total parameters goes with {signed(pf.per_10x_total)} Capability points; 10× more active parameters with {signed(pf.per_10x_active)} at the same total size ({pf.n} open models).</>
        : <>By total size, MoE and dense models fall on much the same trend. With release date held fixed, 10× more total parameters goes with {signed(pf.per_10x_total)} Capability points; 10× more active parameters with {signed(pf.per_10x_active)} at the same total size ({pf.n} open models).</>}
      legend={<>
        <Key kind="dot" color={moeColor}>Mixture of experts ({e.params.n_points_moe})</Key>
        <Key kind="dot" color={DENSE_COLOR}>Dense ({e.params.n_points - e.params.n_points_moe})</Key>
        <Key kind="dash" color="var(--muted-foreground)">Trend per architecture</Key>
        <KeySep />
        <ScaleKey />
      </>}
    >
      <ScatterPlot chart={chart} inset={inset} className="absolute inset-0" />
    </ViewFrame>
  );
}

// 6. What one evaluation of each model costs, full suite against CORE, by release date.
const fmtCost = (unit: 'usd' | 'tokens') => (v: number) =>
  v === 0 ? (unit === 'usd' ? '$0' : '0') : unit === 'usd' ? fmtUsd(v) : formatTokens(v);

// Log-axis ticks are 1-3-10 steps; show $0.3 and $3 rather than $0.30 and $3.00.
const fmtUsdTick = (v: number) => (v < 10 ? `$${+v.toPrecision(1)}` : fmtUsd(v));

function EvalCostView({ e, byId, inset }: ViewProps) {
  const [unit, setUnit] = useState<'usd' | 'tokens'>('usd');
  const ec = e.eval_cost;
  const brand = INDEX_COLOR.capability;
  const full = 'var(--color-ink)';

  const { chart, shown } = useMemo(() => {
    const fmt = fmtCost(unit);
    const kc = `core_${unit}` as const;
    const kf = `full_${unit}` as const;
    const rows = ec.rows.flatMap(r => {
      const m = byId.get(r.id);
      return m && r.release ? [{ m, r, date: Date.parse(r.release) }] : [];
    });
    // Label the most expensive full-suite runs; every CORE run sits far below them.
    const labelled = new Set([...rows].sort((a, b) => b.r[kf] - a.r[kf]).slice(0, 10).map(d => d.r.id));
    const extra = (r: (typeof rows)[number]['r']): [string, string][] => [
      ['Full suite', fmt(r[kf])],
      ['CORE run', fmt(r[kc])],
      ['CORE is', `${f1(r[kf] / r[kc])}× cheaper`],
    ];
    const points: Point[] = rows.flatMap(({ m, r, date }) => [
      { ...point(m, date, r[kf], full, extra(r)), id: `${r.id}:full`, hollow: false, label: labelled.has(r.id) ? shortName(m) : undefined, labelPriority: r[kf] },
      { ...point(m, date, r[kc], brand, extra(r)), id: `${r.id}:core`, hollow: false },
    ]);
    const domain = logDomain(rows.flatMap(d => [d.r[kc], d.r[kf]]));
    // The fitted exponential trends behind the "×/year" figures, straight on the log axis.
    const epoch = Date.parse(ec.trend_epoch);
    const x0 = Math.min(...rows.map(d => d.date));
    const x1 = Math.max(...rows.map(d => d.date));
    const trend = (key: typeof kc | typeof kf, color: string) => {
      const { log10_intercept: a, log10_per_year: b } = ec.trend[key];
      const at = (x: number) => 10 ** (a + (b * (x - epoch)) / (365.25 * 864e5));
      return <ReferenceLine key={`trend:${key}`} segment={[{ x: x0, y: at(x0) }, { x: x1, y: at(x1) }]} stroke={color} strokeWidth={2} strokeOpacity={0.75} strokeDasharray="6 4" ifOverflow="hidden" />;
    };
    return {
      chart: {
        points,
        x: dateAxis(rows.map(d => d.date)),
        y: { label: unit === 'usd' ? 'Cost per model (USD, log scale)' : 'Tokens per model (log scale)', domain, log: true, ticks: logTicks(domain), format: unit === 'usd' ? fmtUsdTick : formatTokens, width: 64 },
        refs: [
          ...rows.map(({ r, date }) => (
            <ReferenceLine key={r.id} segment={[{ x: date, y: r[kc] }, { x: date, y: r[kf] }]} stroke="var(--color-subtle)" strokeOpacity={0.6} strokeWidth={1.5} ifOverflow="hidden" />
          )),
          trend(kf, full),
          trend(kc, brand),
        ],
      } as ChartSpec,
      shown: rows.length,
    };
  }, [ec, byId, unit, brand, full]);

  const t = ec.totals;
  return (
    <ViewFrame
      controls={<Segmented label="Measure" value={unit} onChange={setUnit} options={[{ id: 'usd', label: 'Cost' }, { id: 'tokens', label: 'Tokens' }]} />}
      status={<>
        {shown} models. All runs together: {unit === 'usd' ? `${fmtUsd(t.full_usd)} for the full suite, ${fmtUsd(t.core_usd)} for CORE.` : `${formatTokens(t.full_tokens)} tokens for the full suite, ${formatTokens(t.core_tokens)} for CORE.`}
        {unit === 'usd' ? ` ${ec.price_note}` : ''}
      </>}
      legend={<>
        <Key kind="dot" color={full}>Full suite</Key>
        <Key kind="dot" color={brand}>CORE</Key>
        <Key kind="dash">Trend</Key>
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
  manipulation: ManipulationView,
  'over-time': OverTimeView,
  parameters: ParametersView,
  labs: LabsView,
  'eval-cost': EvalCostView,
};
