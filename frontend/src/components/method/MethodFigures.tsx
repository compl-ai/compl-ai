'use client';
import { useSyncExternalStore } from 'react';
import { CartesianGrid, ErrorBar, ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import type { MethodCostRow, MethodEciRow, MethodLomoRow } from '@/lib/data';

const AXIS_TICK = { fill: 'var(--muted-foreground)', fontSize: 11 };
const AXIS_LABEL = { fill: 'var(--color-ink)', fontSize: 12, fontWeight: 600 };
const GRID = 'var(--color-grid)';
const DIAGONAL = 'var(--color-rule)';

const noopSubscribe = () => () => {};
const useMounted = () => useSyncExternalStore(noopSubscribe, () => true, () => false);

const INDEX_STYLE: Record<MethodLomoRow['index'], { label: string; color: string }> = {
  capability: { label: 'Capability', color: 'var(--color-brand)' },
  reliability: { label: 'Reliability', color: 'var(--color-teal-strong)' },
  safety: { label: 'Safety', color: 'var(--color-lavender-strong)' },
};

function TooltipBox({ title, rows }: { title: string; rows: [string, string][] }) {
  return (
    <div className="bg-card border border-border shadow-lg p-3 text-xs rounded-xl min-w-[200px]">
      <div className="font-bold text-ink text-sm mb-2">{title}</div>
      <div className="flex flex-col gap-1">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-6">
            <span className="text-muted-foreground">{k}</span>
            <span className="tabular font-medium text-ink">{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

const f1 = (v: number) => v.toFixed(1);

type LomoPoint = MethodLomoRow & { err: [number, number] };

function LomoDot(props: { cx?: number; cy?: number; payload?: LomoPoint; fill?: string }) {
  const { cx, cy, payload, fill } = props;
  if (cx === undefined || cy === undefined || !payload) return null;
  return payload.inside
    ? <circle cx={cx} cy={cy} r={4} fill={fill} fillOpacity={0.85} stroke="var(--card)" strokeWidth={1} />
    : <circle cx={cx} cy={cy} r={4} fill="var(--card)" stroke="var(--color-ink)" strokeWidth={1.5} />;
}

function LomoTooltip({ active, payload }: { active?: boolean; payload?: { payload: LomoPoint }[] }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <TooltipBox
      title={d.model}
      rows={[
        ['Full-suite score', f1(d.observed)],
        ['Predicted from core', f1(d.predicted)],
        ['90% interval', `${f1(d.lower)}–${f1(d.upper)}`],
        ['Error', `${d.predicted >= d.observed ? '+' : ''}${f1(d.predicted - d.observed)}`],
      ]}
    />
  );
}

const LOMO_DOMAIN: [number, number] = [45, 130];

export function LomoFigure({ rows }: { rows: MethodLomoRow[] }) {
  const mounted = useMounted();
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
      {(Object.keys(INDEX_STYLE) as MethodLomoRow['index'][]).map(index => {
        const style = INDEX_STYLE[index];
        const data: LomoPoint[] = rows
          .filter(r => r.index === index)
          .map(r => ({ ...r, err: [r.predicted - r.lower, r.upper - r.predicted] }));
        const mae = data.reduce((s, r) => s + Math.abs(r.predicted - r.observed), 0) / data.length;
        const inside = data.filter(r => r.inside).length / data.length;
        return (
          <div key={index} data-figure-panel={index}>
            <div className="text-sm font-bold text-ink flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: style.color }} />
              {style.label}
              <span className="font-normal text-muted-foreground">· {data.length} models</span>
            </div>
            <div className="mt-0.5 mb-1 text-xs text-muted-foreground tabular">
              mean error {mae.toFixed(1)} pts · {Math.round(inside * 100)}% inside 90% interval
            </div>
            <div className="aspect-square w-full max-w-[360px] mx-auto sm:max-w-none">
              {mounted && (
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 8, right: 8, bottom: 22, left: 0 }}>
                    <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
                    <XAxis
                      type="number" dataKey="observed" domain={LOMO_DOMAIN} ticks={[60, 80, 100, 120]}
                      tick={AXIS_TICK} tickLine={false} axisLine={{ stroke: GRID }}
                      label={{ value: 'Full suite', position: 'insideBottom', offset: -12, ...AXIS_LABEL }}
                    />
                    <YAxis
                      type="number" dataKey="predicted" domain={LOMO_DOMAIN} ticks={[60, 80, 100, 120]}
                      tick={AXIS_TICK} tickLine={false} axisLine={{ stroke: GRID }} width={44}
                      label={{ value: 'From core', angle: -90, position: 'insideLeft', offset: 14, ...AXIS_LABEL, textAnchor: 'middle' }}
                    />
                    <ReferenceLine
                      segment={[{ x: LOMO_DOMAIN[0], y: LOMO_DOMAIN[0] }, { x: LOMO_DOMAIN[1], y: LOMO_DOMAIN[1] }]}
                      stroke={DIAGONAL} strokeDasharray="4 4"
                    />
                    <Tooltip cursor={false} isAnimationActive={false} content={<LomoTooltip />} />
                    <Scatter data={data} fill={style.color} shape={LomoDot} isAnimationActive={false}>
                      <ErrorBar dataKey="err" direction="y" stroke={style.color} strokeOpacity={0.5} width={4} />
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

type EciLabel = { dx: number; dy: number; anchor: 'start' | 'end'; mobile?: Omit<EciLabel, 'mobile'> };
type EciPoint = MethodEciRow & { errX: [number, number]; errY: [number, number]; label?: EciLabel & { text: string; cited: boolean } };

// Pixel offsets from each point; offsets past LEADER_MIN get a leader line into empty space.
const ECI_LABELS: Record<string, EciLabel> = {
  'Claude Opus 5.5': { dx: 9, dy: 4, anchor: 'start', mobile: { dx: -4, dy: -29, anchor: 'end' } },
  'Gemini 4 Argon': { dx: 0, dy: -12, anchor: 'end', mobile: { dx: -4, dy: -11, anchor: 'end' } },
  'Claude Sonnet 5.5': { dx: 9, dy: 4, anchor: 'start', mobile: { dx: -9, dy: 5, anchor: 'end' } },
  'Gemini 3.8 Flash': { dx: -9, dy: -6, anchor: 'end', mobile: { dx: -9, dy: 2, anchor: 'end' } },
  'Claude Haiku 5.5': { dx: 9, dy: 4, anchor: 'start' },
  'GPT-6 Luna': { dx: 9, dy: 12, anchor: 'start' },
  'Muse Spark Open': { dx: -9, dy: -4, anchor: 'end' },
  'Kimi-K2.6': { dx: -9, dy: -4, anchor: 'end' },
  'GLM-5.3-Flash': { dx: 9, dy: 10, anchor: 'start' },
  'Qwen3.6-27B': { dx: -176, dy: -70, anchor: 'end' },
  'Claude Sonnet 4.5': { dx: -186, dy: -59, anchor: 'end' },
  'Claude Haiku 4.5': { dx: -147, dy: -55, anchor: 'end' },
  'gpt-oss-120b': { dx: 176, dy: 51, anchor: 'start' },
  'DeepSeek-V3.2': { dx: 131, dy: 23, anchor: 'start' },
  'gpt-oss-20b': { dx: 9, dy: 4, anchor: 'start' },
  'Qwen3-8B': { dx: 9, dy: 10, anchor: 'start' },
  'Phi-4': { dx: 9, dy: 8, anchor: 'start' },
  'Llama-3.3-70B-Instruct': { dx: -9, dy: 4, anchor: 'end' },
  'Llama-3.1-8B-Instruct': { dx: 9, dy: 4, anchor: 'start' },
};
const LEADER_MIN = 20;

const ECI_STYLE = {
  closed: { color: 'var(--color-lavender-strong)', label: 'Proprietary' },
  open: { color: 'var(--color-teal-strong)', label: 'Open weights' },
} as const;
const eciColor = (type: string) => (type === 'closed' ? ECI_STYLE.closed.color : ECI_STYLE.open.color);
const ECI_X: [number, number] = [105, 180];
const ECI_Y: [number, number] = [60, 140];

function EciDot(props: { cx?: number; cy?: number; payload?: EciPoint }) {
  const { cx, cy, payload: p } = props;
  if (cx === undefined || cy === undefined || !p) return null;
  const l = p.label;
  const leader = l && Math.hypot(l.dx, l.dy) > LEADER_MIN;
  return (
    <g>
      {leader && (
        <line className="max-md:hidden" x1={cx} y1={cy} x2={cx + l.dx} y2={cy + l.dy - 4} stroke="var(--muted-foreground)" strokeOpacity={0.5} strokeWidth={0.75} />
      )}
      <circle cx={cx} cy={cy} r={l?.cited ? 6 : 5} fill={eciColor(p.type)} stroke={l?.cited ? 'var(--color-ink)' : 'var(--card)'} strokeWidth={1.5} />
      {l && (l.mobile ? [{ ...l, cls: 'max-md:hidden' }, { ...l, ...l.mobile, cls: 'md:hidden' }] : [{ ...l, cls: l.cited ? '' : 'max-md:hidden' }]).map(v => (
        <text
          key={v.cls}
          x={cx + v.dx + (leader ? (v.anchor === 'end' ? -3 : 3) : 0)} y={cy + v.dy} textAnchor={v.anchor} className={`pointer-events-none ${v.cls}`}
          style={l.cited
            ? { fontSize: 12, fontWeight: 700, fill: 'var(--color-ink)', paintOrder: 'stroke', stroke: 'var(--card)', strokeWidth: 3 }
            : { fontSize: 10.5, fontWeight: 500, fill: 'var(--muted-foreground)', paintOrder: 'stroke', stroke: 'var(--card)', strokeWidth: 3 }}
        >
          {l.text}
        </text>
      ))}
    </g>
  );
}

function EciTooltip({ active, payload }: { active?: boolean; payload?: { payload: EciPoint }[] }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <TooltipBox
      title={d.model}
      rows={[
        ['ComplAI CORE Capability', `${f1(d.ours)} (${f1(d.ours_lower)}–${f1(d.ours_upper)})`],
        ['Epoch ECI', `${f1(d.eci)} (${f1(d.eci_lower)}–${f1(d.eci_upper)})`],
      ]}
    />
  );
}

// Least-squares line of ours on ECI, clipped to the plot area.
function fitSegment(rows: MethodEciRow[]) {
  const n = rows.length;
  const mx = rows.reduce((s, r) => s + r.eci, 0) / n;
  const my = rows.reduce((s, r) => s + r.ours, 0) / n;
  const b = rows.reduce((s, r) => s + (r.eci - mx) * (r.ours - my), 0) / rows.reduce((s, r) => s + (r.eci - mx) ** 2, 0);
  const at = (x: number) => my + b * (x - mx);
  const inv = (y: number) => mx + (y - my) / b;
  const x0 = Math.max(ECI_X[0], inv(ECI_Y[0]));
  const x1 = Math.min(ECI_X[1], inv(ECI_Y[1]));
  return [{ x: x0, y: at(x0) }, { x: x1, y: at(x1) }] as const;
}

export function EciFigure({ rows, frontier, cited }: { rows: MethodEciRow[]; frontier: number; cited: string[] }) {
  const mounted = useMounted();
  const data: EciPoint[] = rows.map(r => ({
    ...r,
    errX: [r.eci - r.eci_lower, r.eci_upper - r.eci],
    errY: [r.ours - r.ours_lower, r.ours_upper - r.ours],
    label: ECI_LABELS[r.model] ? { text: r.model, cited: cited.includes(r.model), ...ECI_LABELS[r.model] } : undefined,
  }));
  const byEci = rows.map(r => r.eci).sort((a, b) => b - a);
  const bandFrom = (byEci[frontier - 1] + byEci[frontier]) / 2;
  return (
    <div>
      <div className="flex flex-wrap gap-4 text-xs text-muted-foreground mb-2">
        {Object.values(ECI_STYLE).map(s => (
          <span key={s.label} className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: s.color }} />
            {s.label}
          </span>
        ))}
        <span className="flex items-center gap-1.5">
          <span className="w-4 border-t border-dashed border-muted-foreground" />
          Linear fit
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-2.5 rounded-sm bg-brand/15" />
          Frontier check ({frontier} models)
        </span>
      </div>
      <div className="w-full h-[440px]">
        {mounted && (
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 12, right: 24, bottom: 26, left: 4 }}>
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
              <ReferenceArea
                x1={bandFrom} x2={ECI_X[1]} fill="var(--color-brand)" fillOpacity={0.1} stroke="none"
                label={{ value: `Frontier check · ${frontier} models`, position: 'insideTopLeft', fill: 'var(--color-brand-strong)', fontSize: 11, fontWeight: 600 }}
              />
              <XAxis
                type="number" dataKey="eci" domain={ECI_X} ticks={[110, 120, 130, 140, 150, 160, 170, 180]}
                tick={AXIS_TICK} tickLine={false} axisLine={{ stroke: GRID }}
                label={{ value: 'Epoch Capabilities Index (ECI)', position: 'insideBottom', offset: -14, ...AXIS_LABEL }}
              />
              <YAxis
                type="number" dataKey="ours" domain={ECI_Y} ticks={[60, 80, 100, 120, 140]}
                tick={AXIS_TICK} tickLine={false} axisLine={{ stroke: GRID }} width={52}
                label={{ value: 'ComplAI CORE Capability', angle: -90, position: 'insideLeft', offset: 12, ...AXIS_LABEL, textAnchor: 'middle' }}
              />
              <ReferenceLine segment={fitSegment(rows)} stroke="var(--muted-foreground)" strokeOpacity={0.6} strokeDasharray="5 4" />
              <Tooltip cursor={false} isAnimationActive={false} content={<EciTooltip />} />
              <Scatter data={data} shape={EciDot} isAnimationActive={false}>
                <ErrorBar dataKey="errX" direction="x" stroke="var(--color-subtle)" strokeOpacity={0.35} strokeWidth={1} width={0} />
                <ErrorBar dataKey="errY" direction="y" stroke="var(--color-subtle)" strokeOpacity={0.35} strokeWidth={1} width={0} />
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}

// Linear axis with room past the longest bar for its label; ticks at 1, 2, 2.5 or 5 x 10^k.
function costAxis(max: number) {
  // Smallest nice step that covers the largest bar in at most five intervals, so the axis ends just past it.
  const raw = max / 5;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(f => f * mag).find(v => Math.ceil(max / v) <= 5)!;
  const ticks = Array.from({ length: Math.ceil(max / step) + 1 }, (_, i) => i * step);
  const end = ticks[ticks.length - 1];
  return { ticks, pos: (v: number) => (v / end) * 100 };
}
const usd = (v: number) => (v >= 100 ? `$${Math.round(v).toLocaleString()}` : `$${v.toFixed(v >= 10 ? 0 : 2)}`);

export function CostFigure({ rows, items }: { rows: MethodCostRow[]; items: number }) {
  const { ticks: COST_TICKS, pos } = costAxis(Math.max(...rows.map(r => r.full)));
  return (
    <div className="text-xs" data-figure="cost">
      <div className="flex flex-wrap gap-4 text-muted-foreground mb-3">
        <span className="flex items-center gap-1.5"><span className="w-3 h-2.5 rounded-sm bg-brand" />Core run (2,000 items)</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-2.5 rounded-sm bg-track" />Full suite ({items.toLocaleString('en-US')} items)</span>
      </div>
      <div className="grid grid-cols-[minmax(0,7.5rem)_1fr_3rem] sm:grid-cols-[11rem_1fr_3.5rem] gap-x-3 gap-y-2.5 items-center">
        <div />
        <div />
        <div className="text-right text-muted-foreground">cheaper</div>
        {rows.map(r => (
          <div key={r.model} className="contents">
            <div className="min-w-0">
              <div className="font-semibold text-ink truncate">{r.model}</div>
              <div className="text-muted-foreground truncate">{r.source}</div>
            </div>
            <div className="relative h-7">
              {COST_TICKS.map(t => (
                <span key={t} className="absolute inset-y-0 border-l border-dashed border-border" style={{ left: `${pos(t)}%` }} />
              ))}
              <div className="absolute top-0 h-3 rounded-r-sm bg-track" style={{ width: `max(2px, ${pos(r.full)}%)` }} />
              {pos(r.full) > 85 ? (
                <span className="absolute top-0 h-3 leading-3 pr-1.5 text-[10px] text-foreground tabular whitespace-nowrap -translate-x-full" style={{ left: `${pos(r.full)}%` }}>{usd(r.full)}</span>
              ) : (
                <span className="absolute top-0 h-3 leading-3 pl-1.5 text-[10px] text-muted-foreground tabular whitespace-nowrap" style={{ left: `calc(max(2px, ${pos(r.full)}%))` }}>{usd(r.full)}</span>
              )}
              <div className="absolute top-3.5 h-3 rounded-r-sm bg-brand" style={{ width: `max(2px, ${pos(r.core)}%)` }} />
              <span className="absolute top-3.5 h-3 leading-3 pl-1.5 text-[10px] text-brand-strong font-semibold tabular whitespace-nowrap" style={{ left: `calc(max(2px, ${pos(r.core)}%))` }}>{usd(r.core)}</span>
            </div>
            <div className="text-right font-bold text-ink tabular">{r.ratio.toFixed(1)}&times;</div>
          </div>
        ))}
        <div />
        <div className="relative h-4 text-muted-foreground tabular">
          {COST_TICKS.map(t => (
            <span key={t} className="absolute -translate-x-1/2 first:translate-x-0 last:-translate-x-full" style={{ left: `${pos(t)}%` }}>${t >= 1000 ? `${t / 1000}k` : t}</span>
          ))}
        </div>
        <div />
      </div>
    </div>
  );
}
