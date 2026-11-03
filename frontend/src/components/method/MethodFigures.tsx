'use client';
import { useSyncExternalStore } from 'react';
import { CartesianGrid, ErrorBar, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import type { MethodCostRow, MethodEciRow, MethodLomoRow } from '@/lib/data';

const AXIS_TICK = { fill: '#5B6670', fontSize: 11 };
const AXIS_LABEL = { fill: '#051C2C', fontSize: 12, fontWeight: 600 };
const GRID = '#EEF0F2';
const DIAGONAL = '#C9CDD2';

const noopSubscribe = () => () => {};
const useMounted = () => useSyncExternalStore(noopSubscribe, () => true, () => false);

const INDEX_STYLE: Record<MethodLomoRow['index'], { label: string; color: string }> = {
  capability: { label: 'Capability', color: 'var(--color-brand)' },
  reliability: { label: 'Reliability', color: 'var(--color-teal-strong)' },
  safety: { label: 'Safety', color: 'var(--color-lavender-strong)' },
};

function TooltipBox({ title, rows }: { title: string; rows: [string, string][] }) {
  return (
    <div className="bg-white border border-border shadow-lg p-3 text-xs rounded-xl min-w-[200px]">
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
    ? <circle cx={cx} cy={cy} r={4} fill={fill} fillOpacity={0.85} stroke="#FFFFFF" strokeWidth={1} />
    : <circle cx={cx} cy={cy} r={4} fill="#FFFFFF" stroke="#051C2C" strokeWidth={1.5} />;
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

type EciLabel = { dx: number; dy: number; anchor: 'start' | 'end' };
type EciPoint = MethodEciRow & { errX: [number, number]; errY: [number, number]; label?: EciLabel & { text: string } };

const ECI_LABELS: Record<string, EciLabel> = {
  'Gemini 4 Argon': { dx: -8, dy: -6, anchor: 'end' },
  'Claude Opus 5.5': { dx: 8, dy: 4, anchor: 'start' },
  'Claude Sonnet 5.5': { dx: -8, dy: 12, anchor: 'end' },
};

const ECI_STYLE = {
  closed: { color: 'var(--color-lavender-strong)', label: 'Proprietary' },
  open: { color: 'var(--color-teal-strong)', label: 'Open weights' },
} as const;
const eciColor = (type: string) => (type === 'closed' ? ECI_STYLE.closed.color : ECI_STYLE.open.color);

function EciDot(props: { cx?: number; cy?: number; payload?: EciPoint }) {
  const { cx, cy, payload: p } = props;
  if (cx === undefined || cy === undefined || !p) return null;
  return (
    <g>
      <circle cx={cx} cy={cy} r={4} fill={eciColor(p.type)} fillOpacity={0.9} stroke="#FFFFFF" strokeWidth={1.25} />
      {p.label && (
        <text
          x={cx + p.label.dx} y={cy + p.label.dy} textAnchor={p.label.anchor} className="pointer-events-none"
          style={{ fontSize: 11, fontWeight: 600, fill: '#051C2C', paintOrder: 'stroke', stroke: '#FFFFFF', strokeWidth: 3 }}
        >
          {p.label.text}
        </text>
      )}
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

export function EciFigure({ rows }: { rows: MethodEciRow[] }) {
  const mounted = useMounted();
  const data: EciPoint[] = rows.map(r => ({
    ...r,
    errX: [r.eci - r.eci_lower, r.eci_upper - r.eci],
    errY: [r.ours - r.ours_lower, r.ours_upper - r.ours],
    label: ECI_LABELS[r.model] ? { text: r.model, ...ECI_LABELS[r.model] } : undefined,
  }));
  return (
    <div>
      <div className="flex flex-wrap gap-4 text-xs text-muted-foreground mb-2">
        {Object.values(ECI_STYLE).map(s => (
          <span key={s.label} className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
      <div className="w-full h-[360px]">
        {mounted && (
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 12, right: 16, bottom: 26, left: 4 }}>
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
              <XAxis
                type="number" dataKey="eci" domain={[100, 180]} ticks={[100, 120, 140, 160, 180]}
                tick={AXIS_TICK} tickLine={false} axisLine={{ stroke: GRID }}
                label={{ value: 'Epoch Capabilities Index (ECI)', position: 'insideBottom', offset: -14, ...AXIS_LABEL }}
              />
              <YAxis
                type="number" dataKey="ours" domain={[60, 145]} ticks={[60, 80, 100, 120, 140]}
                tick={AXIS_TICK} tickLine={false} axisLine={{ stroke: GRID }} width={52}
                label={{ value: 'ComplAI CORE Capability', angle: -90, position: 'insideLeft', offset: 12, ...AXIS_LABEL, textAnchor: 'middle' }}
              />
              <Tooltip cursor={false} isAnimationActive={false} content={<EciTooltip />} />
              <Scatter data={data} shape={EciDot} isAnimationActive={false}>
                <ErrorBar dataKey="errX" direction="x" stroke="#AEB4BA" strokeOpacity={0.7} width={3} />
                <ErrorBar dataKey="errY" direction="y" stroke="#AEB4BA" strokeOpacity={0.7} width={3} />
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}

const COST_MIN = 1;
const COST_MAX = 5000;
const COST_TICKS = [1, 10, 100, 1000];
const logPos = (v: number) => (Math.log10(Math.max(v, COST_MIN) / COST_MIN) / Math.log10(COST_MAX / COST_MIN)) * 100;
const usd = (v: number) => (v >= 100 ? `$${Math.round(v).toLocaleString()}` : `$${v.toFixed(v >= 10 ? 0 : 2)}`);

export function CostFigure({ rows }: { rows: MethodCostRow[] }) {
  return (
    <div className="text-xs" data-figure="cost">
      <div className="flex flex-wrap gap-4 text-muted-foreground mb-3">
        <span className="flex items-center gap-1.5"><span className="w-3 h-2.5 rounded-sm bg-brand" />Core run (2,000 items)</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-2.5 rounded-sm bg-[#D5D9DD]" />Full suite (92,210 items)</span>
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
                <span key={t} className="absolute inset-y-0 border-l border-dashed border-border" style={{ left: `${logPos(t)}%` }} />
              ))}
              <div className="absolute top-0 h-3 rounded-r-sm bg-[#D5D9DD]" style={{ width: `${logPos(r.full)}%` }} />
              <span className="absolute top-0 h-3 leading-3 pl-1.5 text-[10px] text-muted-foreground tabular whitespace-nowrap" style={{ left: `${logPos(r.full)}%` }}>{usd(r.full)}</span>
              <div className="absolute top-3.5 h-3 rounded-r-sm bg-brand" style={{ width: `${logPos(r.core)}%` }} />
              <span className="absolute top-3.5 h-3 leading-3 pl-1.5 text-[10px] text-brand-strong font-semibold tabular whitespace-nowrap" style={{ left: `${logPos(r.core)}%` }}>{usd(r.core)}</span>
            </div>
            <div className="text-right font-bold text-ink tabular">{r.ratio.toFixed(1)}&times;</div>
          </div>
        ))}
        <div />
        <div className="relative h-4 text-muted-foreground tabular">
          {COST_TICKS.map(t => (
            <span key={t} className="absolute -translate-x-1/2 first:translate-x-0" style={{ left: `${logPos(t)}%` }}>${t.toLocaleString()}</span>
          ))}
        </div>
        <div />
      </div>
    </div>
  );
}
