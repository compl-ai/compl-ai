const INK = '#051C2C';
const WRONG = '#D5D9DD';
const EMPTY = '#EEF0F2';
const BRAND = 'var(--color-brand)';

const CELL = 5;
const STEP = 6;
const BLOCK_COLS = 6;
const BLOCK_W = BLOCK_COLS * STEP;
const BLOCK_GAP = 6;
const BLOCKS = 3;

const colX = (b: number, c: number) => b * (BLOCK_W + BLOCK_GAP) + c * STEP;
const hash = (a: number, b: number) => ((((a + 1) * 73856093) ^ ((b + 1) * 19349663)) >>> 0) % 1000 / 1000;

const STRIP_ROWS = 3;
const STRIP_TOP = 22;
const SPREAD = [16, 5, 10];
const SELECTED = new Set(['0-0-1', '0-1-4', '0-2-2', '0-0-5', '0-2-0', '1-1-2', '1-2-5', '2-0-3', '2-1-0', '2-2-4']);
const cellKey = (b: number, r: number, c: number) => `${b}-${r}-${c}`;

function stripCells() {
  const cells: { b: number; r: number; c: number; x: number; y: number; selected: boolean; seed: number }[] = [];
  for (let b = 0; b < BLOCKS; b++)
    for (let r = 0; r < STRIP_ROWS; r++)
      for (let c = 0; c < BLOCK_COLS; c++)
        cells.push({ b, r, c, x: colX(b, c), y: STRIP_TOP + r * STEP, selected: SELECTED.has(cellKey(b, r, c)), seed: hash(b * 10 + r, c + 3) });
  return cells;
}

function Glyph({ children, label }: { children: React.ReactNode; label: string }) {
  return (
    <svg viewBox="-2 -2 124 46" className="h-14 w-full" role="img" aria-label={label}>
      {children}
    </svg>
  );
}

function PanelGlyph() {
  const missing = new Set(['1-2', '3-1', '5-2', '6-0']);
  const cells = [];
  for (let m = 0; m < 7; m++)
    for (let b = 0; b < BLOCKS; b++)
      for (let c = 0; c < BLOCK_COLS; c++) {
        const x = colX(b, c);
        const y = m * STEP + 1;
        if (missing.has(`${m}-${b}`)) {
          cells.push(<rect key={`${m}-${b}-${c}`} x={x} y={y} width={CELL} height={CELL} fill="none" stroke={WRONG} strokeWidth={0.6} strokeDasharray="1 1" />);
          continue;
        }
        const ability = 0.85 - m * 0.1;
        const difficulty = c / BLOCK_COLS * 0.7 + b * 0.08;
        const correct = hash(m * 7 + b, c) < ability - difficulty + 0.3;
        cells.push(<rect key={`${m}-${b}-${c}`} x={x} y={y} width={CELL} height={CELL} rx={0.8} fill={correct ? INK : WRONG} fillOpacity={correct ? 0.75 : 1} />);
      }
  return <Glyph label="Matrix of models by items, with some benchmarks not run by some models">{cells}</Glyph>;
}

function IccGlyph() {
  const curve = (alpha: number, beta: number) =>
    Array.from({ length: 41 }, (_, i) => {
      const x = 8 + i * 2.75;
      const theta = (x - 8) / 110 * 6 - 3;
      const p = 1 / (1 + Math.exp(-alpha * (theta - beta)));
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${(38 - p * 34).toFixed(1)}`;
    }).join(' ');
  const betaX = 8 + (0.4 + 3) / 6 * 110;
  return (
    <Glyph label="Item characteristic curves: probability of a correct answer rising with model ability">
      <line x1={8} y1={38} x2={118} y2={38} stroke={WRONG} />
      <line x1={8} y1={4} x2={8} y2={38} stroke={WRONG} />
      <path d={curve(0.8, -0.6)} fill="none" stroke="#AEB4BA" strokeWidth={1.5} />
      <path d={curve(3, 0.4)} fill="none" stroke={BRAND} strokeWidth={2} />
      <line x1={betaX} y1={21} x2={betaX} y2={38} stroke={INK} strokeWidth={0.75} strokeDasharray="2 2" />
      <text x={betaX + 3} y={36} fontSize={8} fontStyle="italic" fill={INK}>β</text>
      <text x={betaX + 9} y={14} fontSize={8} fontStyle="italic" fill={INK}>α</text>
      <text x={0} y={8} fontSize={6} fill="#5B6670">P</text>
      <text x={110} y={45} fontSize={6} fill="#5B6670">θ</text>
    </Glyph>
  );
}

function SpreadBars() {
  return (
    <>
      {SPREAD.map((h, b) => (
        <rect key={b} x={colX(b, 0)} y={18 - h} width={BLOCK_W - 1} height={h} rx={1} fill={BRAND} fillOpacity={0.25} />
      ))}
    </>
  );
}

function SelectionGlyph() {
  return (
    <Glyph label="Items allocated to benchmarks in proportion to how much models differ on them">
      <SpreadBars />
      {stripCells().map(s => (
        <rect key={cellKey(s.b, s.r, s.c)} x={s.x} y={s.y} width={CELL} height={CELL} rx={0.8} fill={s.selected ? BRAND : EMPTY} />
      ))}
    </Glyph>
  );
}

function AnswersGlyph() {
  return (
    <Glyph label="The new model answers only the core items">
      {stripCells().map(s =>
        s.selected ? (
          s.seed < 0.6
            ? <rect key={cellKey(s.b, s.r, s.c)} x={s.x} y={s.y} width={CELL} height={CELL} rx={0.8} fill={INK} fillOpacity={0.8} />
            : <rect key={cellKey(s.b, s.r, s.c)} x={s.x + 0.4} y={s.y + 0.4} width={CELL - 0.8} height={CELL - 0.8} rx={0.8} fill="#FFFFFF" stroke={INK} strokeWidth={0.8} />
        ) : (
          <rect key={cellKey(s.b, s.r, s.c)} x={s.x} y={s.y} width={CELL} height={CELL} fill="none" stroke={WRONG} strokeWidth={0.6} strokeDasharray="1 1" />
        ),
      )}
      <text x={0} y={12} fontSize={7} fill="#5B6670">2,000 of 92,210 items</text>
    </Glyph>
  );
}

function PredictGlyph() {
  return (
    <Glyph label="Predicted probability of a correct answer on every other item">
      {stripCells().map(s =>
        s.selected ? (
          s.seed < 0.6
            ? <rect key={cellKey(s.b, s.r, s.c)} x={s.x} y={s.y} width={CELL} height={CELL} rx={0.8} fill={INK} fillOpacity={0.8} />
            : <rect key={cellKey(s.b, s.r, s.c)} x={s.x + 0.4} y={s.y + 0.4} width={CELL - 0.8} height={CELL - 0.8} rx={0.8} fill="#FFFFFF" stroke={INK} strokeWidth={0.8} />
        ) : (
          <rect key={cellKey(s.b, s.r, s.c)} x={s.x} y={s.y} width={CELL} height={CELL} rx={0.8} fill={BRAND} fillOpacity={0.15 + 0.75 * (1 - s.c / BLOCK_COLS) * (0.6 + 0.4 * s.seed)} />
        ),
      )}
      <text x={0} y={12} fontSize={7} fill="#5B6670">P(correct) on all items</text>
    </Glyph>
  );
}

const INDEX_ROWS = [
  { label: 'C', color: 'var(--color-brand)', mid: 118, lo: 113, hi: 123 },
  { label: 'R', color: 'var(--color-teal-strong)', mid: 106, lo: 98, hi: 114 },
  { label: 'S', color: 'var(--color-lavender-strong)', mid: 94, lo: 89, hi: 100 },
];
const idxX = (v: number) => 14 + (v - 70) / 70 * 104;

function IndicesGlyph() {
  return (
    <Glyph label="Three index scores, each with a 90% interval">
      <line x1={idxX(100)} y1={0} x2={idxX(100)} y2={42} stroke={WRONG} strokeDasharray="2 2" />
      {INDEX_ROWS.map((r, i) => {
        const y = 7 + i * 14;
        return (
          <g key={r.label}>
            <text x={0} y={y + 3} fontSize={8} fontWeight={700} fill={INK}>{r.label}</text>
            <line x1={14} y1={y} x2={118} y2={y} stroke={EMPTY} strokeWidth={4} strokeLinecap="round" />
            <line x1={idxX(r.lo)} y1={y} x2={idxX(r.hi)} y2={y} stroke={r.color} strokeOpacity={0.45} strokeWidth={4} strokeLinecap="round" />
            <circle cx={idxX(r.mid)} cy={y} r={3} fill={r.color} stroke="#FFFFFF" strokeWidth={1} />
          </g>
        );
      })}
    </Glyph>
  );
}

function Tile({ n, title, glyph, children, accent = false }: { n: string; title: string; glyph: React.ReactNode; children: React.ReactNode; accent?: boolean }) {
  return (
    <div className={`flex flex-col rounded-xl border bg-white p-3 ${accent ? 'border-brand/40' : 'border-border'}`}>
      <div className="mb-2 flex items-baseline gap-1.5 text-xs font-bold text-ink">
        <span className={`font-serif ${accent ? 'text-brand-strong' : 'text-muted-foreground'}`}>{n}</span>
        {title}
      </div>
      {glyph}
      <p className="mt-2 text-[11.5px] leading-snug text-ink/70">{children}</p>
    </div>
  );
}

function Arrow() {
  return (
    <div className="flex items-center justify-center text-muted-foreground" aria-hidden>
      <svg viewBox="0 0 16 16" className="h-4 w-4 rotate-90 md:rotate-0">
        <path d="M2 8h11M9 4l4 4-4 4" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

function Lane({ title, note, tone, children }: { title: string; note: string; tone: 'muted' | 'brand'; children: React.ReactNode }) {
  return (
    <div className={`rounded-2xl p-3 sm:p-4 ${tone === 'brand' ? 'bg-brand-soft/60 ring-1 ring-brand/30' : 'bg-[#F4F5F6]'}`}>
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 px-1">
        <div className={`text-xs font-extrabold uppercase tracking-wider ${tone === 'brand' ? 'text-brand-strong' : 'text-ink'}`}>{title}</div>
        <div className="text-[11px] text-muted-foreground">{note}</div>
      </div>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-[1fr_16px_1fr_16px_1fr]">{children}</div>
    </div>
  );
}

export function MethodDiagram() {
  return (
    <div data-figure="method">
      <Lane title="Done once: calibration" note="full-suite runs of 41 open-weight models" tone="muted">
        <Tile n="A1" title="Calibration panel" glyph={<PanelGlyph />}>
          41 models × 23 benchmarks, 92,210 items, about 2.04M scored responses. Not every model ran every benchmark (dashed).
        </Tile>
        <Arrow />
        <Tile n="A2" title="Fit item parameters" glyph={<IccGlyph />}>
          A 2PL IRT model per benchmark gives every item a difficulty β and a discrimination α.
        </Tile>
        <Arrow />
        <Tile n="A3" title="Select the core" glyph={<SelectionGlyph />}>
          2,000 items, allocated to benchmarks by how much models differ on them (bars); seeded random sampling within each.
        </Tile>
      </Lane>

      <div className="flex flex-col items-center py-2">
        <div className="h-3 w-px bg-ink/30" />
        <div className="rounded-full border border-ink/15 bg-white px-3 py-1 text-center text-[11px] font-semibold text-ink">
          Frozen and public: α, β for all 92,210 items · the 2,000-item core list
        </div>
        <svg viewBox="0 0 12 14" className="h-3.5 w-3 text-ink/40" aria-hidden>
          <path d="M6 0v12M2 8l4 4 4-4" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      <Lane title="Every new model: evaluation" note="core run only, about 2% of the items" tone="brand">
        <Tile n="B1" title="Run the core" glyph={<AnswersGlyph />} accent>
          The model answers only the 2,000 core items (filled: correct, hollow: wrong).
        </Tile>
        <Arrow />
        <Tile n="B2" title="Predict the full suite" glyph={<PredictGlyph />} accent>
          Its ability θ per benchmark, with the frozen α and β, gives the chance of a correct answer on the other 90,210 items.
        </Tile>
        <Arrow />
        <Tile n="B3" title="Three indices" glyph={<IndicesGlyph />} accent>
          Benchmarks combine, weighted by how well they separate models, into Capability, Reliability and Safety: 100&nbsp;+&nbsp;15θ, with a 90% interval.
        </Tile>
      </Lane>
    </div>
  );
}
