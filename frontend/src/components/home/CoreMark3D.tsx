'use client';

import { useEffect, useRef } from 'react';

type Face = 'left' | 'right' | 'bottom';
type Seg = { cmd: string; pts: [number, number][] };

// Paths from public/assets/icon-core-color.svg. Each coloured facet is one face of a triangular pyramid.
const SHAPES: { d: string; fill: string; face: Face | null }[] = [
  { d: 'M109.838 116.112C105.615 115.3 101.148 115.869 97.3304 117.818L36.7409 148.113C34.1419 149.412 35.6039 153.311 38.3653 152.661L141.676 128.782C145.087 127.97 145.006 123.097 141.595 122.447L109.919 116.112H109.838Z', fill: '#8A90D1', face: 'bottom' },
  { d: 'M44.8633 6.87259L0.436413 151.361C-1.7565 158.509 4.65981 165.412 11.9695 163.707L156.458 130.326C163.687 128.62 166.448 119.848 161.494 114.325L61.432 3.21773C56.3964 -2.38639 47.1374 -0.355909 44.8633 6.87259ZM88.6404 116.6L13.1878 154.367C9.37052 156.316 5.0659 152.661 6.36541 148.6L47.5435 14.6696C48.843 10.365 54.8532 10.0401 56.64 14.1011L94.7318 99.706C97.5745 106.041 94.8942 113.513 88.6404 116.6ZM109.839 116.112L141.514 122.447C144.925 123.097 145.006 128.051 141.595 128.782L38.2845 152.661C35.4419 153.311 34.0611 149.412 36.6601 148.113L97.2496 117.818C101.148 115.869 105.534 115.3 109.757 116.112H109.839ZM100.58 100.843L65.6554 22.3042C64.5183 19.7864 67.8483 17.5935 69.7163 19.7052L154.672 114.082C156.783 116.437 154.672 120.092 151.585 119.442L114.549 112.051C108.377 110.833 103.097 106.61 100.58 100.843Z', fill: '#051C2C', face: null },
  { d: 'M13.1878 154.366L88.6404 116.6C94.8943 113.513 97.4933 106.041 94.7318 99.706L56.6401 14.1011C54.772 9.9589 48.843 10.365 47.5435 14.6696L6.36543 148.6C5.06593 152.661 9.37054 156.316 13.1878 154.366Z', fill: '#FFAB36', face: 'left' },
  { d: 'M154.672 114.082L69.7164 19.7052C67.8484 17.6747 64.5184 19.7864 65.6555 22.3042L100.58 100.843C103.179 106.61 108.377 110.833 114.549 112.051L151.585 119.442C154.672 120.092 156.702 116.437 154.672 114.082Z', fill: '#80DDDA', face: 'right' },
];

const parse = (d: string): Seg[] => {
  const tokens = d.match(/[MLCZHV]|-?\d*\.?\d+(?:e-?\d+)?/g) ?? [];
  const segs: Seg[] = [];
  let i = 0;
  let last: [number, number] = [0, 0];
  while (i < tokens.length) {
    const cmd = tokens[i++];
    const pts: [number, number][] = [];
    // H and V are stored as L so every point has both coordinates.
    if (cmd === 'H') pts.push([+tokens[i++], last[1]]);
    else if (cmd === 'V') pts.push([last[0], +tokens[i++]]);
    else {
      const n = cmd === 'C' ? 3 : cmd === 'Z' ? 0 : 1;
      for (let k = 0; k < n; k++) { pts.push([+tokens[i], +tokens[i + 1]]); i += 2; }
    }
    if (pts.length) last = pts[pts.length - 1];
    segs.push({ cmd: cmd === 'H' || cmd === 'V' ? 'L' : cmd, pts });
  }
  return segs;
};
const PARSED = SHAPES.map(s => parse(s.d));

// Pyramid: the outer triangle's edges sit at z=0 and the faces rise to an apex where the facets meet.
// Each point is stored in barycentric coordinates of its face's triangle (two outer corners + apex).
type P2 = [number, number];
const EDGES: Record<Face, [P2, P2]> = {
  left: [[44.8633, 6.87259], [0.436413, 151.361]],
  bottom: [[11.9695, 163.707], [156.458, 130.326]],
  right: [[161.494, 114.325], [61.432, 3.21773]],
};
const intersect = ([[x1, y1], [x2, y2]]: [P2, P2], [[x3, y3], [x4, y4]]: [P2, P2]): P2 => {
  const d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  const a = x1 * y2 - y1 * x2, b = x3 * y4 - y3 * x4;
  return [(a * (x3 - x4) - (x1 - x2) * b) / d, (a * (y3 - y4) - (y1 - y2) * b) / d];
};
const TOP = intersect(EDGES.left, EDGES.right);
const BL = intersect(EDGES.left, EDGES.bottom);
const BR = intersect(EDGES.bottom, EDGES.right);
const FACES: Face[] = ['left', 'bottom', 'right'];
const CORNERS: Record<Face, [P2, P2]> = { left: [TOP, BL], bottom: [BL, BR], right: [BR, TOP] };
const area2 = (a: P2, b: P2, c: P2) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
// Apex as weights of the outer corners (top, bottom-left, bottom-right): the logo as drawn, mostly orange.
const APEX: P2 = [0.21 * TOP[0] + 0.27 * BL[0] + 0.52 * BR[0], 0.21 * TOP[1] + 0.27 * BL[1] + 0.52 * BR[1]];
const HEIGHT = 40;

const bary = (face: Face, p: P2) => {
  const [a, b] = CORNERS[face];
  const total = area2(a, b, APEX);
  return [area2(p, b, APEX) / total, area2(a, p, APEX) / total, area2(a, b, p) / total];
};

// Per point: face index, then weights of corner A, corner B and the apex. The apex weight is also the height.
const MODELS = PARSED.map((segs, i) => {
  const cmds = segs.map(s => [s.cmd, s.pts.length] as const);
  const data = new Float64Array(segs.flatMap(s => s.pts.flatMap(([x, y]) => {
    const p: P2 = [x, y];
    // The dark frame spans all faces: a point belongs to the face whose plane is lowest there.
    const face = SHAPES[i].face ?? FACES.reduce((best, f) => (bary(f, p)[2] < bary(best, p)[2] ? f : best));
    return [FACES.indexOf(face), ...bary(face, p)];
  })));
  return { cmds, data };
});

const CENTER: P2 = [82.5, 88];
const CAMERA = 520;
const LIGHT = (() => { const v = [-0.45, -0.7, 1]; const m = Math.hypot(...v); return v.map(c => c / m); })();

const hexToRgb = (hex: string) => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
const INK = hexToRgb('#051C2C');
const RGB = SHAPES.map(s => hexToRgb(s.fill));
// Shading is baked into the fill colour instead of drawing a translucent overlay path per face.
const shadeFill = (i: number, amount: number) =>
  `rgb(${RGB[i].map((c, k) => Math.round(c + (INK[k] - c) * amount)).join(',')})`;

function pose(yaw: number, pitch: number) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const h = HEIGHT, cz = h * 0.3;
  const paths = MODELS.map(({ cmds, data }) => {
    let d = '';
    let j = 0;
    for (const [cmd, n] of cmds) {
      d += cmd;
      for (let k = 0; k < n; k++, j += 4) {
        const [a, b] = CORNERS[FACES[data[j]]];
        const wa = data[j + 1], wb = data[j + 2], wp = data[j + 3];
        const x = wa * a[0] + wb * b[0] + wp * APEX[0] - CENTER[0];
        const y = wa * a[1] + wb * b[1] + wp * APEX[1] - CENTER[1];
        const z = Math.max(0, wp) * h - cz;
        const z1 = z * cy - x * sy;
        const s = CAMERA / (CAMERA - (y * sp + z1 * cp));
        d += `${k ? ' ' : ''}${(CENTER[0] + (x * cy + z * sy) * s).toFixed(1)} ${(CENTER[1] + (y * cp - z1 * sp) * s).toFixed(1)}`;
      }
    }
    return d;
  });
  const fills = SHAPES.map((shape, i) => {
    if (!shape.face) return shape.fill;
    // Face normal from (B - A) x (apex - A), with A and B at z=0.
    const [a, b] = CORNERS[shape.face];
    const ux = b[0] - a[0], uy = b[1] - a[1], vx = APEX[0] - a[0], vy = APEX[1] - a[1];
    let x = uy * h, y = -ux * h, z = ux * vy - uy * vx;
    const sign = Math.sign(z) / Math.hypot(x, y, z);
    x *= sign; y *= sign; z *= sign;
    const z1 = z * cy - x * sy;
    const lambert = (x * cy + z * sy) * LIGHT[0] + (y * cp - z1 * sp) * LIGHT[1] + (y * sp + z1 * cp) * LIGHT[2];
    return shadeFill(i, Math.min(0.26, Math.max(0, (0.95 - lambert) * 0.4)));
  });
  return { paths, fills };
}

const REST = { yaw: -0.18, pitch: 0.16 };
const RESTING = pose(REST.yaw, REST.pitch);
// Largest turn toward the pointer, in radians.
const MAX_YAW = 0.15, MAX_PITCH = 0.06;
// Gentle, wide falloff: eases in over the first RAMP px (so the icon doesn't snap when the pointer is on it),
// holds at PEAK out to FADE px, then fades smoothly to nothing at LIMIT px.
const PEAK = 0.6, RAMP = 400, FADE = 800, LIMIT = 1200;
const smoothstep = (t: number) => { t = Math.min(1, Math.max(0, t)); return t * t * (3 - 2 * t); };
const pull = (dist: number) => PEAK * smoothstep(dist / RAMP) * (1 - smoothstep((dist - FADE) / (LIMIT - FADE)));
// Share of the remaining distance to the target covered per 60Hz frame.
const EASE = 0.12;

export function CoreMark3D({ className }: { className?: string }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const els = useRef<(SVGPathElement | null)[]>([]);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const current = { ...REST };
    const target = { ...REST };
    let frame = 0;
    let last = 0;
    let visible = false;
    const draw = () => {
      const { paths, fills } = pose(current.yaw, current.pitch);
      paths.forEach((d, i) => {
        const el = els.current[i];
        if (!el) return;
        el.setAttribute('d', d);
        if (SHAPES[i].face) el.setAttribute('fill', fills[i]);
      });
    };
    // Runs only while easing toward the target, then stops until the pointer moves again.
    const tick = (now: number) => {
      const k = 1 - (1 - EASE) ** (Math.min(now - (last || now - 16), 100) / 16.7);
      last = now;
      current.yaw += (target.yaw - current.yaw) * k;
      current.pitch += (target.pitch - current.pitch) * k;
      const settled = Math.abs(target.yaw - current.yaw) < 0.0005 && Math.abs(target.pitch - current.pitch) < 0.0005;
      if (settled) Object.assign(current, target);
      draw();
      frame = settled ? 0 : requestAnimationFrame(tick);
    };
    const aim = (yaw: number, pitch: number) => {
      target.yaw = yaw;
      target.pitch = pitch;
      if (!frame) { last = 0; frame = requestAnimationFrame(tick); }
    };
    const onMove = (e: PointerEvent) => {
      if (!visible || e.pointerType !== 'mouse') return;
      const r = svg.getBoundingClientRect();
      const dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
      const dist = Math.hypot(dx, dy);
      if (!dist) return aim(REST.yaw, REST.pitch);
      const m = pull(dist);
      // Turn the apex toward the pointer: positive yaw swings it right, positive pitch lifts it up.
      aim(REST.yaw + MAX_YAW * m * dx / dist, REST.pitch - MAX_PITCH * m * dy / dist);
    };
    const onLeave = () => aim(REST.yaw, REST.pitch);
    const io = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; });
    io.observe(svg);
    window.addEventListener('pointermove', onMove, { passive: true });
    document.documentElement.addEventListener('pointerleave', onLeave);
    return () => {
      io.disconnect();
      cancelAnimationFrame(frame);
      window.removeEventListener('pointermove', onMove);
      document.documentElement.removeEventListener('pointerleave', onLeave);
    };
  }, []);

  return (
    <svg ref={svgRef} viewBox="0 0 165 165" className={className} overflow="visible" aria-hidden>
      {SHAPES.map((s, i) => (
        <path key={i} ref={el => { els.current[i] = el; }} d={RESTING.paths[i]} fill={RESTING.fills[i]} />
      ))}
    </svg>
  );
}
