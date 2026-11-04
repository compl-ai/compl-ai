'use client';

import { useEffect, useRef } from 'react';

type Face = 'left' | 'right' | 'bottom';
type Seg = { cmd: string; pts: [number, number][] };

// Paths from public/assets/icon-core-color.svg, mapped onto a 165-unit frame. Each coloured facet is one face of a
// triangular pyramid; the facets run slightly under the dark frame, so the frame is drawn last.
const SHAPES: { d: string; fill: string; face: Face | null }[] = [
  { d: 'M155.32 113.609L70.362 19.216C69.27 18.007 67.594 17.773 66.229 18.67C64.864 19.567 64.397 21.204 65.059 22.686L99.955 101.211C102.606 107.215 108.065 111.582 114.498 112.869L151.538 120.277C151.811 120.316 152.084 120.355 152.357 120.355C153.799 120.355 155.086 119.614 155.827 118.327C156.685 116.807 156.49 114.935 155.32 113.648Z', fill: '#80DDDA', face: 'right' },
  { d: 'M141.791 121.68L110.092 115.325C105.686 114.428 101.047 115.052 97.031 117.079L36.441 147.374C34.921 148.115 34.297 149.753 34.843 151.351C35.31 152.677 36.48 153.496 37.806 153.496C38.079 153.496 38.313 153.496 38.586 153.418L141.908 129.595C144.091 129.088 145.066 127.217 145.027 125.618C145.027 123.981 143.974 122.187 141.791 121.719Z', fill: '#8A90D1', face: 'bottom' },
  { d: 'M95.51 99.456L57.456 13.796C56.482 11.613 54.337 10.287 51.959 10.443C49.581 10.599 47.592 12.159 46.89 14.42L5.678 148.388C5.016 150.571 5.678 152.872 7.433 154.353C8.486 155.25 9.772 155.718 11.059 155.718C12.346 155.718 12.775 155.523 13.593 155.094L89.077 117.352C95.666 114.038 98.512 106.162 95.51 99.417Z', fill: '#FFAB36', face: 'left' },
  { d: 'M44.902 6.895L0.454 151.39C-1.769 158.564 4.626 165.426 11.956 163.75L156.451 130.414C163.703 128.737 166.471 119.926 161.519 114.389L61.472 3.23C56.404 -2.384 47.124 -0.357 44.902 6.895ZM88.726 116.651L13.242 154.392C9.421 156.303 5.133 152.716 6.419 148.583L47.631 14.615C48.957 10.326 54.922 9.936 56.755 14.069L94.808 99.729C97.616 106.084 94.964 113.531 88.726 116.651ZM109.936 116.105L141.635 122.46C145.066 123.162 145.144 128.035 141.713 128.815L38.391 152.638C35.583 153.301 34.18 149.363 36.753 148.076L97.343 117.781C101.203 115.832 105.647 115.247 109.897 116.105ZM100.657 100.899L65.761 22.374C64.631 19.84 67.945 17.695 69.816 19.762L154.774 114.155C156.88 116.495 154.774 120.121 151.733 119.497L114.693 112.089C108.494 110.841 103.269 106.669 100.696 100.899Z', fill: '#051C2C', face: null },
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
  left: [[44.902, 6.895], [0.454, 151.39]],
  bottom: [[11.956, 163.75], [156.451, 130.414]],
  right: [[161.519, 114.389], [61.472, 3.23]],
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
