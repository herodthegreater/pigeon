/**
 * The mascot is the scoreboard.
 *   species   <- compression ratio (every step is a genuinely lighter real bird)
 *   condition <- accuracy + pointing (healthy -> emaciated -> skeleton)
 *
 * Birds are drawn procedurally so starvation is continuous: `starve` in [0, 1]
 * thins the body, bares the ribs, ruffles the feathers and droops the posture.
 */

export interface Species {
  id: string;
  name: string;
  latin: string;
  grams: number;
  minRatio: number; // compression ratio at which Pigeon becomes this bird
  note: string;
  scale: number;
  body: { l: number; h: number };
  head: number;
  beak: { l: number; h: number };
  wing: { reach: number; shape: "round" | "pointed" | "scythe" | "blur" };
  tail: { kind: "fan" | "pointed" | "fork" | "streamers" | "short"; l: number };
  leg: number;
  c: {
    body: string; belly: string; head: string; wing: string; mark: string; tail: string;
    beak: string; eye: string; ring?: string; leg: string; throat?: string; neckA?: string; neckB?: string;
  };
}

// Plumage colors are each real bird's colors: illustration data for the plate,
// not UI colors, so they live here rather than in tokens.css.
export const SPECIES: Species[] = [
  {
    id: "pigeon", name: "Rock Pigeon", latin: "Columba livia", grams: 350, minRatio: 0,
    note: "Full-precision Pigeon. Well fed. Waddles.",
    scale: 1, body: { l: 58, h: 40 }, head: 14.5, beak: { l: 11, h: 5 },
    wing: { reach: 20, shape: "round" }, tail: { kind: "fan", l: 30 }, leg: 20,
    c: { body: "#8f99a6", belly: "#a7b0bb", head: "#737d8b", wing: "#9aa4b0", mark: "#2f3540", tail: "#6d7582",
      beak: "#3d4047", eye: "#d4572a", ring: "#e9e3d6", leg: "#c8606c", neckA: "#3f9b7f", neckB: "#8e4c8f" },
  },
  {
    id: "dove", name: "Mourning Dove", latin: "Zenaida macroura", grams: 120, minRatio: 1.8,
    note: "Slimmer, with a long pointed tail.",
    scale: 0.9, body: { l: 52, h: 29 }, head: 13, beak: { l: 9, h: 3.5 },
    wing: { reach: 26, shape: "pointed" }, tail: { kind: "pointed", l: 50 }, leg: 16,
    c: { body: "#c4ae93", belly: "#dcc5a8", head: "#bca58a", wing: "#a8998a", mark: "#2a2622", tail: "#9c8f80",
      beak: "#2d2a27", eye: "#1d1b1a", ring: "#8fb4d6", leg: "#d07b83" },
  },
  {
    id: "swift", name: "Common Swift", latin: "Apus apus", grams: 40, minRatio: 3.5,
    note: "Built for speed. Can stay airborne for ten months.",
    scale: 0.78, body: { l: 44, h: 20 }, head: 11, beak: { l: 4, h: 3 },
    wing: { reach: 64, shape: "scythe" }, tail: { kind: "fork", l: 26 }, leg: 6,
    c: { body: "#3b3632", belly: "#46403b", head: "#37322e", wing: "#2c2825", mark: "#2c2825", tail: "#322d2a",
      beak: "#1c1a18", eye: "#0f0e0d", leg: "#2a2622", throat: "#a39d94" },
  },
  {
    id: "swallow", name: "Barn Swallow", latin: "Hirundo rustica", grams: 18, minRatio: 7,
    note: "Fast, agile, forked like a tuning fork.",
    scale: 0.68, body: { l: 40, h: 19 }, head: 11, beak: { l: 5, h: 3 },
    wing: { reach: 42, shape: "pointed" }, tail: { kind: "streamers", l: 58 }, leg: 7,
    c: { body: "#26407a", belly: "#e7bf96", head: "#243d74", wing: "#1d2f5c", mark: "#1d2f5c", tail: "#1f3363",
      beak: "#1a1a1a", eye: "#0e0e0e", leg: "#3a2e2a", throat: "#b24f2e" },
  },
  {
    id: "hummingbird", name: "Ruby-throated Hummingbird", latin: "Archilochus colubris", grams: 3, minRatio: 14,
    note: "Three grams of pure efficiency.",
    scale: 0.5, body: { l: 32, h: 16 }, head: 10, beak: { l: 26, h: 2.2 },
    wing: { reach: 28, shape: "blur" }, tail: { kind: "short", l: 18 }, leg: 5,
    c: { body: "#3f7d4b", belly: "#ecebe0", head: "#3a7446", wing: "#5b6b62", mark: "#5b6b62", tail: "#34503c",
      beak: "#1b1b1b", eye: "#0c0c0c", leg: "#2b2b2b", throat: "#b3203a" },
  },
];

export const DODO = { name: "Dodo", latin: "Raphus cucullatus", note: "You broke the uncompressed model. Impressive." };

export function speciesFor(ratio: number): Species {
  let s = SPECIES[0];
  for (const sp of SPECIES) if (ratio >= sp.minRatio) s = sp;
  return s;
}

export type Condition = "healthy" | "emaciated" | "skeleton";

/** The "still works" line and the collapse line (accuracy / pointing). */
export const WORKS = { accuracy: 0.9, pointing: 0.75 };
export const BROKEN = { accuracy: 0.6, pointing: 0.35 };

/** The "still works" line in words, so every page quotes the same threshold. */
export const WORKS_LINE = `≥ ${Math.round(WORKS.accuracy * 100)}% accuracy and ≥ ${Math.round(WORKS.pointing * 100)}% pointing`;

export function starvation(accuracy: number, pointing: number): number {
  const a = (WORKS.accuracy - accuracy) / (WORKS.accuracy - BROKEN.accuracy);
  const p = (WORKS.pointing - pointing) / (WORKS.pointing - BROKEN.pointing);
  return Math.max(0, Math.min(1, Math.max(a, p)));
}

export function conditionOf(starve: number): Condition {
  return starve <= 0 ? "healthy" : starve >= 1 ? "skeleton" : "emaciated";
}

/* ------------------------------------------------------------ drawing */

const W = 260, H = 190, GROUND = 160;

function mix(a: string, b: string, t: number): string {
  const p = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const [x, y] = [p(a), p(b)];
  return "#" + x.map((v, i) => Math.round(v + (y[i] - v) * t).toString(16).padStart(2, "0")).join("");
}

/** Mirrors --bird-dull in tokens.css. It is mixed into hex colors here, so it can't be a var(). */
const BIRD_DULL = "#8a8176";

/** Pull colours toward a dull grey-brown as the bird starves. */
const dull = (col: string, s: number) => mix(col, BIRD_DULL, 0.45 * s);

function bodyPath(cx: number, cy: number, l: number, h: number, s: number): string {
  const belly = h * (1 - 0.55 * s); // belly sinks in
  const back = h * (1 - 0.2 * s);
  return [
    `M ${cx - l} ${cy - h * 0.05}`,
    `C ${cx - l * 0.6} ${cy - back * 1.05}, ${cx + l * 0.35} ${cy - back * 1.15}, ${cx + l * 0.82} ${cy - h * 0.55}`,
    `C ${cx + l * 1.02} ${cy - h * 0.25}, ${cx + l * 1.0} ${cy + h * 0.35}, ${cx + l * 0.62} ${cy + belly * 0.75}`,
    `C ${cx + l * 0.15} ${cy + belly * 1.05}, ${cx - l * 0.55} ${cy + belly * 0.8}, ${cx - l} ${cy - h * 0.05}`,
    "Z",
  ].join(" ");
}

function tailPath(sp: Species, rx: number, ry: number, droop: number): string {
  const L = sp.tail.l, d = droop;
  switch (sp.tail.kind) {
    case "fan":
      return `M ${rx + 6} ${ry - 7} L ${rx - L} ${ry - 4 + d} L ${rx - L - 3} ${ry + 9 + d} L ${rx + 6} ${ry + 6} Z`;
    case "pointed":
      return `M ${rx + 6} ${ry - 5} L ${rx - L} ${ry + 3 + d} L ${rx + 6} ${ry + 5} Z`;
    case "fork":
      return `M ${rx + 6} ${ry - 4} L ${rx - L} ${ry - 7 + d} L ${rx - L * 0.6} ${ry + 1 + d} L ${rx - L} ${ry + 8 + d} L ${rx + 6} ${ry + 4} Z`;
    case "streamers":
      return `M ${rx + 6} ${ry - 4} L ${rx - L} ${ry - 9 + d} L ${rx - L * 0.35} ${ry + d} L ${rx - L} ${ry + 9 + d} L ${rx + 6} ${ry + 4} Z`;
    case "short":
      return `M ${rx + 5} ${ry - 4} L ${rx - L} ${ry - 2 + d} L ${rx - L} ${ry + 5 + d} L ${rx + 5} ${ry + 4} Z`;
  }
}

function wingPath(sp: Species, cx: number, cy: number, l: number, h: number, s: number): string {
  const sx = cx + l * 0.45, sy = cy - h * 0.55; // shoulder
  const tipX = cx - l - sp.wing.reach, tipY = cy - h * 0.1 + 6 * s;
  const front = (ex: number, ey: number) => ` C ${sx + l * 0.12} ${ey - h * 0.05}, ${sx + l * 0.12} ${sy + h * 0.2}, ${sx} ${sy} Z`;
  return wingOutline(sp, sx, sy, tipX, tipY, cx, cy, l, h).replace(/ Z$/, "") + front(0, cy + h * 0.2);
}

function wingOutline(sp: Species, sx: number, sy: number, tipX: number, tipY: number, cx: number, cy: number, l: number, h: number): string {
  switch (sp.wing.shape) {
    case "scythe":
      return `M ${sx} ${sy} C ${cx} ${sy - 10}, ${tipX + 30} ${tipY - 16}, ${tipX} ${tipY} C ${tipX + 36} ${tipY + 2}, ${cx} ${cy + h * 0.2}, ${sx - 4} ${cy + h * 0.1} Z`;
    case "pointed":
      return `M ${sx} ${sy} C ${cx} ${sy - 6}, ${tipX + 20} ${tipY - 8}, ${tipX} ${tipY} C ${tipX + 26} ${tipY + 6}, ${cx - l * 0.2} ${cy + h * 0.45}, ${sx - 6} ${cy + h * 0.25} Z`;
    case "blur":
      return `M ${sx} ${sy} C ${cx} ${sy - 30}, ${tipX} ${tipY - 34}, ${tipX - 4} ${tipY - 16} C ${tipX + 14} ${tipY - 2}, ${cx} ${cy}, ${sx - 4} ${cy + h * 0.1} Z`;
    default:
      return `M ${sx} ${sy} C ${cx} ${sy - 8}, ${tipX + 10} ${tipY - 12}, ${tipX} ${tipY} C ${tipX + 14} ${tipY + 12}, ${cx - l * 0.2} ${cy + h * 0.55}, ${sx - 8} ${cy + h * 0.35} Z`;
  }
}

function drawLiving(sp: Species, s: number): string {
  const c = Object.fromEntries(Object.entries(sp.c).map(([k, v]) => [k, v ? dull(v, s) : v])) as Species["c"];
  const l = sp.body.l * (1 - 0.1 * s);
  const h = sp.body.h * (1 - 0.3 * s);
  const cx = 130, cy = GROUND - sp.leg - h * 0.8;
  const droop = 10 * s;
  const hx = cx + l * 0.95, hy = cy - h * 0.9 - sp.head * 0.4 + droop * 0.6; // head
  const hr = sp.head * (1 - 0.08 * s);
  const rx = cx - l * 0.95, ry = cy - h * 0.05;
  const id = `${sp.id}-${Math.round(s * 100)}`;
  const parts: string[] = [];

  parts.push(`<defs><clipPath id="clip-${id}"><path d="${bodyPath(cx, cy, l, h, s)}"/></clipPath>
    ${c.neckA ? `<linearGradient id="neck-${id}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${c.neckA}"/><stop offset="1" stop-color="${c.neckB}"/></linearGradient>` : ""}</defs>`);
  // tail + legs behind the body
  parts.push(`<path d="${tailPath(sp, rx, ry, droop)}" fill="${c.tail}" stroke="var(--bird-line)" stroke-width="1.2" stroke-linejoin="round"/>`);
  for (const dx of [-6, 6]) {
    const lx = cx + l * 0.15 + dx;
    const top = cy + h * (0.75 - 0.4 * s);
    const legW = sp.leg > 10 ? 3.2 * (1 - 0.45 * s) : 2;
    parts.push(`<path d="M ${lx} ${top} L ${lx + 2} ${GROUND - 2} M ${lx - 5} ${GROUND} L ${lx + 2} ${GROUND - 2} L ${lx + 9} ${GROUND}" stroke="${c.leg}" stroke-width="${legW}" stroke-linecap="round" fill="none"/>`);
  }
  // body + belly
  parts.push(`<path d="${bodyPath(cx, cy, l, h, s)}" fill="${c.body}" stroke="var(--bird-line)" stroke-width="1.4"/>`);
  parts.push(`<g clip-path="url(#clip-${id})"><ellipse cx="${cx + l * 0.35}" cy="${cy + h * 0.75}" rx="${l * 0.8}" ry="${h * 0.7}" fill="${c.belly}"/>`);
  if (s > 0) {
    // ribs show through
    const ribs = 5;
    for (let i = 0; i < ribs; i++) {
      const x = cx - l * 0.15 + i * l * 0.16;
      parts.push(`<path d="M ${x} ${cy - h * 0.35} q ${-6} ${h * 0.55} ${2} ${h * 1.1}" stroke="var(--bird-rib)" stroke-width="1.6" fill="none" opacity="${Math.min(1, s * 1.5)}"/>`);
    }
  }
  parts.push(`</g>`);
  // wing (with marks)
  parts.push(`<path d="${wingPath(sp, cx, cy, l, h, s)}" fill="${c.wing}" stroke="var(--bird-line)" stroke-width="1.2" ${sp.wing.shape === "blur" ? 'opacity="0.55"' : ""}/>`);
  if (sp.id === "pigeon") {
    for (const k of [0, 1]) parts.push(`<path d="M ${cx - l * 0.05 - k * 12} ${cy - h * 0.35} q 8 ${h * 0.3} 2 ${h * 0.6}" stroke="${c.mark}" stroke-width="4.5" fill="none" stroke-linecap="round"/>`);
  }
  if (sp.id === "dove") {
    for (const [dx, dy] of [[0, -4], [-10, 2], [-20, -2], [-6, 8]]) parts.push(`<ellipse cx="${cx + dx}" cy="${cy + dy}" rx="3" ry="2" fill="${c.mark}"/>`);
  }
  // ragged feathers
  if (s > 0.15) {
    const tufts: [number, number, number][] = [[-0.55, -0.92, -35], [-0.1, -1.02, -70], [0.3, -0.98, -110], [-0.3, 0.55, 150], [0.25, 0.62, 120]];
    for (const [fx, fy, ang] of tufts) {
      const x = cx + l * fx, y = cy + h * fy, r = (ang * Math.PI) / 180, len = 7 * s;
      parts.push(`<path d="M ${x} ${y} q ${Math.cos(r) * len * 0.6 + 2} ${Math.sin(r) * len * 0.6} ${Math.cos(r) * len} ${Math.sin(r) * len}" stroke="${c.body}" stroke-width="2.4" stroke-linecap="round" fill="none" opacity="${s}"/>`);
      parts.push(`<path d="M ${x} ${y} q ${Math.cos(r) * len * 0.6 + 2} ${Math.sin(r) * len * 0.6} ${Math.cos(r) * len} ${Math.sin(r) * len}" stroke="var(--bird-line)" stroke-width="0.7" stroke-linecap="round" fill="none" opacity="${s}"/>`);
    }
  }
  // neck + head
  if (c.neckA) parts.push(`<ellipse cx="${hx - hr * 0.6}" cy="${hy + hr * 0.9}" rx="${hr * 0.95}" ry="${hr * 0.75}" fill="url(#neck-${id})" opacity="${1 - 0.6 * s}"/>`);
  if (c.throat) parts.push(`<ellipse cx="${hx + hr * 0.15}" cy="${hy + hr * 0.75}" rx="${hr * 0.6}" ry="${hr * 0.5}" fill="${c.throat}"/>`);
  parts.push(`<circle cx="${hx}" cy="${hy}" r="${hr}" fill="${c.head}" stroke="var(--bird-line)" stroke-width="1.3"/>`);
  if (c.throat) parts.push(`<path d="M ${hx - hr * 0.2} ${hy + hr * 0.55} q ${hr * 0.5} ${hr * 0.6} ${hr * 0.95} ${-hr * 0.1}" fill="${c.throat}"/>`);
  const bx = hx + hr * 0.9, by = hy + hr * 0.05;
  parts.push(`<path d="M ${bx - 1} ${by - sp.beak.h} L ${bx + sp.beak.l} ${by + droop * 0.15} L ${bx - 1} ${by + sp.beak.h} Z" fill="${c.beak}" stroke="var(--bird-line)" stroke-width="0.8" stroke-linejoin="round"/>`);
  if (sp.id === "pigeon") parts.push(`<ellipse cx="${bx + 1}" cy="${by - sp.beak.h + 0.5}" rx="3.2" ry="1.8" fill="var(--bird-beak-cere)"/>`);
  const ex = hx + hr * 0.3, ey = hy - hr * 0.2;
  if (c.ring) parts.push(`<circle cx="${ex}" cy="${ey}" r="${3.6 + s}" fill="${c.ring}"/>`);
  if (s > 0) parts.push(`<circle cx="${ex}" cy="${ey}" r="${3.2 + 2.2 * s}" fill="var(--bird-rib)" opacity="${0.5 * s}"/>`);
  parts.push(`<circle cx="${ex}" cy="${ey}" r="${2.4 * (1 - 0.3 * s)}" fill="${c.eye}"/>`);
  parts.push(`<circle cx="${ex + 0.8}" cy="${ey - 0.8}" r="0.8" fill="var(--plate)" opacity="${1 - s}"/>`);
  return parts.join("");
}

function drawSkeleton(sp: Species, dodo: boolean): string {
  const bone = "var(--bone)", line = "var(--bird-line)";
  const l = dodo ? 56 : sp.body.l * 0.9;
  const h = dodo ? 36 : sp.body.h * 0.75;
  const legL = dodo ? 26 : Math.max(sp.leg, 8);
  const cx = 130, cy = GROUND - legL - h * 0.7;
  const hr = dodo ? 16 : sp.head * 0.9;
  const hx = cx + l * 0.95, hy = cy - h * 1.05 - hr * 0.3;
  const p: string[] = [];
  const st = `stroke="${line}" stroke-width="1.2" stroke-linecap="round"`;
  // tail vertebrae + pygostyle
  for (let i = 0; i < 5; i++) p.push(`<circle cx="${cx - l + 4 - i * 6}" cy="${cy - h * 0.1 + i * 1.2}" r="${2.6 - i * 0.3}" fill="${bone}" ${st}/>`);
  // legs: femur, tibiotarsus, tarsometatarsus, toes
  for (const dx of [-5, 7]) {
    const k = cx + l * 0.1 + dx;
    p.push(`<path d="M ${k} ${cy + h * 0.4} L ${k - 6} ${cy + h * 0.4 + legL * 0.45} L ${k + 3} ${GROUND - 2} M ${k - 4} ${GROUND} L ${k + 3} ${GROUND - 2} L ${k + 10} ${GROUND}" stroke="${bone}" stroke-width="3" fill="none" stroke-linecap="round"/>`);
    p.push(`<path d="M ${k} ${cy + h * 0.4} L ${k - 6} ${cy + h * 0.4 + legL * 0.45} L ${k + 3} ${GROUND - 2}" ${st} fill="none" stroke-width="0.8"/>`);
  }
  // spine
  const spine = Array.from({ length: 9 }, (_, i) => [cx - l * 0.9 + i * l * 0.19, cy - h * 0.55 - Math.sin((i / 8) * Math.PI) * 4]);
  for (const [x, y] of spine) p.push(`<circle cx="${x}" cy="${y}" r="2.6" fill="${bone}" ${st}/>`);
  // ribs + keeled sternum
  for (let i = 0; i < 6; i++) {
    const x = cx - l * 0.4 + i * l * 0.17;
    p.push(`<path d="M ${x} ${cy - h * 0.5} C ${x + 8} ${cy - h * 0.1}, ${x + 8} ${cy + h * 0.3}, ${x + 2} ${cy + h * 0.55}" stroke="${bone}" stroke-width="2.6" fill="none"/>`);
    p.push(`<path d="M ${x} ${cy - h * 0.5} C ${x + 8} ${cy - h * 0.1}, ${x + 8} ${cy + h * 0.3}, ${x + 2} ${cy + h * 0.55}" ${st} fill="none" stroke-width="0.7"/>`);
  }
  p.push(`<path d="M ${cx - l * 0.35} ${cy + h * 0.55} Q ${cx + l * 0.2} ${cy + h * 1.05} ${cx + l * 0.6} ${cy + h * 0.35} L ${cx + l * 0.55} ${cy + h * 0.2} Q ${cx + l * 0.1} ${cy + h * 0.6} ${cx - l * 0.35} ${cy + h * 0.55} Z" fill="${bone}" ${st}/>`);
  // wing bones folded along the body
  p.push(`<path d="M ${cx + l * 0.45} ${cy - h * 0.45} L ${cx - l * 0.1} ${cy - h * 0.15} L ${cx - l * 0.75 - (dodo ? 0 : sp.wing.reach * 0.4)} ${cy - h * 0.3}" stroke="${bone}" stroke-width="3" fill="none" stroke-linecap="round"/>`);
  p.push(`<path d="M ${cx + l * 0.45} ${cy - h * 0.45} L ${cx - l * 0.1} ${cy - h * 0.15} L ${cx - l * 0.75 - (dodo ? 0 : sp.wing.reach * 0.4)} ${cy - h * 0.3}" ${st} fill="none" stroke-width="0.8"/>`);
  // neck vertebrae
  for (let i = 0; i < 5; i++) p.push(`<circle cx="${cx + l * 0.7 + i * (hx - cx - l * 0.7) / 5}" cy="${cy - h * 0.5 + i * (hy - cy + h * 0.5 + hr * 0.6) / 5}" r="2.3" fill="${bone}" ${st}/>`);
  // skull, orbit, beak
  p.push(`<circle cx="${hx}" cy="${hy}" r="${hr}" fill="${bone}" ${st}/>`);
  p.push(`<circle cx="${hx + hr * 0.25}" cy="${hy - hr * 0.1}" r="${hr * 0.42}" fill="var(--bird-line)" opacity="0.85"/>`);
  const bl = dodo ? 26 : sp.beak.l, bh = dodo ? 9 : sp.beak.h;
  p.push(
    dodo
      ? `<path d="M ${hx + hr * 0.8} ${hy - bh} Q ${hx + hr + bl} ${hy - bh - 4} ${hx + hr + bl} ${hy + 4} Q ${hx + hr + bl - 6} ${hy + bh} ${hx + hr * 0.8} ${hy + bh * 0.8} Z" fill="${bone}" ${st}/>`
      : `<path d="M ${hx + hr * 0.85} ${hy - bh} L ${hx + hr + bl} ${hy + 1} L ${hx + hr * 0.85} ${hy + bh} Z" fill="${bone}" ${st}/>`,
  );
  return p.join("");
}

export interface MascotState {
  ratio: number;
  accuracy: number;
  pointing: number;
}

/** A model scored some other way (the Coop's detector, by AP kept) brings its own starvation in [0, 1]. */
export interface StarveState {
  ratio: number;
  starve: number;
}

let halos = 0;

/** A light rim around the bird's silhouette (only visible where --bird-halo is set, i.e. dark mode). */
function haloFilter(id: string, k: number): string {
  return `<filter id="${id}" x="-10%" y="-10%" width="120%" height="120%">
    <feMorphology in="SourceAlpha" operator="dilate" radius="${(1.2 / k).toFixed(2)}" result="grow"/>
    <feFlood style="flood-color: var(--bird-halo, transparent)"/>
    <feComposite in2="grow" operator="in" result="rim"/>
    <feMerge><feMergeNode in="rim"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>`;
}

export function mascotSVG(state: MascotState | StarveState): { svg: string; species: Species; condition: Condition; starve: number; dodo: boolean } {
  const species = speciesFor(state.ratio);
  const starve = "starve" in state ? Math.max(0, Math.min(1, state.starve)) : starvation(state.accuracy, state.pointing);
  const condition = conditionOf(starve);
  const dodo = condition === "skeleton" && state.ratio < 1.3;
  const inner = condition === "skeleton" ? drawSkeleton(species, dodo) : drawLiving(species, starve);
  const k = dodo ? 1 : species.scale;
  const halo = `halo-${++halos}`;
  const svg = `<svg viewBox="22 38 ${W - 30} ${H - 30}" role="img" aria-label="${dodo ? "Dodo" : species.name}, ${condition}">
    <defs>${haloFilter(halo, k)}</defs>
    <ellipse cx="135" cy="${GROUND + 2}" rx="${70 * k}" ry="5" fill="var(--bird-shadow)"/>
    <line x1="30" x2="230" y1="${GROUND + 2}" y2="${GROUND + 2}" stroke="var(--rule)" stroke-width="1"/>
    <g transform="translate(${130 * (1 - k)} ${GROUND * (1 - k)}) scale(${k})" filter="url(#${halo})">${inner}</g>
  </svg>`;
  return { svg, species, condition, starve, dodo };
}
