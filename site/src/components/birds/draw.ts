/**
 * One bird drawing from mascotSVG, safe to put many on a page.
 *
 * mascotSVG names its clip paths and gradients after the species and condition
 * ("clip-swift-0"), so two healthy swifts on one page would share ids. Each
 * drawing here gets its own prefix. A drawing next to text that already names
 * the bird is marked decorative, so a screen reader hears the bird once.
 */

import { mascotSVG } from "../../lib/ui/mascot";

let drawn = 0;

/** Prefix every id in an SVG string, and every url(#id) that points at one. */
export function uniqueIds(svg: string, prefix: string): string {
  const ids = [...svg.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]);
  let out = svg;
  for (const id of new Set(ids)) {
    out = out.split(`id="${id}"`).join(`id="${prefix}-${id}"`).split(`url(#${id})`).join(`url(#${prefix}-${id})`);
  }
  return out;
}

export function drawBird(ratio: number, starve: number, opts: { decorative?: boolean; prefix?: string } = {}): string {
  let { svg } = mascotSVG({ ratio, starve });
  if (opts.decorative) svg = svg.replace(/role="img" aria-label="[^"]*"/, 'aria-hidden="true" focusable="false"');
  return uniqueIds(svg, `${opts.prefix ?? "bird"}${++drawn}`);
}
