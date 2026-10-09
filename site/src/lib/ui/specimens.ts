/**
 * Specimen drawer: test images with Pigeon's verdict, its heatmap, the peak it
 * points at, and the real bird boxes, so people can watch the model "lose its
 * eyesight" as they compress it.
 */

import { camPeak } from "../engine/metrics";
import { CAM } from "../engine/constants";
import type { TestImage } from "../engine/testset";

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

export const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

const NAMED: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " " };

/**
 * The test set stores some credits HTML-escaped, as Open Images' source pages do
 * ("R&#x27;lyeh Imaging", "Andy Reago &amp; Chrissy McClarren"). Decode them once,
 * so the name reads as the photographer wrote it; the caller escapes it for HTML.
 */
export function decodeEntities(s: string): string {
  return s.replace(/&(#x[0-9a-f]+|#[0-9]+|[a-z]+);/gi, (whole, e: string) => {
    if (e[0] !== "#") return NAMED[e.toLowerCase()] ?? whole;
    const code = e[1] === "x" || e[1] === "X" ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10);
    return code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : whole;
  });
}

/** A photo's credit line, as text: "Photo: <author>", or Open Images when the author isn't recorded. */
export const creditText = (img: TestImage) => (img.credit.author ? `Photo: ${decodeEntities(img.credit.author)}` : "Photo: Open Images");

/** The same boxes twice: a dark keyline underneath, so the colored line holds its contrast on pale and dark photos alike. */
export const keylined = (rects: string, cls: string) => `<g class="keyline">${rects}</g><g class="${cls}">${rects}</g>`;

/** The heatmap's color, read from the --heat token ("r g b") so the palette stays in tokens.css. */
function heatRGB(): [number, number, number] {
  const raw = getComputedStyle(document.documentElement).getPropertyValue("--heat").trim();
  const rgb = raw.split(/[\s,]+/).map(Number);
  if (rgb.length !== 3 || rgb.some((v) => !Number.isFinite(v))) throw new Error(`the --heat token should be three numbers ("r g b"), got "${raw}"`);
  return rgb as [number, number, number];
}

export class Specimens {
  private tiles: { el: HTMLElement; canvas: HTMLCanvasElement; verdict: HTMLElement; peak: HTMLElement }[] = [];
  private heat = heatRGB();

  constructor(private root: HTMLElement, private images: TestImage[], sheetUrl: (s: number) => string, grid: number, count = 24) {
    const pick = images.slice(0, count);
    root.innerHTML = pick
      .map((img, i) => {
        const col = img.cell % grid, row = Math.floor(img.cell / grid);
        const boxes = img.boxes
          .map(([x0, y0, x1, y1]) => `<rect x="${x0 * 100}" y="${y0 * 100}" width="${(x1 - x0) * 100}" height="${(y1 - y0) * 100}"/>`)
          .join("");
        return `<figure class="specimen" data-i="${i}">
          <div class="frame" style="background-image:url('${sheetUrl(img.sheet)}');background-size:${grid * 100}% ${grid * 100}%;background-position:${(col / (grid - 1)) * 100}% ${(row / (grid - 1)) * 100}%">
            <canvas width="${CAM}" height="${CAM}" aria-hidden="true"></canvas>
            <svg viewBox="0 0 100 100" preserveAspectRatio="none" class="boxes" aria-hidden="true">${boxes ? keylined(boxes, "truth-boxes") : ""}</svg>
            <span class="peak" aria-hidden="true"></span>
          </div>
          <figcaption><span class="truth">${img.label ? "Bird" : img.hard ? "Tricky non-bird" : "No bird"}</span><span class="verdict num"></span></figcaption>
          <a class="credit" href="${esc(img.credit.url)}" target="_blank" rel="noopener">${esc(creditText(img))}<span class="visually-hidden"> (source photo, opens in a new tab)</span></a>
        </figure>`;
      })
      .join("");
    this.tiles = [...root.querySelectorAll<HTMLElement>(".specimen")].map((el) => ({
      el,
      canvas: el.querySelector("canvas")!,
      verdict: el.querySelector(".verdict")!,
      peak: el.querySelector(".peak")!,
    }));
  }

  update(logits: Float32Array, cams: Float32Array, refLogits: Float32Array) {
    this.tiles.forEach((t, i) => {
      const img = this.images[i];
      const p = sigmoid(logits[i]);
      const pred = logits[i] > 0 ? 1 : 0;
      const flipped = pred !== (refLogits[i] > 0 ? 1 : 0);
      t.el.dataset.correct = String(pred === img.label);
      t.el.dataset.flipped = String(flipped);
      t.verdict.textContent = `${pred ? "says bird" : "says no bird"} · ${Math.round((pred ? p : 1 - p) * 100)}%${flipped ? " · changed its mind" : ""}`;
      // heatmap: per-image min-max of the logit map, drawn in the --heat color
      const ctx = t.canvas.getContext("2d")!;
      const off = i * CAM * CAM;
      let lo = Infinity, hi = -Infinity;
      for (let k = 0; k < CAM * CAM; k++) { lo = Math.min(lo, cams[off + k]); hi = Math.max(hi, cams[off + k]); }
      const imgData = ctx.createImageData(CAM, CAM);
      for (let k = 0; k < CAM * CAM; k++) {
        const v = hi > lo ? (cams[off + k] - lo) / (hi - lo) : 0;
        imgData.data.set([...this.heat, Math.round(Math.pow(v, 1.6) * 200)], k * 4);
      }
      ctx.putImageData(imgData, 0, 0);
      const [x, y] = camPeak(cams, i);
      t.peak.style.left = `${x * 100}%`;
      t.peak.style.top = `${y * 100}%`;
    });
  }
}
