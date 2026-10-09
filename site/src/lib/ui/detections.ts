/**
 * The detector's specimen drawer: bird photos from the test set with the true
 * boxes and the boxes YOLOv8n draws, so people can watch it lose birds (or
 * invent them) as they compress it.
 */

import { matchCounts, type Box, type Det } from "../detector/score";
import type { TestImage } from "../engine/testset";

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const rect = (b: ArrayLike<number>, size: number) =>
  `<rect x="${(b[0] / size) * 100}" y="${(b[1] / size) * 100}" width="${((b[2] - b[0]) / size) * 100}" height="${((b[3] - b[1]) / size) * 100}"/>`;

export function tally(dets: Det[], gt: Box[]): string {
  const { found, falseAlarms } = matchCounts(dets, gt);
  return `found ${found} of ${gt.length}${falseAlarms ? ` · ${falseAlarms} false alarm${falseAlarms > 1 ? "s" : ""}` : ""}`;
}

export class Detections {
  private tiles: { el: HTMLElement; found: SVGGElement; verdict: HTMLElement; was: HTMLElement; gt: Box[] }[] = [];

  constructor(root: HTMLElement, images: TestImage[], sheetUrl: (s: number) => string, grid: number, private size: number) {
    root.innerHTML = images
      .map((img) => {
        const col = img.cell % grid, row = Math.floor(img.cell / grid);
        const truth = img.boxes.map((b) => rect(b.map((v) => v * size), size)).join("");
        const credit = img.credit.author ? `Photo: ${img.credit.author}` : "Photo: Open Images";
        return `<figure class="specimen">
          <div class="frame" style="background-image:url('${sheetUrl(img.sheet)}');background-size:${grid * 100}% ${grid * 100}%;background-position:${(col / (grid - 1)) * 100}% ${(row / (grid - 1)) * 100}%">
            <svg viewBox="0 0 100 100" preserveAspectRatio="none" class="boxes" aria-hidden="true"><g class="truth-boxes">${truth}</g><g class="found-boxes"></g></svg>
          </div>
          <figcaption><span class="truth">${img.boxes.length} bird${img.boxes.length > 1 ? "s" : ""}</span><span class="verdict num"></span><span class="was num"></span></figcaption>
          <a class="credit" href="${esc(img.credit.url)}" target="_blank" rel="noopener">${esc(credit)}<span class="visually-hidden"> (source photo, opens in a new tab)</span></a>
        </figure>`;
      })
      .join("");
    this.tiles = [...root.querySelectorAll<HTMLElement>(".specimen")].map((el, i) => ({
      el,
      found: el.querySelector<SVGGElement>(".found-boxes")!,
      verdict: el.querySelector<HTMLElement>(".verdict")!,
      was: el.querySelector<HTMLElement>(".was")!,
      gt: images[i].boxes.map((b) => b.map((v) => v * size) as Box),
    }));
  }

  /** Draw each photo's detections; `native` holds the native model's, to say what changed. */
  update(dets: Det[][], native: Det[][]) {
    this.tiles.forEach((t, i) => {
      t.found.innerHTML = dets[i].map((b) => rect(b, this.size)).join("");
      const now = matchCounts(dets[i], t.gt), was = matchCounts(native[i], t.gt);
      t.el.dataset.lost = String(now.found < was.found);
      const changed = now.found !== was.found || now.falseAlarms !== was.falseAlarms;
      t.verdict.textContent = tally(dets[i], t.gt).replace(/^f/, "F");
      t.was.textContent = changed ? `Native: ${tally(native[i], t.gt)}` : "";
    });
  }
}
