import { describe, expect, it } from "vitest";
import { conditionOfAttempt, paretoSVG, paretoTable, smallestWorking, type Attempt } from "./pareto";

const full: Attempt = { label: "Full precision", bytes: 3_686_116, accuracy: 0.94, pointing: 0.98 };
const int8: Attempt = { label: "INT8 per channel · all", bytes: 966_254, accuracy: 0.948, pointing: 0.984 };
// Measured in the Coop: SE INT3 + pointwise INT6 + depthwise INT8 keeps 87.2% accuracy and 97.2% pointing.
const starving: Attempt = { label: "mixed INT3/6/8", bytes: 700_000, accuracy: 0.872, pointing: 0.972 };
const mx: Attempt = { label: "FP4 E2M1 MX/32 · all", bytes: 523_000, accuracy: 0.484, pointing: 0.584 };

describe("the run table and chart", () => {
  it("name every run with the bird's three words, never works or broken", () => {
    expect([full, int8, starving, mx].map(conditionOfAttempt)).toEqual(["healthy", "healthy", "emaciated", "skeleton"]);
    const table = paretoTable([full, int8, starving, mx], int8);
    expect(table).toContain('<td class="c-emaciated">Starving');
    expect(table).toContain('<td class="c-skeleton">Skeleton');
    expect(table).toContain('<td class="c-healthy">Healthy');
    expect(table).toContain('<th scope="col">Condition</th>');
    expect(table).not.toMatch(/Broken|>Works/);
  });

  it("give each condition its own mark, so color is never the only cue", () => {
    const svg = paretoSVG([full, starving, mx], null, 560);
    expect(svg.match(/<circle[^>]*fill="var\(--teal\)"/g)).toHaveLength(1); // healthy: filled
    expect(svg.match(/<circle[^>]*fill="var\(--card\)"/g)).toHaveLength(1); // starving: hollow
    expect(svg.match(/<path d="M [^"]*" stroke="var\(--bad\)"/g)).toHaveLength(1); // skeleton: a cross
    expect(svg).toContain("Healthy needs 90% accuracy");
  });

  it("only counts a healthy, actually smaller run as the record", () => {
    expect(smallestWorking([full, int8, starving, mx], full.bytes)).toBe(int8);
    expect(smallestWorking([full, starving, mx], full.bytes)).toBeNull();
    // A healthy run any smaller than the reference is compressed, as it is on the life list; the same size is not.
    const headOnly: Attempt = { label: "INT8 per channel · Head", bytes: full.bytes - 9_000, accuracy: 0.94, pointing: 0.98 };
    const zeroPrune: Attempt = { label: "0% pruned · all", bytes: full.bytes, accuracy: 0.94, pointing: 0.98 };
    expect(smallestWorking([full, zeroPrune, starving], full.bytes)).toBeNull();
    expect(smallestWorking([full, zeroPrune, headOnly, starving], full.bytes)).toBe(headOnly);
  });
});
