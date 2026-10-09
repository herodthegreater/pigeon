import { beforeEach, describe, expect, it } from "vitest";
import { bestBird, bestBySpecies, birdFor, clearLifeList, keptCondition, labBirds, readLifeList, recordSighting, type Sighting } from "./birds";
import { detectorCondition } from "../detector/score";

describe("the bird rule", () => {
  it("is healthy at 95% kept, bones at 60%, starving between", () => {
    expect(keptCondition(1.02)).toEqual({ condition: "healthy", starve: 0 });
    expect(keptCondition(0.95).condition).toBe("healthy");
    expect(keptCondition(0.8).condition).toBe("emaciated");
    expect(keptCondition(0.6)).toEqual({ condition: "skeleton", starve: 1 });
    expect(keptCondition(NaN).condition).toBe("skeleton");
  });

  it("is the detector's rule too", () => {
    expect(detectorCondition(0.634, 0.6408).condition).toBe("healthy"); // INT8 per channel, measured
    expect(detectorCondition(0.1728, 0.6408).condition).toBe("skeleton"); // INT4 per channel, measured
  });

  it("picks species by size ratio and draws a broken same-size model as a dodo", () => {
    expect(birdFor(6.21, 0.99).name).toBe("Common Swift");
    expect(birdFor(7.5, 0.97).name).toBe("Barn Swallow");
    expect(birdFor(1, 0.3).name).toBe("Dodo");
    expect(birdFor(4, 0.3)).toMatchObject({ name: "Common Swift", condition: "skeleton" });
  });
});

describe("lab runs", () => {
  const run = {
    metric: "acc",
    rows: [
      { label: "fp32", acc: 0.9139, kbits: 4481.664, ref: true },
      { label: "4-bit, group 16", acc: 0.9135, kbits: 721.536 },
      { label: "2-bit, group 16", acc: 0.8278, kbits: 443.008 },
      { label: "Bayes-optimal (ceiling)", acc: 0.95, model: false },
    ],
  };

  it("scores every model row against the reference and skips non-models", () => {
    const r = labBirds(run);
    if (!r.bird) throw new Error("expected birds");
    expect(r.reference).toBe("fp32");
    expect(r.birds.map((b) => b.label)).toEqual(["4-bit, group 16", "2-bit, group 16"]);
    expect(r.birds[0]).toMatchObject({ name: "Common Swift", condition: "healthy" });
    expect(r.birds[1].condition).toBe("emaciated");
    expect(r.best?.label).toBe("4-bit, group 16");
  });

  it("says why when there's no bird", () => {
    expect(labBirds({ rows: [], bird: false, bird_reason: "Same size." })).toEqual({ bird: false, reason: "Same size." });
    expect(labBirds({ rows: [{ label: "a", acc: 0.5, kbits: 10 }] }).bird).toBe(false);
  });

  it("prefers the smallest healthy bird, else the least starved", () => {
    const a = birdFor(2, 0.99), b = birdFor(8, 0.97), c = birdFor(16, 0.7);
    expect(bestBird([a, b, c])).toBe(b);
    expect(bestBird([c, birdFor(20, 0.65)])).toBe(c);
  });
});

describe("the life list", () => {
  const mem = new Map<string, string>();
  beforeEach(() => {
    mem.clear();
    globalThis.localStorage = {
      getItem: (k: string) => mem.get(k) ?? null,
      setItem: (k: string, v: string) => void mem.set(k, v),
      removeItem: (k: string) => void mem.delete(k),
    } as Storage;
    globalThis.dispatchEvent ??= () => true;
  });
  const s = (over: Partial<Sighting>): Omit<Sighting, "when"> => ({
    ...birdFor(4, 0.99), model: "lab:ptq", modelName: "Post-training quantization", recipe: "4-bit", href: "/guide/ptq/", ...over,
  });

  it("records, replaces the same recipe, and knows a first healthy species", () => {
    expect(recordSighting(s({}), "2026-10-09T00:00:00Z")).toEqual({ ok: true, firstHealthy: true });
    expect(recordSighting(s({}), "2026-10-09T01:00:00Z")).toEqual({ ok: true, firstHealthy: false });
    expect(readLifeList().sightings).toHaveLength(1);
    recordSighting(s({ recipe: "2-bit", ...birdFor(8, 0.5) }));
    const best = bestBySpecies(readLifeList().sightings);
    expect([...best.keys()]).toEqual(["swift", "swallow"]);
    expect(clearLifeList()).toBe(true);
    expect(readLifeList().sightings).toEqual([]);
  });

  it("reports blocked storage instead of pretending", () => {
    globalThis.localStorage = { getItem: () => { throw new Error("blocked"); } } as unknown as Storage;
    expect(readLifeList().ok).toBe(false);
    expect(recordSighting(s({})).ok).toBe(false);
  });
});
