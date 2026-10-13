import { describe, expect, it } from "vitest";
import fixture from "./__fixtures__/torch_parity.json";
import { convBits, fpGrid, nScales, quantize, rowsOf, type Clip, type Granularity, type Levels } from "./quant";
import { ap50, detectorCondition, iou, matchCounts, nms, type Box, type Det } from "./score";
import { describe as describeRecipe, keyOf, linkQuery, nativeRecipe, presetOf, PRESETS, recipeFromLink, type DetectorRecipe } from "./recipe";
import { mascotSVG, SPECIES, speciesFor } from "../ui/mascot";

const f32 = (a: number[]) => Float32Array.from(a);

describe("the precision lab matches cockpit/harness/precision.py", () => {
  for (const t of fixture.quant) {
    const [O, I, kh, kw] = t.shape, cols = I * kh * kw, w = f32(t.w);
    for (const c of t.cases) {
      it(`${t.name}: ${c.bits}-bit ${c.levels} ${c.granularity} ${c.clip}`, () => {
        const q = quantize(w, O, cols, c.bits, c.granularity as Granularity, c.levels as Levels, c.clip as Clip);
        const ref = f32(c.out);
        // Exact, except that torch sums a 1-bit row's mean in its own float32 order,
        // which can move that one shared scale by a unit in the last place.
        const ulp = c.bits === 1 ? 2 ** -23 : 0;
        let off = 0;
        for (let i = 0; i < q.length; i++) if (Math.abs(q[i] - ref[i]) > ulp * Math.abs(ref[i])) off++;
        expect(off).toBe(0);
      });
    }
  }

  it("builds the E2M1 grid for 4-bit floats", () => {
    expect(fpGrid(4).map((v) => +(v * 6).toFixed(6))).toEqual([0, 0.5, 1, 1.5, 2, 3, 4, 6]);
  });

  it("leaves native 16-bit weights untouched", () => {
    const w = f32([0.1, -0.2, 0.3]);
    expect(quantize(w, 1, 3, 16, "channel", "int")).toBe(w);
  });

  it("uses at most 2^bits - 1 integer levels per scale row", () => {
    const w = f32(fixture.quant[0].w), [O, I, kh, kw] = fixture.quant[0].shape, cols = I * kh * kw;
    const q = quantize(w, O, cols, 3, "channel", "int");
    for (let o = 0; o < O; o++) expect(new Set(q.subarray(o * cols, (o + 1) * cols)).size).toBeLessThanOrEqual(7);
  });
});

describe("storage accounting", () => {
  it("counts scale rows like the harness", () => {
    expect(nScales(16, 27, "tensor")).toBe(1);
    expect(nScales(16, 27, "channel")).toBe(16);
    expect(nScales(16, 144, 32)).toBe(16 * 5);
    expect(rowsOf(2, 5, 4)).toEqual([[0, 4, 4], [4, 5, 4], [5, 9, 4], [9, 10, 4]]);
  });

  it("charges weights at their bits, scales and biases at 16", () => {
    expect(convBits(16, 3, 9, 16, "channel")).toBe(16 * 27 * 16 + 16 * 16);
    expect(convBits(16, 3, 9, 8, "channel")).toBe(16 * 27 * 8 + 16 * 16 + 16 * 16);
    expect(convBits(16, 3, 9, 4, "tensor")).toBe(16 * 27 * 4 + 16 + 16 * 16);
    expect(convBits(16, 3, 9, 4, 16)).toBe(16 * 27 * 4 + 16 * 2 * 16 + 16 * 16);
  });
});

describe("AP@50 matches the harness's BirdmAPEvaluator", () => {
  for (const c of fixture.ap) {
    it(`${c.gt.length} photo(s), ${c.gt.flat().length} birds`, () => {
      expect(ap50(c.gt as Box[][], c.preds as Det[][])).toBeCloseTo(c.ap, 12);
    });
  }

  it("is 1 for perfect detections and 0 with nothing found", () => {
    const gt: Box[][] = [[[10, 10, 50, 50]], [], [[0, 0, 20, 30], [100, 100, 150, 140]]];
    const perfect = gt.map((g) => g.map((b) => [...b, 0.9] as Det));
    expect(ap50(gt, perfect)).toBe(1);
    expect(ap50(gt, [[], [], []])).toBe(0);
  });

  it("counts a false alarm on a photo with no bird", () => {
    const gt: Box[][] = [[[10, 10, 50, 50]], []];
    expect(ap50(gt, [[[10, 10, 50, 50, 0.5]], [[0, 0, 30, 30, 0.9]]])).toBeCloseTo(0.5, 12);
  });
});

describe("NMS matches torchvision at IoU 0.7", () => {
  for (const c of fixture.nms) {
    it(`${c.boxes.length} boxes`, () => {
      const cands = c.boxes.map((b, i) => [...b, c.scores[i]] as Det);
      const kept = nms(cands.map((d) => [...d] as Det));
      expect(kept.map((k) => cands.findIndex((d) => d.every((v, j) => v === k[j])))).toEqual(c.keep);
    });
  }

  it("computes IoU", () => {
    expect(iou([0, 0, 10, 10], [5, 0, 15, 10])).toBeCloseTo(50 / 150, 12);
    expect(iou([0, 0, 10, 10], [20, 20, 30, 30])).toBe(0);
  });
});

describe("the photo tally", () => {
  it("counts found, missed and false alarms", () => {
    const gt: Box[] = [[0, 0, 10, 10], [20, 20, 40, 40]];
    expect(matchCounts([[0, 0, 10, 10, 0.9], [0, 0, 10, 11, 0.5], [60, 60, 70, 70, 0.4]], gt)).toEqual({ found: 1, missed: 1, falseAlarms: 2 });
  });
});

describe("the detector's bird", () => {
  it("is healthy at 95% of native AP or more", () => {
    expect(detectorCondition(0.95, 1)).toMatchObject({ condition: "healthy", starve: 0 });
    expect(detectorCondition(0.7, 0.64)).toMatchObject({ condition: "healthy", starve: 0 });
  });
  it("is a skeleton at 60% or less, exactly at the line too", () => {
    expect(detectorCondition(0.6, 1).condition).toBe("skeleton");
    expect(detectorCondition(0.384, 0.64).condition).toBe("skeleton");
    expect(detectorCondition(0, 0.64)).toMatchObject({ condition: "skeleton", starve: 1 });
  });
  it("starves in between, thinner as AP falls", () => {
    const a = detectorCondition(0.9, 1), b = detectorCondition(0.7, 1);
    expect(a.condition).toBe("emaciated");
    expect(b.condition).toBe("emaciated");
    expect(b.starve).toBeGreaterThan(a.starve);
  });
  it("draws through mascotSVG with that condition, and Pigeon's birds are unchanged", () => {
    const m = mascotSVG({ ratio: 2, starve: detectorCondition(0.8, 1).starve });
    expect(m.species.id).toBe("dove");
    expect(m.condition).toBe("emaciated");
    expect(mascotSVG({ ratio: 1, starve: 1 }).dodo).toBe(true);
    const p = mascotSVG({ ratio: 4, accuracy: 0.95, pointing: 0.8 });
    expect([p.species.id, p.condition]).toEqual(["swift", "healthy"]);
    expect(speciesFor(14)).toBe(SPECIES[4]);
  });
});

describe("recipes", () => {
  it("key equal recipes alike, whatever the key order", () => {
    const a = { ...nativeRecipe(), bits: 4, masks: { "model.4": 0.2, "model.12": 0.1 } };
    const b = { ...nativeRecipe(), bits: 4, masks: { "model.12": 0.1, "model.4": 0.2, "model.6": 0 } };
    expect(keyOf(a)).toBe(keyOf(b));
    expect(keyOf({ ...nativeRecipe(), granularity: "tensor" })).toBe(keyOf(nativeRecipe())); // 16-bit ignores the scale choice
  });
  it("describe themselves", () => {
    expect(describeRecipe(nativeRecipe())).toBe("Native fp16");
    expect(describeRecipe({ ...nativeRecipe(), bits: 4, granularity: "group", group: 32, levels: "fp" })).toBe("FP4 groups of 32");
    expect(describeRecipe({ ...nativeRecipe(), bits: 8, masks: { "model.12": 0.2, "model.4": 0.1 } })).toBe("INT8 per channel · cut L4 10%, L12 20%");
  });
  it("have unique preset ids", () => {
    expect(new Set(PRESETS.map((p) => p.id)).size).toBe(PRESETS.length);
  });
  it("name every cut, so two recipes never share a label", () => {
    const neck = PRESETS.find((p) => p.id === "neck-20")!.recipe;
    const raised = { ...neck, masks: { ...neck.masks, "model.19": 0.9, "model.21": 0.9 } };
    expect(describeRecipe(neck)).toBe("Native fp16 · cut L12 20%, L15 20%, L16 20%, L18 20%, L19 20%, L21 20%");
    expect(describeRecipe(raised)).toBe("Native fp16 · cut L12 20%, L15 20%, L16 20%, L18 20%, L19 90%, L21 90%");
    expect(keyOf(raised)).not.toBe(keyOf(neck));
  });
  it("key a 1-bit recipe alike on either grid, which 1 bit doesn't use", () => {
    expect(keyOf({ ...nativeRecipe(), bits: 1, levels: "fp" })).toBe(keyOf({ ...nativeRecipe(), bits: 1, levels: "int" }));
    expect(keyOf({ ...nativeRecipe(), bits: 4, levels: "fp" })).not.toBe(keyOf({ ...nativeRecipe(), bits: 4, levels: "int" }));
  });
});

describe("a YOLOv8n recipe in a link", () => {
  const STAGES = ["model.0", "model.4", "model.12", "model.15", "model.16", "model.18", "model.19", "model.21"];
  const read = (q: string) => recipeFromLink(new URLSearchParams(q), STAGES);
  const back = (r: DetectorRecipe) => {
    const got = read(linkQuery(r));
    return got && "recipe" in got ? got.recipe : got;
  };

  it("reopens a preset by its id, and native with no query", () => {
    for (const p of PRESETS) expect(linkQuery(p.recipe)).toBe(p.id === "native" ? "" : `preset=${p.id}`);
    expect(read("preset=fp4-group")).toEqual({ recipe: PRESETS.find((p) => p.id === "fp4-group")!.recipe });
    expect(read("")).toBeNull();
    expect(read("utm_source=x")).toBeNull();
  });

  it("carries precision and every stage's cut, and brings back the same recipe", () => {
    const r: DetectorRecipe = { bits: 6, granularity: "group", group: 64, levels: "fp", masks: { "model.21": 0.9, "model.4": 0.2, "model.16": 0.3, "model.12": 0.1 } };
    expect(linkQuery(r)).toBe("bits=6&scale=group&group=64&grid=fp&cut=L4:20,L12:10,L16:30,L21:90");
    expect(keyOf(back(r) as DetectorRecipe)).toBe(keyOf(r));
    expect(presetOf(back(r) as DetectorRecipe)).toBeNull();
    const cutOnly = { ...nativeRecipe(), masks: { "model.0": 0.5 } };
    expect(linkQuery(cutOnly)).toBe("cut=L0:50");
    expect(keyOf(back(cutOnly) as DetectorRecipe)).toBe(keyOf(cutOnly));
    const oneBit = { ...nativeRecipe(), bits: 1, granularity: "tensor" as const };
    expect(linkQuery(oneBit)).toBe("bits=1&scale=tensor");
    expect(keyOf(back(oneBit) as DetectorRecipe)).toBe(keyOf(oneBit));
  });

  it("clamps every number to what the controls can set", () => {
    expect(read("bits=12&scale=channel&grid=int")).toMatchObject({ recipe: { bits: 8 } });
    expect(read("bits=40")).toMatchObject({ recipe: { bits: 16 } });
    expect(read("bits=0")).toMatchObject({ recipe: { bits: 1 } });
    expect(read("bits=4&scale=group&group=50")).toMatchObject({ recipe: { group: 64 } });
    expect(read("cut=L4:95,L12:14,L16:0")).toMatchObject({ recipe: { masks: { "model.4": 0.9, "model.12": 0.1 } } });
    expect((read("cut=L16:0") as { recipe: DetectorRecipe }).recipe.masks).toEqual({});
  });

  it("names what it couldn't read", () => {
    expect(read("preset=int3")).toEqual({ error: 'there is no preset called "int3"' });
    expect(read("bits=four")).toEqual({ error: '"four" is not a number of bits' });
    expect(read("bits=4&scale=row")).toEqual({ error: '"row" is not a way to share scales' });
    expect(read("bits=4&grid=log")).toEqual({ error: '"log" is not a grid' });
    expect(read("cut=L4")).toEqual({ error: '"L4" is not a cut' });
    expect(read("cut=L22:20")).toEqual({ error: "L22 has no channels to cut" });
  });
});
