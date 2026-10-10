import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { birdFor, readLifeList } from "./birds";
import { BROKEN_LINE, birdDrawer, cageHref, hatches, lifeListCount, sight, SPECIES_LADDER, STORAGE_BLOCKED, type CoopSighting } from "./coopbirds";

describe("the Coop's sightings", () => {
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
  const swift = (over: Partial<CoopSighting> = {}): CoopSighting => ({
    bird: birdFor(4, 0.99), model: "yolov8n", modelName: "YOLOv8n", recipe: "INT4 per channel", href: cageHref("/pigeon/coop/yolo/", "preset=int4-channel"), ...over,
  });

  it("puts the recipe's query on the cage's link, and nothing for the reference", () => {
    expect(cageHref("/pigeon/coop/", "preset=mxfp4")).toBe("/pigeon/coop/?preset=mxfp4");
    expect(cageHref("/pigeon/coop/yolo/", "bits=6&scale=channel&grid=int")).toBe("/pigeon/coop/yolo/?bits=6&scale=channel&grid=int");
    expect(cageHref("/pigeon/coop/", "")).toBe("/pigeon/coop/");
  });

  it("hatches a bird only for a model that came out smaller", () => {
    expect(hatches(1)).toBe(false); // a 0% prune, or steps on no layers
    expect(hatches(0.97)).toBe(false); // a pruning bitmap can make it bigger
    expect(hatches(1.004)).toBe(true);
  });

  it("states which size makes which bird, and Pigeon's skeleton line", () => {
    expect(SPECIES_LADDER).toBe("dove from 1.8×, swift from 3.5×, swallow from 7×, hummingbird from 14×");
    expect(BROKEN_LINE).toBe("≤ 60% accuracy or ≤ 35% pointing");
  });

  it("says a species' first healthy sighting once, and nothing after", () => {
    const warned = { blocked: false };
    expect(lifeListCount()).toBe("no birds yet");
    expect(sight(swift(), warned)).toEqual({ said: "First healthy Common Swift on your life list.", announce: true });
    expect(sight(swift({ recipe: "FP4, groups of 32" }), warned)).toEqual({ said: "", announce: false });
    expect(sight(swift({ bird: birdFor(8, 0.7), recipe: "INT2" }), warned)).toEqual({ said: "", announce: false }); // a starving swallow is no first healthy one
    const list = readLifeList().sightings;
    expect(list.map((s) => [s.model, s.modelName, s.recipe, s.href])).toEqual([
      ["yolov8n", "YOLOv8n", "INT4 per channel", "/pigeon/coop/yolo/?preset=int4-channel"],
      ["yolov8n", "YOLOv8n", "FP4, groups of 32", "/pigeon/coop/yolo/?preset=int4-channel"],
      ["yolov8n", "YOLOv8n", "INT2", "/pigeon/coop/yolo/?preset=int4-channel"],
    ]);
    expect(lifeListCount()).toBe("2 species so far");
  });

  it("keeps a sighting's recipe key, rule and pointing, and tells recipes apart by key", () => {
    const warned = { blocked: false };
    sight(swift({ recipe: "Native fp16 · cut L12 20%", recipeKey: "a", rule: "keep" }), warned);
    sight(swift({ recipe: "Native fp16 · cut L12 20%", recipeKey: "b", rule: "keep" }), warned); // same label, another recipe
    sight({ bird: birdFor(2, 0.98), model: "pigeon", modelName: "Pigeon", recipe: "INT8 per channel · all", href: "/pigeon/coop/?preset=int8-channel", rule: "pigeon", pointing: 0.97 }, warned);
    const list = readLifeList().sightings;
    expect(list.map((s) => [s.recipeKey, s.rule, s.pointing])).toEqual([["a", "keep", undefined], ["b", "keep", undefined], [undefined, "pigeon", 0.97]]);
    expect(list.every((s) => !("bird" in s))).toBe(true);
    sight(swift({ recipe: "Native fp16 · cut L12 20%", recipeKey: "a", rule: "keep" }), warned); // the same recipe again replaces its sighting
    expect(readLifeList().sightings.map((s) => s.recipeKey)).toEqual(["b", undefined, "a"]);
  });

  it("says so when storage is blocked, and announces it only the first time", () => {
    globalThis.localStorage = { getItem: () => { throw new Error("blocked"); } } as unknown as Storage;
    const warned = { blocked: false };
    expect(sight(swift(), warned)).toEqual({ said: STORAGE_BLOCKED, announce: true });
    expect(sight(swift(), warned)).toEqual({ said: STORAGE_BLOCKED, announce: false });
    expect(lifeListCount()).toBe("can't be kept in this browser");
  });
});

describe("the cage's bird", () => {
  beforeEach(() => void vi.useFakeTimers());
  afterEach(() => void vi.useRealTimers());
  const box = () => {
    const classes = new Set<string>();
    return { innerHTML: "", classList: { add: (c: string) => void classes.add(c), remove: (c: string) => void classes.delete(c), contains: (c: string) => classes.has(c) } as unknown as DOMTokenList };
  };

  it("draws the first bird and same-species updates at once", () => {
    const b = box(), draw = birdDrawer(b);
    draw("swift", "swift thin");
    expect(b.innerHTML).toBe("swift thin");
    draw("swift", "swift fat");
    expect(b.innerHTML).toBe("swift fat");
    expect(b.classList.contains("swap")).toBe(false);
  });

  it("fades out before a new species, then draws it", () => {
    const b = box(), draw = birdDrawer(b, 180);
    draw("pigeon", "pigeon");
    draw("swift", "swift");
    expect(b.innerHTML).toBe("pigeon");
    expect(b.classList.contains("swap")).toBe(true);
    vi.advanceTimersByTime(180);
    expect(b.innerHTML).toBe("swift");
    expect(b.classList.contains("swap")).toBe(false);
  });

  it("ends on the latest result when the full score lands during the fade", () => {
    const b = box(), draw = birdDrawer(b, 180);
    draw("pigeon", "pigeon");
    draw("swift", "swift, quick estimate"); // the quick result changes species and starts the fade
    draw("swift", "swift, all 500 photos"); // the cached full result follows at once
    vi.advanceTimersByTime(180);
    expect(b.innerHTML).toBe("swift, all 500 photos");
    vi.advanceTimersByTime(1000);
    expect(b.innerHTML).toBe("swift, all 500 photos");
  });

  it("ends on the latest result when the species changes back during the fade", () => {
    const b = box(), draw = birdDrawer(b, 180);
    draw("pigeon", "pigeon");
    draw("swift", "swift");
    draw("pigeon", "pigeon again"); // a failed run puts the last finished bird back
    vi.advanceTimersByTime(180);
    expect(b.innerHTML).toBe("pigeon again");
    draw("pigeon", "pigeon later");
    expect(b.innerHTML).toBe("pigeon later");
  });
});
