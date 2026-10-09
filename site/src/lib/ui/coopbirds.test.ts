import { beforeEach, describe, expect, it } from "vitest";
import { birdFor, readLifeList } from "./birds";
import { cageHref, lifeListCount, sight, STORAGE_BLOCKED, type CoopSighting } from "./coopbirds";

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
    bird: birdFor(4, 0.99), model: "yolov8n", modelName: "YOLOv8n", recipe: "INT4 per channel", href: cageHref("/pigeon/coop/yolo/", "int4-channel"), ...over,
  });

  it("links a preset back to the cage with ?preset=, and anything else to the cage alone", () => {
    expect(cageHref("/pigeon/coop/", "mxfp4")).toBe("/pigeon/coop/?preset=mxfp4");
    expect(cageHref("/pigeon/coop/", null)).toBe("/pigeon/coop/");
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

  it("says so when storage is blocked, and announces it only the first time", () => {
    globalThis.localStorage = { getItem: () => { throw new Error("blocked"); } } as unknown as Storage;
    const warned = { blocked: false };
    expect(sight(swift(), warned)).toEqual({ said: STORAGE_BLOCKED, announce: true });
    expect(sight(swift(), warned)).toEqual({ said: STORAGE_BLOCKED, announce: false });
    expect(lifeListCount()).toBe("can't be kept in this browser");
  });
});
