import { describe, expect, it } from "vitest";
import { ALL_KINDS, type Recipe, type Step } from "../engine/pipeline";
import { describeRecipe, linkQuery, MAX_LINK_STEPS, PRESETS, presetOf, recipeFromLink } from "./recipe";

describe("a recipe's preset", () => {
  it("finds every preset by its steps, so a sighting links back with ?preset=", () => {
    for (const p of PRESETS) expect(presetOf(JSON.parse(JSON.stringify(p.recipe)))).toBe(p.id);
    expect(new Set(PRESETS.map((p) => describeRecipe(p.recipe))).size).toBe(PRESETS.length);
  });

  it("finds none once a setting is changed by hand", () => {
    const int8 = PRESETS.find((p) => p.id === "int8-channel")!.recipe;
    const edited = JSON.parse(JSON.stringify(int8));
    edited.steps[0].bits = 7;
    expect(presetOf(edited)).toBeNull();
    edited.steps[0].bits = 8;
    edited.steps[0].symmetric = false;
    expect(presetOf(edited)).toBeNull();
  });
});

describe("a recipe in a link", () => {
  const read = (q: string) => recipeFromLink(new URLSearchParams(q));
  const back = (r: Recipe) => {
    const got = read(linkQuery(r));
    return got && "recipe" in got ? got.recipe : got;
  };

  it("reopens every preset by its id, and full precision with no query", () => {
    for (const p of PRESETS) expect(linkQuery(p.recipe)).toBe(p.id === "fp32" ? "" : `preset=${p.id}`);
    expect(read("preset=mxfp4")).toEqual({ recipe: PRESETS.find((p) => p.id === "mxfp4")!.recipe, preset: "mxfp4" });
    expect(read("")).toBeNull();
  });

  it("carries every setting of a hand-built recipe, and brings back the same recipe", () => {
    const steps: Step[] = [
      { op: "int", parts: ["pointwise", "head"], bits: 6, granularity: { kind: "group", size: 64 }, symmetric: false },
      { op: "int", parts: [...ALL_KINDS], bits: 2, granularity: { kind: "tensor" }, symmetric: true, ternary: true },
      { op: "float", parts: ["depthwise"], format: "fp4_e2m1", scaling: { kind: "nv", block: 16 } },
      { op: "prune", parts: [...ALL_KINDS], method: "magnitude", amount: 0.3 },
      { op: "prune", parts: ["conv"], method: "channels", amount: 0.45 },
      { op: "prune", parts: ["squeeze_excite"], method: "nm", amount: 0.5 },
      { op: "cluster", parts: [], k: 32 },
    ];
    const r: Recipe = { steps };
    expect(linkQuery(r)).toBe("steps=int.6.g64.asym.pointwise-head,int.2.tensor.ternary.all,float.fp4_e2m1.nv.depthwise,prune.magnitude.30.all,prune.channels.45.conv,prune.nm.squeeze_excite,cluster.32.none");
    expect(back(r)).toEqual(r);
    expect(describeRecipe(back(r) as Recipe)).toBe(describeRecipe(r));
    expect(read(linkQuery(r))).toMatchObject({ preset: null });
  });

  it("finds the preset again when a link spells one out step by step", () => {
    expect(read("steps=int.8.channel.all")).toMatchObject({ preset: "int8-channel" });
  });

  it("clamps every number to what its control can set", () => {
    expect(back({ steps: [{ op: "prune", parts: ["head"], method: "channels", amount: 0.9 }] })).toEqual({ steps: [{ op: "prune", parts: ["head"], method: "channels", amount: 0.6 }] });
    expect(read("steps=int.12.channel.all,int.0.g20.all,prune.magnitude.33.all,cluster.1000.all")).toEqual({
      recipe: { steps: [
        { op: "int", parts: [...ALL_KINDS], bits: 8, granularity: { kind: "channel" }, symmetric: true },
        { op: "int", parts: [...ALL_KINDS], bits: 1, granularity: { kind: "group", size: 16 }, symmetric: true },
        { op: "prune", parts: [...ALL_KINDS], method: "magnitude", amount: 0.35 },
        { op: "cluster", parts: [...ALL_KINDS], k: 256 },
      ] },
      preset: null,
    });
    expect(read("steps=int.4.channel.ternary.all")).toMatchObject({ recipe: { steps: [{ bits: 4 }] } }); // no ternary box at 4 bits
    expect((read("steps=int.4.channel.ternary.all") as { recipe: Recipe }).recipe.steps[0]).not.toHaveProperty("ternary");
  });

  it("names what it couldn't read, and loads nothing", () => {
    expect(read("preset=int3")).toEqual({ error: 'there is no preset called "int3"' });
    expect(read("steps=int.8.channel.wings")).toEqual({ error: 'step 1, "int.8.channel.wings", isn\'t a step the Coop can build' });
    expect(read("steps=prune.magnitude.lots.all")).toMatchObject({ error: expect.stringContaining("step 1") });
    expect(read("steps=float.fp6.none.all")).toMatchObject({ error: expect.stringContaining("step 1") });
    expect(read(`steps=${Array(MAX_LINK_STEPS + 1).fill("cluster.16.all").join(",")}`)).toMatchObject({ error: expect.stringContaining(`at most ${MAX_LINK_STEPS}`) });
  });
});
