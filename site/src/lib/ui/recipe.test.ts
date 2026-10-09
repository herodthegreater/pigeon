import { describe, expect, it } from "vitest";
import { describeRecipe, PRESETS, presetOf } from "./recipe";

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
