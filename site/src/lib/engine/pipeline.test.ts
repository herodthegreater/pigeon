import { describe, expect, it } from "vitest";
import type { TensorMeta, Weights } from "./model";
import { applyRecipe, type Recipe } from "./pipeline";

// One 16 x 8 pointwise layer with distinct weights, and its bias.
const tensors: TensorMeta[] = [
  { name: "pw.weight", shape: [16, 8, 1, 1], offset: 0, numel: 128, kind: "pointwise", role: "weight", layer: "pw" },
  { name: "pw.bias", shape: [16], offset: 128, numel: 16, kind: "pointwise", role: "bias", layer: "pw" },
];
const original: Weights = new Map([
  ["pw.weight", Float32Array.from({ length: 128 }, (_, i) => (i % 2 ? 1 : -1) * (1 + ((i * 37) % 128)) / 128)],
  ["pw.bias", new Float32Array(16).fill(0.1)],
]);
const zeros = (w: Float32Array) => w.filter((v) => v === 0).length;

describe("a recipe that changes nothing", () => {
  it("leaves the model as big as it was: a 0% prune is the reference", () => {
    const r = applyRecipe(tensors, original, { steps: [{ op: "prune", parts: ["pointwise"], method: "magnitude", amount: 0 }] });
    expect(r.totalBytes).toBe(r.originalBytes);
    expect(zeros(r.weights.get("pw.weight")!)).toBe(0);
  });

  it("but a 2:4 step always removes half, whatever amount it last held", () => {
    // The editor keeps a step's amount when its method switches to 2:4, which has no amount of its own.
    const recipe: Recipe = { steps: [{ op: "prune", parts: ["pointwise"], method: "nm", amount: 0 }] };
    const r = applyRecipe(tensors, original, recipe);
    expect(zeros(r.weights.get("pw.weight")!)).toBe(64);
    expect(r.totalBytes).toBeLessThan(r.originalBytes);
    const half = applyRecipe(tensors, original, { steps: [{ op: "prune", parts: ["pointwise"], method: "nm", amount: 0.5 }] });
    expect(r.totalBytes).toBe(half.totalBytes);
  });
});
