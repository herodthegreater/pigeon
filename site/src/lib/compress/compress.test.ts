import { describe, expect, it } from "vitest";
import fixture from "./__fixtures__/python_quant.json";
import { distinctValues, mse, quantizeFloat, quantizeInt, roundMiniFloat, scaleGroups, FORMATS } from "./quant";
import { kmeansCluster, pruneChannels, pruneMagnitude, pruneNM, sparsity } from "./prune";
import { scaleCount, tensorBits } from "./size";

const W = Float32Array.from(fixture.w);
const SHAPE = fixture.shape;

describe("int quantization matches the v0 Python harness", () => {
  for (const c of fixture.cases) {
    it(`${c.bits}-bit, ${c.group ? `group ${c.group}` : "per-tensor"}`, () => {
      const q = quantizeInt(W, SHAPE, {
        bits: c.bits,
        granularity: c.group ? { kind: "group", size: c.group } : { kind: "tensor" },
        symmetric: true,
      });
      const ref = Float32Array.from(c.out);
      for (let i = 0; i < q.length; i++) expect(q[i]).toBeCloseTo(ref[i], 6);
    });
  }
});

describe("int quantization behaves", () => {
  it("uses at most 2^bits levels per scale group", () => {
    for (const bits of [2, 3, 4]) {
      const q = quantizeInt(W, SHAPE, { bits, granularity: { kind: "channel" }, symmetric: false });
      for (let c = 0; c < 4; c++) expect(distinctValues(q, [c * 64, (c + 1) * 64])).toBeLessThanOrEqual(2 ** bits);
    }
  });
  it("finer granularity never hurts (the outlier only poisons its own group)", () => {
    const e = (g: any) => mse(W, quantizeInt(W, SHAPE, { bits: 4, granularity: g, symmetric: true }));
    expect(e({ kind: "channel" })).toBeLessThan(e({ kind: "tensor" }));
    expect(e({ kind: "group", size: 16 })).toBeLessThanOrEqual(e({ kind: "channel" }));
  });
  it("is idempotent", () => {
    const o = { bits: 4, granularity: { kind: "channel" } as const, symmetric: true };
    const q = quantizeInt(W, SHAPE, o);
    expect(Array.from(quantizeInt(q, SHAPE, o))).toEqual(Array.from(q));
  });
  it("ternary yields {-s, 0, +s} per group", () => {
    const q = quantizeInt(W, SHAPE, { bits: 2, ternary: true, granularity: { kind: "channel" }, symmetric: true });
    expect(distinctValues(q, [0, 64])).toBeLessThanOrEqual(3);
  });
});

describe("minifloats", () => {
  it("represents the FP4 E2M1 grid exactly", () => {
    const grid = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
    for (const v of grid) expect(roundMiniFloat(v, FORMATS.fp4_e2m1)).toBe(v);
    expect(roundMiniFloat(5, FORMATS.fp4_e2m1)).toBe(4); // tie -> even mantissa
    expect(roundMiniFloat(100, FORMATS.fp4_e2m1)).toBe(6); // saturates
  });
  it("matches known FP8 E4M3 values", () => {
    expect(roundMiniFloat(448, FORMATS.fp8_e4m3)).toBe(448);
    expect(roundMiniFloat(1000, FORMATS.fp8_e4m3)).toBe(448);
    expect(roundMiniFloat(2 ** -9, FORMATS.fp8_e4m3)).toBe(2 ** -9); // smallest subnormal
    expect(roundMiniFloat(0.3, FORMATS.fp8_e4m3)).toBeCloseTo(0.3125, 6);
  });
  it("FP16 cast is near-lossless on small weights; MXFP4 beats per-tensor FP4", () => {
    expect(mse(W, quantizeFloat(W, SHAPE, { format: "fp16", scaling: { kind: "none" } }))).toBeLessThan(1e-8);
    const t = mse(W, quantizeFloat(W, SHAPE, { format: "fp4_e2m1", scaling: { kind: "tensor" } }));
    const mx = mse(W, quantizeFloat(W, SHAPE, { format: "fp4_e2m1", scaling: { kind: "mx", block: 32 } }));
    const nv = mse(W, quantizeFloat(W, SHAPE, { format: "fp4_e2m1", scaling: { kind: "nv", block: 16 } }));
    expect(mx).toBeLessThan(t);
    expect(nv).toBeLessThan(t);
  });
});

describe("pruning and clustering", () => {
  it("magnitude pruning hits the requested sparsity and keeps the big weights", () => {
    const p = pruneMagnitude(W, 0.5);
    expect(sparsity(p)).toBeCloseTo(0.5, 2);
    expect(p[1 * 64 + 5]).toBeCloseTo(0.9, 6); // the outlier survives
  });
  it("2:4 keeps exactly two of every four", () => {
    const p = pruneNM(W, SHAPE, 2, 4);
    for (let i = 0; i < p.length; i += 4) expect([p[i], p[i + 1], p[i + 2], p[i + 3]].filter((v) => v !== 0).length).toBe(2);
  });
  it("channel pruning zeros whole rows and never removes them all", () => {
    const { w, keep } = pruneChannels(W, SHAPE, 1);
    expect(keep.reduce((a, b) => a + b, 0)).toBe(1);
    expect(sparsity(w)).toBeCloseTo(0.75, 6);
  });
  it("k-means uses at most k distinct values", () => {
    expect(distinctValues(kmeansCluster(W, 16).w)).toBeLessThanOrEqual(16);
  });
});

describe("size accounting", () => {
  const t = { numel: 256, outChannels: 4 };
  it("counts scale overhead", () => {
    expect(tensorBits(t, { kind: "int", bits: 4, granularity: { kind: "tensor" }, symmetric: true })).toBe(256 * 4 + 16);
    expect(tensorBits(t, { kind: "int", bits: 4, granularity: { kind: "group", size: 32 }, symmetric: true })).toBe(256 * 4 + 8 * 16);
  });
  it("2:4 sparsity stores half the weights plus 2-bit indices", () => {
    expect(tensorBits(t, { kind: "fp32" }, { kind: "nm", n: 2, m: 4 })).toBe(128 * (32 + 2));
  });
  it("counts MX and NV block scales per output-channel row, the way the quantizer cuts the blocks", () => {
    const mx = { kind: "float", bits: 4, block: 32, scaleBitsPerBlock: 8 } as const;
    const nv = { kind: "float", bits: 4, block: 16, scaleBitsPerBlock: 8, tensorScaleBits: 32 } as const;
    // A depthwise 3x3 with 4 channels: rows of 9 weights, so each row is one short block.
    expect(tensorBits({ numel: 36, outChannels: 4 }, mx)).toBe(36 * 4 + 4 * 8);
    // Rows of 40: a block of 32 and a block of 8 in every row, not ceil(160 / 32) = 5 blocks.
    expect(tensorBits({ numel: 160, outChannels: 4 }, mx)).toBe(160 * 4 + 8 * 8);
    expect(tensorBits({ numel: 160, outChannels: 4 }, nv)).toBe(160 * 4 + 12 * 8 + 32);
  });
  it("stores one scale for every range the quantizer scales on its own", () => {
    for (const [out, per] of [[4, 64], [16, 9], [8, 25], [24, 8], [3, 40]]) {
      const shape = [out, per], n = out * per;
      for (const g of [{ kind: "tensor" }, { kind: "channel" }, { kind: "group", size: 16 }, { kind: "group", size: 32 }] as const) {
        expect(scaleCount({ numel: n, outChannels: out }, g)).toBe(scaleGroups(n, shape, g).length);
      }
    }
  });
});
