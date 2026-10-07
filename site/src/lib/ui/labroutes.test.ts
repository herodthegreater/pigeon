import { describe, expect, it } from "vitest";
import { labRedirectScript, type LabIds, type RedirectOptions } from "./labroutes";

const ids: LabIds = {
  methods: ["kd", "gptq"],
  families: ["small", "llm", "vision"],
  lessons: ["softmax_ce", "gptq_loop"],
  experiments: ["label_budget"],
};

/** A stand-in `window`: a location to read and replace, and a hashchange to fire. */
function fakeWindow(hash: string) {
  const w = {
    to: null as string | null,
    onHash: null as (() => void) | null,
    location: { hash, replace: (u: string) => { w.to = u; } },
    addEventListener: (type: string, fn: () => void) => { if (type === "hashchange") w.onHash = fn; },
  };
  return w;
}

function load(hash: string, o: Partial<RedirectOptions> = {}) {
  const w = fakeWindow(hash);
  new Function("window", labRedirectScript(ids, { base: "/pigeon/", fallback: null, strict: true, ...o }))(w);
  return w;
}

/** Load a page at `hash` and return where the script sent us, or null. */
const go = (hash: string, o: Partial<RedirectOptions> = {}) => load(hash, o).to;

describe("old lab URLs on the home page (strict, no fallback)", () => {
  it("maps every lab route", () => {
    expect(go("#/method/kd")).toBe("/pigeon/guide/kd/");
    expect(go("#/family/llm")).toBe("/pigeon/guide/#llm");
    expect(go("#/learn")).toBe("/pigeon/school/#build");
    expect(go("#/learn/gptq_loop")).toBe("/pigeon/school/gptq_loop/");
    expect(go("#/results")).toBe("/pigeon/results/");
    expect(go("#/results/label_budget")).toBe("/pigeon/results/label_budget/");
  });

  it("sends unknown ids to the section index, not a missing page", () => {
    expect(go("#/method/nope")).toBe("/pigeon/guide/");
    expect(go("#/family/nope")).toBe("/pigeon/guide/");
    expect(go("#/learn/nope")).toBe("/pigeon/school/#build");
    expect(go("#/results/nope")).toBe("/pigeon/results/");
  });

  it("leaves in-page anchors and the bare lab home alone", () => {
    expect(go("")).toBeNull();
    expect(go("#main")).toBeNull();
    expect(go("#/")).toBeNull();
    expect(go("#/somewhere")).toBeNull();
  });

  it("survives a malformed escape", () => {
    expect(go("#/method/%E0%A4%A")).toBe("/pigeon/guide/");
  });

  it("follows an old link pasted while already on the page", () => {
    const w = load("");
    expect(w.to).toBeNull();
    w.location.hash = "#/learn/softmax_ce";
    w.onHash!();
    expect(w.to).toBe("/pigeon/school/softmax_ce/");
  });
});

describe("old lab URLs on /lab/ (any hash, falls back to the Field Guide)", () => {
  const lab = { strict: false, fallback: "guide/" };
  it("sends a bare /lab/ to the Field Guide", () => {
    expect(go("", lab)).toBe("/pigeon/guide/");
    expect(go("#/", lab)).toBe("/pigeon/guide/");
  });
  it("accepts the lab's slashless form too", () => {
    expect(go("#method/gptq", lab)).toBe("/pigeon/guide/gptq/");
    expect(go("#/learn/softmax_ce", lab)).toBe("/pigeon/school/softmax_ce/");
  });
});
