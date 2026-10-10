import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { idleLine, runActivity } from "./results";

const LAB = new URL("../../data/lab/", import.meta.url);
const methods: { id: string; status: string }[] = JSON.parse(readFileSync(new URL("methods.json", LAB), "utf8"));
const params = (id: string): { name: string; kind: string; default: unknown; choices: string[] | null }[] =>
  JSON.parse(readFileSync(new URL(`methods/${id}.json`, LAB), "utf8")).params;
const runnable = new Set(methods.filter((m) => m.status === "runnable").map((m) => m.id));
const defaults = (id: string) => Object.fromEntries(params(id).map((p) => [p.name, p.default]));

describe("what a run says it is doing", () => {
  it("says Training only for runs that train", () => {
    for (const id of ["ptq", "rtn", "gptq", "awq"]) expect(runActivity(id, defaults(id))).toBe("Rounding the weights");
    for (const id of ["smoothquant", "wanda", "sparsegpt", "block_influence"]) expect(runActivity(id, defaults(id))).toBe("Running");
    for (const id of ["kd", "qat", "qlora", "lottery_ticket", "structured_pruning"]) expect(runActivity(id, defaults(id))).toBe("Training");
  });

  it("drops Training when the only training is a recovery step that is turned off", () => {
    expect(runActivity("magnitude_pruning", { ...defaults("magnitude_pruning"), recovery: "none" })).toBe("Running");
    expect(runActivity("magnitude_pruning", { ...defaults("magnitude_pruning"), recovery: "kd" })).toBe("Training");
    expect(runActivity("low_rank", { rank: 128, recovery: "none" })).toBe("Running");
    expect(runActivity("layer_dropping", { recovery: "none" })).toBe("Running");
    expect(runActivity("weight_sharing", { k: 4, finetune: false })).toBe("Running");
    expect(runActivity("weight_sharing", { k: 4, finetune: true })).toBe("Training");
  });

  it("names only methods that exist, run in the browser, and have the setting it reads", () => {
    for (const id of ["ptq", "rtn", "gptq", "awq", "smoothquant", "wanda", "sparsegpt", "block_influence"]) expect(runnable.has(id), id).toBe(true);
    for (const id of ["magnitude_pruning", "layer_dropping", "low_rank"]) {
      expect(params(id).find((p) => p.name === "recovery")?.choices, id).toContain("none");
    }
    expect(params("weight_sharing").find((p) => p.name === "finetune")?.kind).toBe("bool");
  });

  it("says before a first run what the run does", () => {
    expect(idleLine("ptq")).toMatch(/^A run loads the lab's trained model, rounds its weights .* Nothing is trained\./);
    expect(idleLine("wanda")).toContain("Nothing is trained.");
    expect(idleLine("kd")).not.toContain("Nothing is trained");
    for (const id of runnable) expect(idleLine(id)).toContain("Every model a run makes smaller ends in a bird.");
  });
});
