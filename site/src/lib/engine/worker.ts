/// <reference lib="webworker" />
/**
 * The engine worker: owns the model and test set, applies recipes, runs the
 * eval set, and reports the scoreboard — all off the main thread.
 */

import { loadModel, run, type PigeonModel } from "./model";
import { predictions, score, type Score } from "./metrics";
import { applyRecipe, type Recipe } from "./pipeline";
import { batch, loadTestSet, type TestSet } from "./testset";

export type Request = { type: "init" } | { type: "evaluate"; id: number; recipe: Recipe; count?: number };

export interface Result {
  id: number;
  score: Score;
  totalBytes: number;
  originalBytes: number;
  logits: Float32Array;
  cams: Float32Array;
  ms: { compress: number; infer: number };
  tensors: ReturnType<typeof applyRecipe>["tensors"];
}

export type Message =
  | { type: "ready"; images: TestSet["images"]; baseline: Result; tensors: PigeonModel["tensors"] }
  | { type: "progress"; id: number; done: number; total: number }
  | { type: "result"; result: Result }
  | { type: "error"; id?: number; message: string };

const BATCH = 25;
let model: PigeonModel;
let testset: TestSet;
let refPred: Uint8Array;
let latest = 0; // newest request id: stale evaluations stop early
let pending: Extract<Request, { type: "evaluate" }> | null = null;
let running = false;

/** One evaluation at a time; a newer request replaces any queued one and aborts the running one. */
async function pump() {
  if (running) return;
  running = true;
  while (pending) {
    const req = pending;
    pending = null;
    const result = await evaluate(req.id, req.recipe, req.count);
    if (result) self.postMessage({ type: "result", result } satisfies Message);
  }
  running = false;
}

async function evaluate(id: number, recipe: Recipe, count = testset.images.length): Promise<Result | null> {
  const t0 = performance.now();
  const applied = applyRecipe(model.tensors, model.original, recipe);
  const t1 = performance.now();
  const logits = new Float32Array(count);
  const cams = new Float32Array(count * 49);
  for (let i = 0; i < count; i += BATCH) {
    if (id !== 0 && id !== latest) return null;
    const n = Math.min(BATCH, count - i);
    const out = await run(model, batch(testset, i, i + n), n, applied.weights);
    logits.set(out.logits, i);
    cams.set(out.cams, i * 49);
    self.postMessage({ type: "progress", id, done: i + n, total: count } satisfies Message);
  }
  const images = testset.images.slice(0, count);
  return {
    id,
    score: score(images, logits, cams, refPred?.subarray(0, count)),
    totalBytes: applied.totalBytes,
    originalBytes: applied.originalBytes,
    logits,
    cams,
    ms: { compress: t1 - t0, infer: performance.now() - t1 },
    tensors: applied.tensors,
  };
}

self.onmessage = async (e: MessageEvent<Request>) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      [model, testset] = await Promise.all([loadModel(), loadTestSet()]);
      const baseline = (await evaluate(0, { steps: [] }))!;
      refPred = predictions(baseline.logits);
      baseline.score.agreement = 1;
      self.postMessage({ type: "ready", images: testset.images, baseline, tensors: model.tensors } satisfies Message);
    } else if (msg.type === "evaluate") {
      latest = msg.id;
      pending = msg;
      await pump();
    }
  } catch (err) {
    self.postMessage({ type: "error", id: "id" in msg ? msg.id : undefined, message: String(err) } satisfies Message);
  }
};
