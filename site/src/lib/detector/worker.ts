/// <reference lib="webworker" />
/**
 * The detector worker: downloads YOLOv8n and the test photos, starts TF.js on
 * WebGL (through an OffscreenCanvas, so the page's main thread stays free),
 * scores the native model once as the reference, then scores each recipe the
 * page sends: a quick pass on the first 100 photos, then all 500. A newer
 * request stops the running one between batches.
 */

import { Detector, startWebGL, type Cost, type PruneStage } from "./engine";
import { ap50, type Det } from "./score";
import { keyOf, nativeRecipe, QUICK, type DetectorRecipe } from "./recipe";
import { cutPhotos, downloadAll, MODEL_DIR, parseManifest, parseTestSet, sheetUrl, TESTSET_URL, type Photos } from "./files";
import type { TestImage } from "../engine/testset";

const BATCH = 10;
const SHOW_CONF = 0.25; // boxes drawn on the preview photos, as Ultralytics' predict shows them
const CACHE = 12; // recipes whose detections are kept, besides the native one

export type Request = { type: "init" } | { type: "evaluate"; id: number; recipe: DetectorRecipe; quick: boolean };

export interface Result {
  id: number;
  /** false for the quick estimate on the first 100 photos */
  final: boolean;
  n: number;
  ap: number;
  /** The native model's AP on the same photos. */
  nativeAp: number;
  cost: Cost;
  ms: { quantize: number; infer: number };
  /** Photos this request ran through the model; 0 when every detection was kept from an earlier run. */
  ran: number;
  /** Detections above 25% confidence on each preview photo (final results only). */
  previews?: Det[][];
}

export interface Ready {
  stages: (PruneStage & { kind: string })[];
  native: { ap: number; apQuick: number; cost: Cost; previews: Det[][] };
  previews: (TestImage & { index: number })[];
  grid: number;
  n: number;
  float32: boolean;
  ms: number;
  recorded: { ap: Record<string, { all: number; quick: number }>; ultralytics: string; torch: string; date: string };
}

export type Boot =
  | { stage: "download"; got: number; total: number }
  | { stage: "photos"; done: number; total: number }
  | { stage: "gpu" }
  | { stage: "score"; done: number; total: number };

export type Message =
  | { type: "boot"; boot: Boot }
  | { type: "ready"; ready: Ready }
  | { type: "progress"; id: number; done: number; total: number; final: boolean }
  | { type: "result"; result: Result }
  | { type: "error"; id?: number; message: string; code?: "no-webgl" };

const post = (m: Message) => self.postMessage(m);
const errText = (err: unknown) => (err instanceof Error ? err.message : String(err));

let det: Detector;
let photos: Photos;
let previewIdx: number[] = [];
let native: { all: number; quick: number };
const cache = new Map<string, { preds: Map<number, Det[]>; cost?: Cost }>();
let latest = 0;
let pending: Extract<Request, { type: "evaluate" }> | null = null;
let running = false;

function entry(key: string) {
  let e = cache.get(key);
  if (e) { cache.delete(key); cache.set(key, e); return e; } // most recent last
  e = { preds: new Map() };
  cache.set(key, e);
  const nativeKey = keyOf(nativeRecipe());
  for (const k of cache.keys()) { if (cache.size <= CACHE + 1) break; if (k !== nativeKey && k !== key) cache.delete(k); }
  return e;
}

const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const apOn = (n: number, preds: Map<number, Det[]>) => ap50(photos.gt.slice(0, n), range(n).map((i) => preds.get(i)!));

/**
 * Detections for photos [0, n) under the recipe now applied. Returns how many photos
 * it had to run (the rest were kept from before), or null if a newer request took over.
 */
async function predict(e: { preds: Map<number, Det[]> }, n: number, onDone: (done: number) => void, stale?: () => boolean) {
  const todo = range(n).filter((i) => !e.preds.has(i));
  const already = n - todo.length;
  if (todo.length) {
    const got = await det.detect(todo, BATCH, (d) => onDone(already + d), stale);
    if (!got) return null;
    todo.forEach((i, j) => e.preds.set(i, got[j]));
  }
  onDone(n);
  return todo.length;
}

const shown = (preds: Map<number, Det[]>) => previewIdx.map((i) => preds.get(i)!.filter((b) => b[4] > SHOW_CONF));

async function init() {
  const dir = MODEL_DIR();
  const [manBuf, setBuf] = await downloadAll([`${dir}model.json`, TESTSET_URL()], () => {});
  const m = parseManifest(manBuf), meta = parseTestSet(setBuf);
  if (meta.size !== m.imgsz) throw new Error(`the test photos are ${meta.size} px but the model expects ${m.imgsz} px`);
  const sheetIds = [...new Set(meta.images.map((i) => i.sheet))].sort((a, b) => a - b);
  const bufs = await downloadAll([`${dir}${m.weights_file}`, ...sheetIds.map(sheetUrl)], (got, total) => post({ type: "boot", boot: { stage: "download", got, total } }));
  if (bufs[0].byteLength !== m.weights_bytes) throw new Error(`weights.bin is ${bufs[0].byteLength} bytes, not the ${m.weights_bytes} model.json expects`);
  photos = await cutPhotos(meta, new Map(sheetIds.map((s, i) => [s, bufs[i + 1]])), (done, total) => post({ type: "boot", boot: { stage: "photos", done, total } }));
  const ids = new Map(meta.images.map((img, i) => [img.id, i]));
  previewIdx = m.preview.map((id) => ids.get(id)).filter((i): i is number => i !== undefined);

  post({ type: "boot", boot: { stage: "gpu" } });
  const gl = await startWebGL();
  if (!gl.ok) {
    post({ type: "error", code: "no-webgl", message: "this browser didn't give the page's worker WebGL" });
    return;
  }
  det = new Detector(m, bufs[0], photos.px);

  const n = photos.images.length, t0 = performance.now();
  const e = entry(keyOf(nativeRecipe()));
  await predict(e, n, (done) => post({ type: "boot", boot: { stage: "score", done, total: n } }));
  const ms = performance.now() - t0;
  e.cost = det.cost(nativeRecipe(), null);
  native = { all: apOn(n, e.preds), quick: apOn(Math.min(QUICK, n), e.preds) };
  const kinds = new Map(m.graph.map((g) => [`model.${g.i}`, g.type]));
  post({
    type: "ready",
    ready: {
      stages: m.prune_stages.map((s) => ({ ...s, kind: kinds.get(s.stage) ?? "" })),
      native: { ap: native.all, apQuick: native.quick, cost: e.cost, previews: shown(e.preds) },
      previews: previewIdx.map((i) => ({ ...photos.images[i], index: i })),
      grid: photos.grid,
      n,
      float32: gl.float32,
      ms,
      recorded: m.recorded,
    },
  });
}

async function evaluate(req: Extract<Request, { type: "evaluate" }>): Promise<void> {
  const { id, recipe } = req, n = photos.images.length;
  const stale = () => id !== latest;
  const e = entry(keyOf(recipe));
  const t0 = performance.now();
  det.apply(recipe);
  const t1 = performance.now();
  // Masking changes which channels are live; precision alone stores every channel.
  if (!e.cost) e.cost = det.cost(recipe, Object.values(recipe.masks).some((v) => v > 0) ? det.probe() : null);
  const counts = req.quick ? [Math.min(QUICK, n), n] : [n];
  let ran = 0;
  for (const count of counts) {
    const got = await predict(e, count, (done) => post({ type: "progress", id, done, total: count, final: count === n }), stale);
    if (got === null) return;
    ran += got;
    const final = count === n;
    post({
      type: "result",
      result: {
        id, final, n: count, ap: apOn(count, e.preds), nativeAp: final ? native.all : native.quick, cost: e.cost,
        ms: { quantize: t1 - t0, infer: performance.now() - t1 }, ran,
        previews: final ? shown(e.preds) : undefined,
      },
    });
  }
}

/** One evaluation at a time; a newer request replaces a queued one and stops the running one. */
async function pump() {
  if (running) return;
  running = true;
  while (pending) {
    const req = pending;
    pending = null;
    try {
      await evaluate(req);
    } catch (err) {
      post({ type: "error", id: req.id, message: errText(err) });
    }
  }
  running = false;
}

self.onmessage = (ev: MessageEvent<Request>) => {
  const req = ev.data;
  if (req.type === "init") {
    init().catch((err) => post({ type: "error", message: errText(err) }));
  } else {
    latest = req.id;
    pending = req;
    void pump();
  }
};
