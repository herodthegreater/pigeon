/**
 * Pigeon in the browser: ONNX Runtime Web with every weight fed as an input.
 * The page owns the weights (Float32Arrays), edits them, and passes them in.
 */

import * as ort from "onnxruntime-web/wasm";

export type Kind = "conv" | "depthwise" | "pointwise" | "squeeze_excite" | "head";

export interface TensorMeta {
  name: string;
  shape: number[];
  offset: number;
  numel: number;
  kind: Kind;
  role: "weight" | "bias";
  layer: string;
}

export type Weights = Map<string, Float32Array>;

export interface PigeonModel {
  session: ort.InferenceSession;
  tensors: TensorMeta[];
  original: Weights;
}

export interface Output {
  logits: Float32Array; // (N,)
  cams: Float32Array; // (N, 7, 7)
}

export { INPUT, CAM } from "./constants";
import { INPUT } from "./constants";

const base = () => (import.meta.env.BASE_URL ?? "/").replace(/\/?$/, "/");

export async function loadModel(): Promise<PigeonModel> {
  // ORT finds its wasm runtime next to its own module; Vite emits it as an asset in the build.
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const [graph, tensors, bin] = await Promise.all([
    fetch(`${base()}model/pigeon.onnx`).then((r) => r.arrayBuffer()),
    fetch(`${base()}model/weights.json`).then((r) => r.json() as Promise<TensorMeta[]>),
    fetch(`${base()}model/weights.bin`).then((r) => r.arrayBuffer()),
  ]);
  const session = await ort.InferenceSession.create(graph, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  const original: Weights = new Map();
  for (const t of tensors) original.set(t.name, new Float32Array(bin, t.offset, t.numel));
  return { session, tensors, original };
}

/** Run a batch of images (N x 3 x 224 x 224, values 0..1) through Pigeon with the given weights. */
export async function run(model: PigeonModel, images: Float32Array, n: number, weights: Weights): Promise<Output> {
  const feeds: Record<string, ort.Tensor> = { image: new ort.Tensor("float32", images, [n, 3, INPUT, INPUT]) };
  for (const t of model.tensors) feeds[t.name] = new ort.Tensor("float32", weights.get(t.name)!, t.shape);
  const out = await model.session.run(feeds);
  return { logits: out.logit.data as Float32Array, cams: out.cam.data as Float32Array };
}
