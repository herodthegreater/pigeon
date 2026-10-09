/**
 * Main-thread handle on the engine worker. `evaluate` is "live": each call
 * supersedes the previous one; you get a quick estimate on the first 100 images,
 * then the full 500-image score.
 */

import type { Recipe } from "./pipeline";
import type { Message, Result } from "./worker";
import type { TensorMeta } from "./model";
import type { TestImage } from "./testset";

export interface Ready {
  images: TestImage[];
  baseline: Result;
  tensors: TensorMeta[];
}

export interface LiveHandlers {
  onQuick?: (r: Result) => void;
  onFull: (r: Result) => void;
  onProgress?: (done: number, total: number, stage: "quick" | "full") => void;
  onError?: (message: string) => void;
}

/** Start-up progress: downloading the files, then scoring the original Pigeon. */
export type Boot = { stage: "download"; bytes: number } | { stage: "score"; done: number; total: number };

export const QUICK = 100;

/** The browser lacks what the engine needs (Web Workers, WebAssembly): a reload or a better connection can't fix it. */
export class Unsupported extends Error {}

export class Engine {
  private worker: Worker | null;
  private nextId = 1;
  private current: { quick: number; full: number; recipe: Recipe; h: LiveHandlers } | null = null;
  readonly ready: Promise<Ready>;

  constructor(private onBoot?: (b: Boot) => void) {
    if (typeof Worker === "undefined" || typeof WebAssembly === "undefined") {
      this.worker = null;
      this.ready = Promise.reject(new Unsupported("this browser can't run Web Workers with WebAssembly"));
      return;
    }
    let worker: Worker;
    try {
      worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    } catch {
      // Browsers without module workers throw here rather than failing later.
      this.worker = null;
      this.ready = Promise.reject(new Unsupported("this browser can't start the engine's background worker"));
      return;
    }
    this.worker = worker;
    let booted = false;
    this.ready = new Promise((resolve, rejectReady) => {
      // A start-up failure is final: stop the worker so straggling downloads can't report progress.
      const reject = (err: Error) => {
        worker.terminate();
        this.onBoot = undefined;
        rejectReady(err);
      };
      worker.onmessage = (e: MessageEvent<Message>) => {
        const m = e.data;
        if (m.type === "ready") {
          booted = true;
          resolve({ images: m.images, baseline: m.baseline, tensors: m.tensors });
        } else if (m.type === "error") {
          if (!booted) reject(new Error(m.message));
          else this.fail(m.message, m.id);
        } else this.route(m);
      };
      // The worker script itself failed to load or crashed outside a request.
      worker.onerror = (e) => {
        e.preventDefault();
        const message = e.message || "the engine worker crashed";
        if (!booted) reject(new Error(message));
        else this.fail(message);
      };
    });
    worker.postMessage({ type: "init" });
  }

  private fail(message: string, id?: number) {
    const c = this.current;
    if (!c || (id !== undefined && id !== c.quick && id !== c.full)) return; // stale run
    this.current = null;
    c.h.onError?.(message);
  }

  evaluate(recipe: Recipe, h: LiveHandlers) {
    const quick = this.nextId++, full = this.nextId++;
    this.current = { quick, full, recipe, h };
    this.worker?.postMessage({ type: "evaluate", id: quick, recipe, count: QUICK });
  }

  private route(m: Message) {
    if (m.type === "boot") return this.onBoot?.({ stage: "download", bytes: m.bytes });
    if (m.type === "progress" && m.id === 0) return this.onBoot?.({ stage: "score", done: m.done, total: m.total });
    const c = this.current;
    if (!c) return;
    if (m.type === "progress") {
      if (m.id === c.quick) c.h.onProgress?.(m.done, m.total, "quick");
      if (m.id === c.full) c.h.onProgress?.(m.done, m.total, "full");
    } else if (m.type === "result") {
      if (m.result.id === c.quick) {
        c.h.onQuick?.(m.result);
        this.worker?.postMessage({ type: "evaluate", id: c.full, recipe: c.recipe });
      } else if (m.result.id === c.full) {
        c.h.onFull(m.result);
      }
    }
  }
}
