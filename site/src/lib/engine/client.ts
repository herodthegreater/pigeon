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
}

export const QUICK = 100;

export class Engine {
  private worker: Worker;
  private nextId = 1;
  private current: { quick: number; full: number; recipe: Recipe; h: LiveHandlers } | null = null;
  readonly ready: Promise<Ready>;

  constructor() {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.ready = new Promise((resolve, reject) => {
      this.worker.onmessage = (e: MessageEvent<Message>) => {
        const m = e.data;
        if (m.type === "ready") resolve({ images: m.images, baseline: m.baseline, tensors: m.tensors });
        else if (m.type === "error") (this.current ? console.error(m.message) : reject(new Error(m.message)));
        else this.route(m);
      };
    });
    this.worker.postMessage({ type: "init" });
  }

  evaluate(recipe: Recipe, h: LiveHandlers) {
    const quick = this.nextId++, full = this.nextId++;
    this.current = { quick, full, recipe, h };
    this.worker.postMessage({ type: "evaluate", id: quick, recipe, count: QUICK });
  }

  private route(m: Message) {
    const c = this.current;
    if (!c) return;
    if (m.type === "progress") {
      if (m.id === c.quick) c.h.onProgress?.(m.done, m.total, "quick");
      if (m.id === c.full) c.h.onProgress?.(m.done, m.total, "full");
    } else if (m.type === "result") {
      if (m.result.id === c.quick) {
        c.h.onQuick?.(m.result);
        this.worker.postMessage({ type: "evaluate", id: c.full, recipe: c.recipe });
      } else if (m.result.id === c.full) {
        c.h.onFull(m.result);
      }
    }
  }
}
