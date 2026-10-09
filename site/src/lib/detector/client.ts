/**
 * The page's handle on the detector worker. `evaluate` is live: each call
 * supersedes the one before; handlers hear the quick estimate on the first 100
 * photos, then the score on all 500.
 */

import type { DetectorRecipe } from "./recipe";
import type { Boot, Message, Ready, Result } from "./worker";

export type { Boot, Ready, Result };

export interface LiveHandlers {
  onProgress?: (done: number, total: number, final: boolean) => void;
  onQuick?: (r: Result) => void;
  onFull: (r: Result) => void;
  onError?: (message: string) => void;
}

/** Why the detector can't start in this browser, said plainly; null when it can. */
export function missingSupport(): string | null {
  if (typeof Worker === "undefined") return "this browser can't run background workers (Web Workers)";
  if (typeof OffscreenCanvas === "undefined") return "this browser can't draw off screen (OffscreenCanvas), which the detector needs to reach WebGL from a worker";
  const canvas = document.createElement("canvas");
  if (!canvas.getContext("webgl2")) return "this browser has WebGL 2 turned off or doesn't support it";
  return null;
}

export class NoWebGL extends Error {}

export class DetectorEngine {
  private worker: Worker;
  private nextId = 1;
  private current: { id: number; h: LiveHandlers } | null = null;
  readonly ready: Promise<Ready>;

  constructor(private onBoot: (b: Boot) => void) {
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker = worker;
    let booted = false;
    this.ready = new Promise((resolve, rejectReady) => {
      // A start-up failure is final: stop the worker so straggling downloads can't report progress.
      const reject = (err: Error) => {
        worker.terminate();
        this.onBoot = () => {};
        rejectReady(err);
      };
      worker.onmessage = (e: MessageEvent<Message>) => {
        const m = e.data;
        if (m.type === "boot") this.onBoot(m.boot);
        else if (m.type === "ready") { booted = true; resolve(m.ready); }
        else if (m.type === "error") {
          if (!booted) reject(m.code === "no-webgl" ? new NoWebGL(m.message) : new Error(m.message));
          else this.fail(m.message, m.id);
        } else this.route(m);
      };
      // The worker script failed to load, or crashed outside a request.
      worker.onerror = (e) => {
        e.preventDefault();
        const message = e.message || "the detector's worker crashed";
        if (!booted) reject(new Error(message));
        else this.fail(message);
      };
    });
    worker.postMessage({ type: "init" });
  }

  evaluate(recipe: DetectorRecipe, h: LiveHandlers) {
    const id = this.nextId++;
    this.current = { id, h };
    this.worker.postMessage({ type: "evaluate", id, recipe, quick: true });
  }

  private fail(message: string, id?: number) {
    const c = this.current;
    if (!c || (id !== undefined && id !== c.id)) return; // a superseded run
    this.current = null;
    c.h.onError?.(message);
  }

  private route(m: Message) {
    const c = this.current;
    if (!c) return;
    if (m.type === "progress" && m.id === c.id) c.h.onProgress?.(m.done, m.total, m.final);
    else if (m.type === "result" && m.result.id === c.id) {
      if (m.result.final) { this.current = null; c.h.onFull(m.result); }
      else c.h.onQuick?.(m.result);
    }
  }
}
