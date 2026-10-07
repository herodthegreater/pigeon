/**
 * Runs the lab's Python in the visitor's browser: a Web Worker (the lab's own
 * worker.js, served from <base>/lab/) loads Pyodide, numpy and the lab bundle,
 * then executes method runs and lesson checks. Nothing goes to a server.
 */

export type PyStatus = (text: string) => void;

let worker: Worker | null = null;
let seq = 0;
let ready = false; // Python finished loading at least once in this worker
const pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> | null }>();
const listeners = new Set<PyStatus>();

const base = () => import.meta.env.BASE_URL.replace(/\/?$/, "/");

function spawn(): Worker {
  if (typeof Worker === "undefined" || typeof WebAssembly === "undefined") {
    throw new Error("this browser can't run Python here: it needs Web Workers and WebAssembly");
  }
  const w = new Worker(`${base()}lab/worker.js`);
  w.onmessage = (e: MessageEvent) => {
    const m = e.data;
    if (m.type === "status") {
      if (m.text === "Python is ready.") ready = true;
      listeners.forEach((fn) => fn(m.text));
      return;
    }
    const p = pending.get(m.id);
    if (!p) return;
    pending.delete(m.id);
    if (p.timer) clearTimeout(p.timer);
    m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
  };
  w.onerror = (e) => {
    e.preventDefault();
    restart(new Error(e.message || "the Python worker crashed"));
  };
  return w;
}

/** Kill the worker and fail everything in flight (Python can't be interrupted otherwise). */
function restart(err: Error) {
  worker?.terminate();
  worker = null;
  ready = false;
  for (const [, p] of pending) {
    if (p.timer) clearTimeout(p.timer);
    p.reject(err);
  }
  pending.clear();
}

function call<T>(kind: string, args: unknown, timeoutMs = 0): Promise<T> {
  return new Promise((resolve, reject) => {
    try {
      worker ??= spawn();
    } catch (err) {
      return reject(err);
    }
    const id = ++seq;
    const timer = timeoutMs
      ? setTimeout(() => restart(new Error(`timed out after ${timeoutMs / 1000} s, so Python was restarted`)), timeoutMs)
      : null;
    pending.set(id, { resolve, reject, timer });
    worker.postMessage({ id, kind, args });
  });
}

/** Stop whatever is running: Python can't be interrupted, so the worker restarts (the next call reloads it). */
export function stop() {
  restart(new Error("stopped"));
}

/** Status lines while Python loads ("Downloading Python (about 20 MB, only the first time)…"). */
export function onStatus(fn: PyStatus): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** True once Python has loaded in this page; the first call also downloads it. */
export const isReady = () => ready;

export interface RunResult {
  rows: Record<string, number | string>[];
  note?: string;
  chart?: { x: string; series: Record<string, [number, number][]> } | null;
  metric?: "acc" | "target_ap";
  params: Record<string, unknown>;
  seconds: number;
}

/** Train and evaluate a runnable method with these parameters. */
export async function run(mid: string, params: Record<string, unknown>): Promise<RunResult> {
  await call("warm", {});
  const t0 = performance.now();
  const res = await call<Omit<RunResult, "seconds">>("run", { mid, params });
  return { ...res, seconds: (performance.now() - t0) / 1000 };
}

export interface CheckResult {
  passed: boolean;
  error?: string;
  tests: { name: string; doc?: string; ok: boolean; message?: string; trace?: string }[];
}

export const CHECK_LIMIT_S = 30;

/** Run a lesson's hidden tests against the visitor's code (loading Python doesn't count against the limit). */
export async function check(lid: string, code: string): Promise<CheckResult> {
  await call("warm", {});
  return call<CheckResult>("check", { lid, code }, CHECK_LIMIT_S * 1000);
}
