/**
 * The Field Guide's own numbers, counted from the lab data at build time: how many
 * methods run where, and which pages really show code. The index's lede is built
 * from them, so it can't claim code a page doesn't have.
 */

import { method, methods, type Status } from "../../lib/lab/data";

export interface CodeCounts {
  total: number;
  runnable: number;
  cockpit: number;
  reference: number;
  /** Runnable and cockpit pages that show the code that runs them. */
  runsWithCode: number;
  /** Reference pages that show the closest code that does run, and those with no code. */
  referenceWithCode: number;
  referenceWithoutCode: number;
}

export function codeCounts(): CodeCounts {
  const all = methods();
  const by = (s: Status) => all.filter((m) => m.status === s);
  const hasCode = (id: string) => method(id).code.length > 0;
  const runs = [...by("runnable"), ...by("cockpit")];
  const ref = by("reference");
  return {
    total: all.length,
    runnable: by("runnable").length,
    cockpit: by("cockpit").length,
    reference: ref.length,
    runsWithCode: runs.filter((m) => hasCode(m.id)).length,
    referenceWithCode: ref.filter((m) => hasCode(m.id)).length,
    referenceWithoutCode: ref.filter((m) => !hasCode(m.id)).length,
  };
}

const n = (k: number, one: string, many: string) => `${k} ${k === 1 ? one : many}`;
const v = (k: number, one: string, many: string) => (k === 1 ? one : many);

export function guideLede(c: CodeCounts): { lede: string; description: string } {
  const runs = c.runnable + c.cockpit;
  const where =
    `${c.runnable} ${v(c.runnable, "runs", "run")} in your browser and ${c.cockpit} in the YOLOv8n cockpit on your own machine`;
  const code =
    c.runsWithCode === runs
      ? runs === 1 ? "shown with the source code that runs it" : "each shown with the source code that runs it"
      : `${c.runsWithCode} of those ${runs} shown with the source code that runs them`;

  let refs = "";
  if (c.reference > 0) {
    const shows =
      c.referenceWithoutCode === 0
        ? c.reference === 1 ? "it shows the closest code that does run" : "each shows the closest code that does run"
        : c.referenceWithCode === 0
          ? c.reference === 1 ? "it has no code" : "none of them has code"
          : `${c.referenceWithCode} ${v(c.referenceWithCode, "shows", "show")} the closest code that does run, ` +
            `and ${c.referenceWithoutCode} ${v(c.referenceWithoutCode, "has", "have")} no code at all`;
    refs =
      ` The other ${n(c.reference, "is a reference write-up", "are reference write-ups")} for kinds of models ` +
      `the lab doesn't have: ${shows}.`;
  }

  const lede = `${n(c.total, "compression method", "compression methods")}. ${where}, ${code}.${refs}`;
  const description =
    `${n(c.total, "model compression method", "model compression methods")}: ${where}` +
    (c.reference > 0 ? `; ${n(c.reference, "is a reference write-up", "are reference write-ups")}.` : ".");
  return { lede, description };
}
