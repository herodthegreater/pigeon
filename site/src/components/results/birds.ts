/**
 * The birds an experiment's table hatches (DESIGN.md: Motif), as a key under
 * the table. They are the lab's recorded results, measured against the
 * experiment's own teacher, and they never go on anyone's life list.
 *
 * A bird needs each model's storage, and the lab has one storage rule
 * (lab/compress.py storage_bits). A row's storage is its kbits column where
 * the table records one. Without that column, only a dense fp32 net's storage
 * can be worked out here: storage_bits counts every weight and bias at 32
 * bits, which is the row's parameter count times 32 (a trained dense net has
 * no zero weights, so the table's nonzero count is every parameter).
 * DENSE_FP32 names the experiments whose rows lab/experiments/*.py builds that
 * way. A masked (pruned) or quantized row without kbits gets no bird: its
 * storage depends on a format the table doesn't record.
 *
 * Scored by labBirds() in lib/ui/birds.ts, on a view of the table built here.
 */

import { fmt } from "../../lib/lab/results";
import { METRICS, cap, label } from "../../lib/lab/labels";
import { bestBird, labBirds, type LabBird } from "../../lib/ui/birds";
import { SPECIES, type Condition } from "../../lib/ui/mascot";
import { metricWord, pct, ratioShort } from "../birds/format";
import { cellText, columns } from "./table";

type Row = Record<string, number | string>;

export interface ExperimentLike {
  name: string;
  rows: Row[];
  refs?: Record<string, Record<string, number>>;
}

interface Dense {
  /** Which rows are dense fp32 nets. */
  rows: (r: Row) => boolean;
  /** Why the other model rows can't have a bird, when an experiment has some. */
  others?: string;
  /** Which net the teacher is, when the table holds more than one big net, and what that means for the others. */
  teacher?: { name: string; note: string };
}

const every: Dense = { rows: () => true };

/** Experiments whose model rows are dense fp32 nets, read off lab/experiments/*.py. */
export const DENSE_FP32: Record<string, Dense> = {
  // distill.py, teacher.py, transfer.py, small.py: students and teachers trained
  // from scratch or by distillation (small.py's pruned and collapsed students
  // are cut down by whole neurons, which leaves a dense net).
  size_ladder: every,
  temperature_alpha: every,
  loss_zoo: every,
  label_budget: every,
  label_noise: every,
  ensemble_teacher: every,
  oracle_teacher: every,
  fidelity_gap: every,
  biased_teacher: every,
  transfer_set: every,
  function_matching: every,
  small_tricks: every,
  // specialist.py: neuron pruning and collapsed heads leave dense nets.
  specialist_size: every,
  specialist_prune: every,
  specialist_data: every,
  // shrink.py: neuron pruning, SVD factors and deleted layers leave dense nets...
  prune_structured: every,
  low_rank: every,
  // Its reference points (refs.teacher) are the plain deep net's; the residual one is a row.
  layer_drop: {
    ...every,
    teacher: { name: "the plain 8x128 net", note: "The residual rows are measured against it too, as the reference points are." },
  },
  // ...magnitude pruning keeps a mask, and only the dense small nets compared with it are dense.
  prune_unstructured: {
    rows: (r) => typeof r.dense_width === "number",
    others: "a magnitude-pruned net also stores where its remaining weights sit, and this table doesn't record that storage",
  },
};

/** The score an experiment is about: target-class AP for the one-class specialists, accuracy elsewhere. */
const METRIC: Record<string, "acc" | "target_ap"> = {
  specialist_size: "target_ap",
  specialist_prune: "target_ap",
  specialist_data: "target_ap",
};

/** Every lab teacher is a dense fp32 net (common.teacher trains one), so its storage is 32 bits a parameter. */
const FP32 = 32;
export const denseKbits = (params: number) => (params * FP32) / 1000;

/**
 * A row's storage in kilobits: its kbits column, or 32 bits a parameter when
 * the experiment builds the row as a dense fp32 net. A dense net has the same
 * parameter count on every seed, so a spread in the count means it isn't one.
 */
export function rowKbits(name: string, r: Row): number | null {
  if (typeof r.kbits === "number") return r.kbits > 0 ? r.kbits : null;
  const dense = DENSE_FP32[name];
  if (!dense || !dense.rows(r)) return null;
  if (typeof r.params !== "number" || !(r.params > 0)) return null;
  if (typeof r.params_std === "number" && r.params_std !== 0) return null;
  return denseKbits(r.params);
}

/** Measurements the lab keeps beside a row's settings (ensemble accuracy, teacher NLL): not part of its name. */
const MEASURED = /(^|_)(acc|nll|ece|kl_bayes)$/;

/** A row's name, from the settings the table shows for it: "Hidden width: 64; method: KD, + unlabeled pool". */
export function rowLabel(r: Row, settings: string[]): string {
  const parts = settings
    .filter((k) => r[k] !== undefined && r[k] !== "")
    .map((k) => `${label(k)}: ${cellText(r[k], k)}`);
  return cap(parts.join("; "));
}

export interface KeyBird {
  bird: LabBird; // the best model of its kind
  count: number; // how many of the table's smaller models hatch this kind of bird
}

export interface ExperimentBirds {
  metric: "acc" | "target_ap";
  reference: { params: number; kbits: number; score: number };
  models: number; // distinct smaller models with a bird
  repeats: number; // rows that repeat an earlier model's size and score, so count once
  kinds: KeyBird[]; // one per species and condition, biggest species first
  source: "kbits" | "dense" | "both"; // where the drawn models' storage comes from
  skipped: number; // smaller-looking model rows left out because their storage isn't known
  skippedWhy?: string;
  teacher?: { name: string; note: string }; // which net the teacher is, when the table has more than one big net
}

const ORDER = SPECIES.map((s) => s.id);
const rank = (species: string) => (species === "dodo" ? 0 : ORDER.indexOf(species));
const CONDITIONS: Condition[] = ["healthy", "emaciated", "skeleton"];

/**
 * The key for one experiment, or null when its data can't support an honest
 * bird: no teacher to measure against, no storage, or nothing smaller.
 */
export function experimentBirds(e: ExperimentLike): ExperimentBirds | null {
  const metric = METRIC[e.name] ?? "acc";
  const t = e.refs?.teacher;
  if (!t || !(t.params > 0) || !(t[metric] > 0)) return null;
  const reference = { params: t.params, kbits: denseKbits(t.params), score: t[metric] };

  const settings = columns(e.rows).filter((k) => !METRICS.includes(k) && !MEASURED.test(k));
  const isModel = (r: Row) => typeof r[metric] === "number" && typeof r.params === "number";
  const sized = e.rows.filter(isModel).map((r) => ({ r, label: rowLabel(r, settings), kbits: rowKbits(e.name, r) }));
  const known = sized.filter((x): x is typeof x & { kbits: number } => x.kbits !== null);
  const scored = labBirds({
    metric,
    rows: [
      { label: "the teacher", kbits: reference.kbits, [metric]: reference.score, ref: true },
      ...known.map((x) => ({ label: x.label, kbits: x.kbits, [metric]: x.r[metric] })),
    ],
  });
  if (!scored.bird) return null;

  const groups = new Map<string, LabBird[]>();
  for (const b of scored.birds) {
    const key = `${b.species}|${b.condition}`;
    groups.set(key, [...(groups.get(key) ?? []), b]);
  }
  const kinds = [...groups.values()]
    .map((bs) => ({ bird: bestBird(bs) as LabBird, count: bs.length }))
    .sort((a, b) => rank(a.bird.species) - rank(b.bird.species) || CONDITIONS.indexOf(a.bird.condition) - CONDITIONS.indexOf(b.bird.condition));

  const drawn = new Set(scored.birds.map((b) => b.label));
  const fromKbits = known.filter((x) => drawn.has(x.label)).map((x) => typeof x.r.kbits === "number");
  const source = fromKbits.every(Boolean) ? "kbits" : fromKbits.some(Boolean) ? "both" : "dense";
  // Rows that hatched a bird, before a row repeating an earlier model (same size, same score) counts once.
  const smaller = known.filter((x) => x.kbits < reference.kbits).length;
  // Model rows the page can't size, with fewer parameters than the teacher.
  const skipped = sized.filter((x) => x.kbits === null && (x.r.params as number) < reference.params).length;
  return {
    metric, reference, models: scored.birds.length, repeats: smaller - scored.birds.length, kinds, source,
    skipped, skippedWhy: DENSE_FP32[e.name]?.others, teacher: DENSE_FP32[e.name]?.teacher,
  };
}

const plural = (n: number, one: string, many: string) => `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;

export interface KeyEntry {
  ratio: number;
  starve: number;
  name: string; // "Barn Swallow", or "Dodo"
  condition: Condition;
  label: string; // the best model's row
  nums: string; // "13.4× smaller, keeps 97.5% of the teacher's accuracy"
  count: string; // "Best of 4 models", or "The only model of its kind"
}

/** The key's words, in the order the page shows them. */
export function keyWords(b: ExperimentBirds) {
  const what = metricWord(b.metric);
  const kinds = b.kinds.length;
  // Within one kind every bird has the same condition, so bestBird() picks the smallest
  // healthy or bones model, and the starving model that keeps the most.
  const intro =
    `The table's ${plural(b.models, "smaller model hatches", "smaller models hatch")} ${plural(kinds, "kind of bird", "kinds of bird")}.` +
    (b.models > kinds ? " Each kind is drawn once, for one model: the smallest if the bird is healthy or bones, the one that keeps the most if it's starving." : "") +
    (b.repeats > 0 ? ` Rows with the same size and score count as one model.` : "");
  const entries: KeyEntry[] = b.kinds.map(({ bird, count }) => ({
    ratio: bird.ratio,
    starve: bird.starve,
    name: bird.name,
    condition: bird.condition,
    label: bird.label,
    nums: `${ratioShort(bird.ratio)}, ${bird.kept > 1.0005 ? "scores" : "keeps"} ${pct(bird.kept)} of the teacher's ${what}`,
    count: count > 1 ? `Best of ${count.toLocaleString("en-US")} models` : "The only model of its kind",
  }));
  const storage = {
    kbits: "Storage is the table's own column.",
    dense: "Storage is 32 bits a parameter, the lab's rule for a dense fp32 net, which every model drawn here is.",
    both: "Storage is the table's own column where it has one, and otherwise 32 bits a parameter for a dense fp32 net.",
  }[b.source];
  const reference =
    `Measured against this experiment's teacher${b.teacher ? `, ${b.teacher.name}` : ""}: ${fmt(b.reference.params, "params")} parameters, ` +
    `${fmt(b.reference.kbits, "kbits")} kbit, ${what} ${fmt(b.reference.score)}.${b.teacher ? ` ${b.teacher.note}` : ""} ${storage}`;
  const skipped = b.skipped
    ? `${cap(plural(b.skipped, "other model has", "other models have"))} no bird: ${b.skippedWhy ?? "this table doesn't record their storage"}.`
    : "";
  return { intro, entries, reference, skipped };
}
