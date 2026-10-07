/**
 * The lab's content, read at build time from the JSON that
 * `python -m lab.web.build_static --data-out site/src/data/lab` writes
 * (git-ignored; CI builds it, and `--data-only` builds it locally in a second).
 */

export type Family = "small" | "llm" | "vision";
export type Status = "runnable" | "cockpit" | "reference";

export interface Meta {
  families: Record<Family, { title: string; blurb: string }>;
  categories: Record<string, string>;
  counts: Record<Family, Record<Status, number>>;
}

export interface MethodSummary {
  id: string;
  name: string;
  family: Family;
  category: string;
  status: Status;
  summary: string;
}

export interface Param {
  name: string;
  label: string;
  kind: "int" | "float" | "choice" | "bool";
  default: number | string | boolean;
  lo: number | null;
  hi: number | null;
  step: number | null;
  choices: string[] | null;
  help: string;
}

export interface CodeRef {
  ref: string;
  file: string;
  name: string;
  line: number;
  end_line: number;
  source: string;
  doc: string;
}

export interface ExperimentSummary {
  name: string;
  title: string;
  question: string;
  group: string;
  seconds: number;
  figures: string[]; // "figures/<file>.png", relative to the lab's runtime folder
  notes: string;
  methods?: string[];
}

export interface Method extends MethodSummary {
  page: string; // markdown
  papers: string[];
  params: Param[];
  code: CodeRef[];
  experiments: ExperimentSummary[];
  lessons: { id: string; title: string }[];
}

export interface LessonSummary {
  id: string;
  title: string;
  methods: string[];
}

export interface Lesson {
  id: string;
  title: string;
  intro: string; // markdown
  skeleton: string;
  hints: string[];
  methods: MethodSummary[];
  prev: string | null;
  next: string | null;
}

export interface Experiment extends ExperimentSummary {
  rows: Record<string, number | string>[];
  refs: Record<string, Record<string, number>>;
  methods: string[];
}

type Json = Record<string, unknown>;
const files = import.meta.glob<Json>("../../data/lab/**/*.json", { eager: true, import: "default" });

function read<T>(rel: string): T {
  const hit = files[`../../data/lab/${rel}.json`];
  if (!hit) {
    throw new Error(
      `Lab data "${rel}.json" is missing. Build it with:\n` +
        "  python -m lab.web.build_static --out site/public/lab --data-out site/src/data/lab --no-frontend --data-only\n" +
        "(from the repo root; drop --data-only for the full build that Run and Check need)",
    );
  }
  return hit as T;
}

export const meta = () => read<Meta>("meta");
export const methods = () => read<MethodSummary[]>("methods");
export const method = (id: string) => read<Method>(`methods/${id}`);
export const lessons = () => read<LessonSummary[]>("lessons");
export const lesson = (id: string) => read<Lesson>(`lessons/${id}`);
export const solution = (id: string) => read<{ solution: string }>(`lessons/${id}/solution`).solution;
export const experiments = () => read<(ExperimentSummary & { methods: string[] })[]>("experiments");
export const experiment = (name: string) => read<Experiment>(`experiments/${name}`);
export const report = () => read<{ text: string }>("report").text;

export const FAMILY_ORDER: Family[] = ["small", "llm", "vision"];
export const STATUS_ORDER: Record<Status, number> = { runnable: 0, cockpit: 1, reference: 2 };

/** Plain words for a method's status, used as a label (never as a badge). */
export const STATUS_LABEL: Record<Status, string> = {
  runnable: "Runs here",
  cockpit: "Runs in the cockpit",
  reference: "Reference",
};

const base = () => import.meta.env.BASE_URL.replace(/\/?$/, "/");
/** URL of a lab figure ("figures/x.png" in the data) on the site. */
export const figureUrl = (rel: string) => `${base()}lab/${rel.replace(/^\/?/, "")}`;
