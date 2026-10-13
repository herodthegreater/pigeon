/**
 * Flight School's twelve chapters as a reading map. Each chapter cites sections
 * of the Field Guide's method write-ups and the coding lessons that teach it
 * (verified against the write-ups on 2026-10-07), plus a Coop preset where the
 * idea can be tried on Pigeon's real weights. `gap` says what is not written
 * yet, for chapters the write-ups only partly cover.
 */

import { lessons, method, methods } from "../../lib/lab/data";
import { slug } from "../../lib/lab/markdown";

export interface ChapterSource {
  n: number;
  title: string;
  subtitle: string;
  reading: { method: string; section: string }[];
  lessons: string[];
  coop: { preset: string; label: string }[];
  gap?: string;
}

export const CHAPTERS: ChapterSource[] = [
  {
    n: 1,
    title: "What a weight is",
    subtitle: "FP32, FP16 and BF16: bits, range and precision",
    reading: [],
    lessons: [],
    coop: [{ preset: "fp16", label: "FP16" }],
    gap: "Not written yet: how FP32, FP16 and BF16 lay out their bits, and how they trade range for precision.",
  },
  {
    n: 2,
    title: "Scale and zero-point",
    subtitle: "The one formula behind all quantization",
    reading: [
      { method: "ptq", section: "How it works" },
      { method: "rtn", section: "How it works" },
      { method: "conv_quantization", section: "How it works" },
      { method: "kv_cache_quant", section: "How it works" },
    ],
    lessons: ["fake_quant"],
    coop: [],
    gap: "Not written yet: the zero point, for ranges that aren't centered on zero, with a worked example and an exercise.",
  },
  {
    n: 3,
    title: "Bits and rounding",
    subtitle: "INT8, INT4, INT2 and binary",
    reading: [
      { method: "ptq", section: "How it works" },
      { method: "rtn", section: "How it works" },
      { method: "conv_quantization", section: "How it works" },
    ],
    lessons: ["fake_quant"],
    coop: [],
  },
  {
    n: 4,
    title: "Granularity",
    // The syllabus called this "the knob that matters most"; no write-up backs that ranking, so it is not claimed here.
    subtitle: "Per tensor, per channel, per group: how many scales a tensor gets",
    reading: [
      { method: "ptq", section: "The idea" },
      { method: "rtn", section: "How it works" },
      { method: "rtn", section: "What the simulations show" },
      { method: "conv_quantization", section: "When it breaks" },
      { method: "smoothquant", section: "How it works" },
    ],
    lessons: ["fake_quant"],
    coop: [
      { preset: "int8-tensor", label: "INT8 with one scale" },
      { preset: "int8-channel", label: "INT8 per channel" },
    ],
  },
  {
    n: 5,
    title: "Outliers",
    subtitle: "Why a few huge values ruin everything",
    reading: [
      { method: "smoothquant", section: "The idea" },
      { method: "rtn", section: "When it breaks" },
      { method: "ptq", section: "When it breaks" },
      { method: "conv_quantization", section: "How it works" },
      { method: "kv_cache_quant", section: "When it breaks" },
    ],
    lessons: [],
    coop: [],
  },
  {
    n: 6,
    title: "Activations and calibration",
    subtitle: "Weight-only vs W8A8, static vs dynamic",
    reading: [
      { method: "smoothquant", section: "The idea" },
      { method: "smoothquant", section: "How it works" },
      { method: "smoothquant", section: "The code, line by line" },
      { method: "kv_cache_quant", section: "The code, line by line" },
      { method: "gptq", section: "When it breaks" },
      { method: "awq", section: "How it works" },
    ],
    lessons: [],
    coop: [],
    gap: "Not written yet: static activation scales fitted on a calibration set, and when to choose weight-only over W8A8.",
  },
  {
    n: 7,
    title: "Sensitivity and mixed precision",
    subtitle: "Not every layer is equal",
    reading: [
      { method: "conv_quantization", section: "When it breaks" },
      { method: "rtn", section: "How it works" },
    ],
    lessons: [],
    coop: [{ preset: "mixed", label: "mixed precision" }],
    gap: "Not written yet: how to measure each layer's sensitivity, and how to pick bit widths that fit a size budget.",
  },
  {
    n: 8,
    title: "Codebooks",
    subtitle: "Non-uniform levels and palettes",
    reading: [
      { method: "weight_sharing", section: "The idea" },
      { method: "qlora", section: "How it works" },
    ],
    lessons: [],
    coop: [{ preset: "palette", label: "a 16-colour palette" }],
    gap: "Not written yet: fixed non-uniform grids such as NF4 and FP4, and how they compare with learned palettes.",
  },
  {
    n: 9,
    title: "Pruning",
    subtitle: "Unstructured, 2:4 and structured, and the cliff",
    reading: [
      { method: "magnitude_pruning", section: "The idea" },
      { method: "structured_pruning", section: "What the simulations show" },
      { method: "channel_pruning", section: "When it breaks" },
    ],
    lessons: ["magnitude_mask", "neuron_surgery"],
    coop: [{ preset: "sparse24", label: "2:4 sparsity" }],
    gap: "Not written yet: 2:4 sparsity, how its mask is chosen and why sparse tensor cores run it faster.",
  },
  {
    n: 10,
    title: "Recovery",
    subtitle: "Fine-tuning and quantization-aware training",
    reading: [
      { method: "qat", section: "The idea" },
      { method: "prune_and_regrow", section: "The idea" },
      { method: "magnitude_pruning", section: "The idea" },
      { method: "specialist_data", section: "What the simulations show" },
    ],
    lessons: [],
    coop: [],
  },
  {
    n: 11,
    title: "Distillation",
    subtitle: "A big teacher trains a small student",
    reading: [{ method: "kd", section: "The idea" }],
    lessons: ["kd_loss"],
    coop: [],
  },
  {
    n: 12,
    title: "Low-rank",
    subtitle: "One big matrix becomes two thin ones",
    reading: [{ method: "low_rank", section: "The idea" }],
    lessons: ["low_rank_svd"],
    coop: [],
  },
];

const base = () => import.meta.env.BASE_URL.replace(/\/?$/, "/");

/** "7. Fake quantization with group scales" -> "Fake quantization with group scales". */
export const bareTitle = (title: string) => title.replace(/^\d+\.\s*/, "");

export const lessonHref = (id: string) => `${base()}school/${id}/`;
export const guideHref = (id: string, section?: string) => `${base()}guide/${id}/${section ? `#${slug(section)}` : ""}`;
export const coopHref = (preset: string) => `${base()}coop/?preset=${encodeURIComponent(preset)}`;

/** The "## " headings of a method write-up, as the Field Guide renders them (ids are slug(heading)). */
export function sectionsOf(methodId: string): string[] {
  return method(methodId)
    .page.split("\n")
    .filter((l) => l.startsWith("## "))
    .map((l) => l.slice(3).trim());
}

export interface ChapterLink { href: string; label: string; kind: "read" | "code" | "try" }
export interface Chapter { n: number; title: string; subtitle: string; gap?: string; links: ChapterLink[]; dropped: string[] }

/**
 * Resolve every chapter's citations to links. A citation whose method, section
 * or lesson no longer exists is dropped (and reported in `dropped`), so the page
 * never links to an anchor that isn't there.
 */
export function course(): Chapter[] {
  const names = new Map(methods().map((m) => [m.id, m.name]));
  const order = lessons();
  const lessonIndex = new Map(order.map((l, i) => [l.id, i]));
  const sections = new Map<string, string[]>();
  const sectionList = (id: string) => {
    if (!sections.has(id)) sections.set(id, names.has(id) ? sectionsOf(id) : []);
    return sections.get(id)!;
  };

  return CHAPTERS.map((c) => {
    const links: ChapterLink[] = [];
    const dropped: string[] = [];
    for (const r of c.reading) {
      if (names.has(r.method) && sectionList(r.method).includes(r.section)) {
        links.push({ kind: "read", href: guideHref(r.method, r.section), label: `${names.get(r.method)}: ${r.section}` });
      } else dropped.push(`${r.method}#${r.section}`);
    }
    const coded = c.lessons.filter((id) => (lessonIndex.has(id) ? true : (dropped.push(`lesson ${id}`), false)));
    coded.sort((a, b) => lessonIndex.get(a)! - lessonIndex.get(b)!);
    for (const id of coded) {
      links.push({ kind: "code", href: lessonHref(id), label: `Code it: ${bareTitle(order[lessonIndex.get(id)!].title)}` });
    }
    for (const p of c.coop) links.push({ kind: "try", href: coopHref(p.preset), label: `Try it in the Coop: ${p.label}` });
    if (dropped.length) console.warn(`Flight School chapter ${c.n}: dropped missing citations ${dropped.join(", ")}`);
    return { n: c.n, title: c.title, subtitle: c.subtitle, gap: c.gap, links, dropped };
  });
}
