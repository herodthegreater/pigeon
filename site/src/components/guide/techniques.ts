/**
 * Techniques the Field Guide doesn't cover, or covers only in part. Each line was
 * checked against the method write-ups and the Coop's recipe editor on 2026-10-07.
 *
 * Link syntax inside `line`:
 *   [](method:id)         the method's name from the lab data, linked to its page
 *   [text](method:id)     the same link with custom text
 *   [text](coop:preset)   the Coop opened with that preset (?preset=<id>)
 * Unknown method ids or preset ids fail the build, so no link can point nowhere.
 */

import { esc } from "../../lib/lab/markdown";
import { methods } from "../../lib/lab/data";
import { PRESETS } from "../../lib/ui/recipe";

export type Coverage = "not_covered" | "partly";

export interface Technique {
  name: string;
  coverage: Coverage;
  line: string;
}

export const COVERAGE_LABEL: Record<Coverage, string> = {
  not_covered: "Not covered",
  partly: "Partly covered",
};

export const NOT_COVERED: Technique[] = [
  {
    name: "GGUF k-quants and the importance matrix",
    coverage: "not_covered",
    line:
      "No page covers k-quants, their super-blocks or the importance matrix. Their starting point, " +
      "round-to-nearest with one scale per group of weights, is on [](method:rtn) and [](method:ptq).",
  },
  {
    name: "FP8 (E4M3 and E5M2)",
    coverage: "not_covered",
    line:
      "No page covers FP8 yet. In the Coop, a Change number format step converts Pigeon's weights " +
      "(not its activations) to FP8 E4M3 or E5M2: [open the Coop](coop:) and add that step.",
  },
  {
    name: "FP4 microscaling (MXFP4 and NVFP4)",
    coverage: "partly",
    line:
      "[](method:conv_quantization) uses a 4-bit float grid (E2M1) with a scale per group of 32 or 16 weights, " +
      "but not the MX or NVFP4 scale formats. The Coop has both for Pigeon's weights: " +
      "[the MXFP4 preset](coop:mxfp4), and NVFP4 as a scaling option in its Change number format step.",
  },
  {
    name: "bitsandbytes NF4 and QLoRA",
    coverage: "partly",
    line:
      "[](method:qlora) runs the core idea, low-rank adapters trained on a frozen quantized model, but its base " +
      "is plain integers with a scale per 16 weights: NF4, double quantization and paged optimizers are explained " +
      "but not implemented. [](method:weight_sharing) mentions NF4's fixed codebook next to its own learned ones.",
  },
  {
    name: "Rotations (QuaRot, SpinQuant, ParoQuant)",
    coverage: "not_covered",
    line:
      "Rotation methods get one sentence each, in the further-reading notes of [](method:smoothquant) and " +
      "[](method:gptq). Hadamard transforms and ParoQuant are not explained. SmoothQuant tackles the same " +
      "outlier problem with per-channel scaling instead of a rotation.",
  },
  {
    name: "KV-cache quantization",
    coverage: "partly",
    line:
      "[](method:kv_cache_quant) explains KIVI and KVQuant but runs nothing: the lab's toy model has no " +
      "attention and no cache. [](method:smoothquant) does run the per-token activation quantizer it points to.",
  },
  {
    name: "MLX",
    coverage: "not_covered",
    line:
      "MLX isn't mentioned anywhere in the lab. Its default 4-bit scheme (a scale and a zero point per 64 weights) " +
      "can be rebuilt in the Coop with a Quantize to integers step at 4 bits, per group of 64, asymmetric, and its " +
      "mxfp4 mode matches [the Coop's MXFP4 preset](coop:mxfp4). Both reproduce the number formats, not MLX itself.",
  },
  {
    name: "2:4 sparsity",
    coverage: "partly",
    line:
      "[](method:sparsegpt), [](method:wanda) and [](method:magnitude_pruning) remove single weights with no " +
      "pattern; each mentions 2:4, but none simulates it. [The Coop's 2:4 sparsity preset](coop:sparse24) keeps " +
      "the 2 largest of every 4 weights in Pigeon, by magnitude, with no calibration and no retraining.",
  },
  {
    name: "TorchAO, LiteRT and Core ML",
    coverage: "not_covered",
    line:
      "None of these toolchains is covered. Some of the schemes they apply are: int8 and int4 weights on " +
      "[](method:ptq) and [](method:qat), k-means palettes on [](method:weight_sharing), and [](method:bn_folding). " +
      "The Coop applies two of them to Pigeon: [its INT8 per-channel preset](coop:int8-channel) and " +
      "[its 16-colour palette preset](coop:palette).",
  },
];

/**
 * Techniques the lab does cover, as one sentence with links to their pages. The
 * pruning-then-distillation pages listed are the ones whose Run has a KD recovery
 * option (checked in lab/methods.py on 2026-10-09); ShortGPT only deletes layers.
 */
export const COVERED =
  "Covered with pages that run here: [](method:awq), [](method:gptq), [](method:smoothquant); pruning followed " +
  "by distillation ([](method:magnitude_pruning), [](method:structured_pruning), [](method:layer_dropping) and " +
  "others in the families above); and quantization-aware distillation, the distillation option of [](method:qat).";

const LINK = /\[([^\]]*)\]\((method|coop):([\w-]*)\)/g;

/** Escape a line and turn its links into real hrefs, checking every target exists. */
export function renderLine(line: string, base: string): string {
  const names = new Map(methods().map((m) => [m.id, m.name]));
  const presets = new Set(PRESETS.map((p) => p.id));
  let out = "", last = 0, m: RegExpExecArray | null;
  const re = new RegExp(LINK.source, "g");
  while ((m = re.exec(line))) {
    out += esc(line.slice(last, m.index));
    const [, text, kind, target] = m;
    if (kind === "method") {
      const name = names.get(target);
      if (!name) throw new Error(`Field Guide: technique line links to unknown method "${target}"`);
      out += `<a href="${base}guide/${target}/">${esc(text || name)}</a>`;
    } else {
      if (target && !presets.has(target)) throw new Error(`Field Guide: technique line links to unknown Coop preset "${target}"`);
      out += `<a href="${base}coop/${target ? `?preset=${target}` : ""}">${esc(text)}</a>`;
    }
    last = re.lastIndex;
  }
  return out + esc(line.slice(last));
}
