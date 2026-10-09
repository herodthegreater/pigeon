/**
 * The Coop's recipe editor: presets plus a stack of editable steps.
 * Vanilla DOM; calls `onChange(recipe)` after every edit.
 */

import type { Granularity } from "../compress/quant";
import type { Kind, TensorMeta } from "../engine/model";
import { ALL_KINDS, type Recipe, type Step } from "../engine/pipeline";

export const PART_NAMES: Record<Kind, string> = {
  conv: "First conv",
  depthwise: "Depthwise",
  pointwise: "Pointwise",
  squeeze_excite: "Squeeze-excite",
  head: "Head",
};

const ALL = [...ALL_KINDS];
const chan: Granularity = { kind: "channel" };

export const PRESETS: { id: string; name: string; hint: string; recipe: Recipe }[] = [
  { id: "fp32", name: "Full precision", hint: "The untouched 32-bit Pigeon.", recipe: { steps: [] } },
  { id: "fp16", name: "FP16", hint: "Half-precision floats. Almost always free.", recipe: { steps: [{ op: "float", parts: ALL, format: "fp16", scaling: { kind: "none" } }] } },
  { id: "int8-tensor", name: "INT8, one scale", hint: "One scale for a whole tensor. Watch what happens.", recipe: { steps: [{ op: "int", parts: ALL, bits: 8, granularity: { kind: "tensor" }, symmetric: true }] } },
  { id: "int8-channel", name: "INT8, per channel", hint: "A scale for every output channel.", recipe: { steps: [{ op: "int", parts: ALL, bits: 8, granularity: chan, symmetric: true }] } },
  { id: "mixed", name: "Mixed precision", hint: "Squeeze-excite at 4-bit, everything else at 8-bit.", recipe: { steps: [
    { op: "int", parts: ["conv", "depthwise", "pointwise", "head"], bits: 8, granularity: chan, symmetric: true },
    { op: "int", parts: ["squeeze_excite"], bits: 4, granularity: chan, symmetric: true },
  ] } },
  { id: "mxfp4", name: "MXFP4", hint: "4-bit floats. Each block of up to 32 weights in an output channel shares a power-of-two scale.", recipe: { steps: [{ op: "float", parts: ALL, format: "fp4_e2m1", scaling: { kind: "mx", block: 32 } }] } },
  { id: "sparse24", name: "2:4 sparsity", hint: "Keep the biggest 2 of every 4 weights.", recipe: { steps: [{ op: "prune", parts: ALL, method: "nm", amount: 0.5 }] } },
  { id: "palette", name: "16-colour palette", hint: "k-means: every weight becomes one of 16 shared values.", recipe: { steps: [{ op: "cluster", parts: ALL, k: 16 }] } },
];

const clone = (r: Recipe): Recipe => JSON.parse(JSON.stringify(r));

export function describeStep(s: Step): string {
  const parts = s.parts.length === ALL.length ? "all" : s.parts.length ? s.parts.map((p) => PART_NAMES[p]).join(" + ") : "no layers";
  switch (s.op) {
    case "int": {
      if (s.bits >= 16) return `off (${parts})`;
      const g = s.granularity.kind === "group" ? `group ${s.granularity.size}` : `per ${s.granularity.kind}`;
      return `${s.ternary ? "ternary" : `INT${s.bits}`} ${g}${s.symmetric ? "" : " asym"} · ${parts}`;
    }
    case "float": {
      const sc = s.scaling.kind === "mx" ? ` MX/${s.scaling.block}` : s.scaling.kind === "nv" ? ` NV/${s.scaling.block}` : "";
      return `${s.format.toUpperCase().replace("_", " ")}${sc} · ${parts}`;
    }
    case "prune":
      return `${s.method === "nm" ? "2:4" : `${Math.round(s.amount * 100)}% ${s.method === "channels" ? "channels" : "pruned"}`} · ${parts}`;
    case "cluster":
      return `${s.k}-value palette · ${parts}`;
  }
}

export function describeRecipe(r: Recipe): string {
  return r.steps.length ? r.steps.map(describeStep).join(", then ") : "Full precision";
}

/** The preset a recipe matches, if any: the label covers every setting a step has, so equal labels are equal recipes. */
export function presetOf(r: Recipe): string | null {
  const label = describeRecipe(r);
  return PRESETS.find((p) => describeRecipe(p.recipe) === label)?.id ?? null;
}

const STEP_TITLES = { int: "Quantize to integers", float: "Change number format", prune: "Prune", cluster: "Palettize (k-means)" } as const;

function newStep(op: Step["op"]): Step {
  switch (op) {
    case "int": return { op, parts: [...ALL], bits: 8, granularity: chan, symmetric: true };
    case "float": return { op, parts: [...ALL], format: "fp8_e4m3", scaling: { kind: "tensor" } };
    case "prune": return { op, parts: [...ALL], method: "magnitude", amount: 0.2 };
    case "cluster": return { op, parts: [...ALL], k: 16 };
  }
}

/** What each float scaling stores, said once under its select so the option labels stay short enough for a phone. */
const SCALING_NOTE: Record<"none" | "tensor" | "mx" | "nv", string> = {
  none: "Each weight rounds to the nearest value the format holds, and anything past its largest value is clipped to it. No scale is stored.",
  tensor: "Each tensor is scaled so its largest weight lands on the format's largest value. That one scale is stored as 32 bits.",
  mx: "Each block of up to 32 weights in an output channel shares a power-of-two scale, stored as 8 bits (E8M0).",
  nv: "Each block of up to 16 weights in an output channel shares an FP8 (E4M3) scale, and each tensor adds one 32-bit scale.",
};

const NO_PRESET = "Pick a preset to load its steps, or build your own below.";
const presetHint = (id: string | null) => {
  const p = PRESETS.find((x) => x.id === id);
  return p ? `${p.name}: ${p.hint}` : NO_PRESET;
};

export class RecipeEditor {
  recipe: Recipe = { steps: [] };
  private shares: Record<Kind, number>;
  private active: string | null = "fp32"; // preset the recipe still matches; null once edited

  constructor(private root: HTMLElement, tensors: TensorMeta[], private onChange: (r: Recipe, label: string) => void) {
    const w = tensors.filter((t) => t.role === "weight");
    const total = w.reduce((a, t) => a + t.numel, 0);
    this.shares = Object.fromEntries(ALL.map((k) => [k, w.filter((t) => t.kind === k).reduce((a, t) => a + t.numel, 0) / total])) as Record<Kind, number>;
    root.addEventListener("input", (e) => this.onInput(e));
    root.addEventListener("click", (e) => this.onClick(e));
    // Hovering or focusing a preset previews its explanation; leaving shows the active one again.
    const preview = (e: Event) => this.setHint((e.target as HTMLElement).closest<HTMLElement>(".preset")?.dataset.preset ?? this.active);
    root.addEventListener("mouseover", preview);
    root.addEventListener("focusin", preview);
    root.addEventListener("mouseleave", () => this.setHint(this.active));
    root.addEventListener("focusout", () => this.setHint(this.active));
    this.render();
  }

  /** `focus` keeps focus on the clicked preset; a preset loaded from the URL leaves focus where it is. */
  load(r: Recipe, preset: string | null = null, focus = true) {
    this.recipe = clone(r);
    this.active = preset;
    this.render(preset && focus ? `[data-preset="${preset}"]` : this.focusKey());
    this.emit();
  }

  /** Load a preset by id, as if its button were clicked. Returns false for an unknown id. */
  loadPreset(id: string, focus = true): boolean {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return false;
    this.load(p.recipe, p.id, focus);
    return true;
  }

  private setHint(id: string | null) {
    const el = this.root.querySelector(".preset-hint");
    if (el) el.textContent = presetHint(id);
  }

  private emit() {
    this.onChange(clone(this.recipe), describeRecipe(this.recipe));
  }

  /** A selector for the focused control, so a re-render can put keyboard focus back on its replacement. */
  private focusKey(): string | null {
    const a = document.activeElement as HTMLElement | null;
    if (!a || !this.root.contains(a)) return null;
    const d = a.dataset;
    if (d.preset) return `[data-preset="${d.preset}"]`;
    if (d.add) return `[data-add="${d.add}"]`;
    if (d.i && d.f) return `[data-i="${d.i}"][data-f="${d.f}"]${a instanceof HTMLInputElement && (a.type === "radio" || a.type === "checkbox") ? `[value="${a.value}"]` : ""}`;
    return null;
  }

  private render(focus: string | null = this.focusKey()) {
    const presets = PRESETS.map((p) => `<button type="button" class="preset" data-preset="${p.id}" aria-pressed="${p.id === this.active}" aria-describedby="preset-hint">${p.name}</button>`).join("");
    const steps = this.recipe.steps.map((s, i) => this.stepHTML(s, i)).join("");
    this.root.innerHTML = `
      <div class="presets" role="group" aria-labelledby="presets-label"><span class="sub" id="presets-label">Presets</span><div class="preset-row">${presets}</div><p class="preset-hint" id="preset-hint">${presetHint(this.active)}</p></div>
      ${steps ? `<ol class="steps">${steps}</ol>` : `<p class="empty">No steps yet: this is the full-precision Pigeon. Add a step, or start from a preset.</p>`}
      <div class="add" role="group" aria-labelledby="add-label"><span class="sub" id="add-label">Add a step</span><div class="add-row">
        ${(Object.keys(STEP_TITLES) as Step["op"][]).map((op) => `<button type="button" class="btn secondary add-btn" data-add="${op}">+ ${STEP_TITLES[op]}</button>`).join("")}
      </div></div>`;
    if (focus) this.root.querySelector<HTMLElement>(focus)?.focus();
  }

  private partsHTML(s: Step, i: number) {
    return `<fieldset class="parts"><legend class="sub">Applies to</legend>${ALL.map(
      (k) => `<label class="part"><input type="checkbox" data-i="${i}" data-f="part" value="${k}" ${s.parts.includes(k) ? "checked" : ""}/>
        <span>${PART_NAMES[k]}</span><span class="share num">${(this.shares[k] * 100).toFixed(k === "head" || k === "conv" ? 2 : 0)}%</span></label>`,
    ).join("")}</fieldset>`;
  }

  private stepHTML(s: Step, i: number): string {
    const id = (f: string) => `s${i}-${f}`;
    let body = "";
    if (s.op === "int") {
      const g = s.granularity.kind;
      body = `
        <label for="${id("bits")}" class="field"><span>Bits per weight</span><output class="num">${s.ternary ? "1.58 (ternary)" : s.bits >= 16 ? "off" : s.bits}</output></label>
        <input id="${id("bits")}" type="range" min="1" max="8" step="1" value="${Math.min(s.bits, 8)}" data-i="${i}" data-f="bits"/>
        <fieldset class="gran"><legend class="sub">Scale granularity</legend><div class="seg">${(["tensor", "channel", "group"] as const).map(
          (k) => `<label><input type="radio" name="${id("g")}" value="${k}" data-i="${i}" data-f="gran" ${g === k ? "checked" : ""}/><span>${k === "tensor" ? "Per tensor" : k === "channel" ? "Per channel" : "Per group"}</span></label>`,
        ).join("")}</div></fieldset>
        ${g === "group" ? `<label class="field" for="${id("gs")}"><span>Group size</span><select id="${id("gs")}" data-i="${i}" data-f="gsize">${[16, 32, 64, 128].map((n) => `<option ${s.granularity.kind === "group" && s.granularity.size === n ? "selected" : ""}>${n}</option>`).join("")}</select></label>` : ""}
        <label class="check"><input type="checkbox" data-i="${i}" data-f="asym" ${s.symmetric ? "" : "checked"}/> Asymmetric (scale + zero-point)</label>
        <label class="check ternary"${s.bits === 2 ? "" : " hidden"}><input type="checkbox" data-i="${i}" data-f="ternary" ${s.ternary ? "checked" : ""}/> Ternary {−1, 0, +1} (BitNet-style)</label>`;
    } else if (s.op === "float") {
      const sc = s.scaling.kind;
      body = `
        <label class="field" for="${id("fmt")}"><span>Format</span><select id="${id("fmt")}" data-i="${i}" data-f="format">${[
          ["fp16", "FP16 (16 bits)"], ["bf16", "BF16 (16 bits)"], ["fp8_e4m3", "FP8 E4M3"], ["fp8_e5m2", "FP8 E5M2"], ["fp4_e2m1", "FP4 E2M1"],
        ].map(([v, n]) => `<option value="${v}" ${s.format === v ? "selected" : ""}>${n}</option>`).join("")}</select></label>
        <label class="field" for="${id("sc")}"><span>Scaling</span><select id="${id("sc")}" data-i="${i}" data-f="scaling" aria-describedby="${id("sc-note")}">${[
          ["none", "None (plain cast)"], ["tensor", "One scale per tensor"], ["mx", "MX, blocks of 32"], ["nv", "NVFP4, blocks of 16"],
        ].map(([v, n]) => `<option value="${v}" ${sc === v ? "selected" : ""}>${n}</option>`).join("")}</select></label>
        <p class="note" id="${id("sc-note")}">${SCALING_NOTE[sc]}</p>`;
    } else if (s.op === "prune") {
      body = `
        <label class="field" for="${id("m")}"><span>Method</span><select id="${id("m")}" data-i="${i}" data-f="method">${[
          ["magnitude", "Smallest weights (unstructured)"], ["nm", "2:4 pattern (semi-structured)"], ["channels", "Whole channels (structured)"],
        ].map(([v, n]) => `<option value="${v}" ${s.method === v ? "selected" : ""}>${n}</option>`).join("")}</select></label>
        ${s.method === "nm" ? "" : `<label for="${id("amt")}" class="field"><span>Remove</span><output class="num">${Math.round(s.amount * 100)}%</output></label>
        <input id="${id("amt")}" type="range" min="0" max="${s.method === "channels" ? 60 : 95}" step="5" value="${Math.round(s.amount * 100)}" data-i="${i}" data-f="amount"/>`}`;
    } else {
      body = `
        <label for="${id("k")}" class="field"><span>Palette size</span><output class="num">${s.k} values (${Math.log2(s.k)} bits)</output></label>
        <input id="${id("k")}" type="range" min="1" max="8" step="1" value="${Math.log2(s.k)}" data-i="${i}" data-f="k"/>`;
    }
    return `<li class="step" data-op="${s.op}">
      <div class="step-head"><span class="step-n num">${i + 1}</span><h3>${STEP_TITLES[s.op]}</h3>
        <button type="button" class="btn secondary remove" data-remove="${i}" aria-label="Remove step ${i + 1}">Remove</button></div>
      <div class="step-body">${body}${this.partsHTML(s, i)}</div></li>`;
  }

  private onClick(e: Event) {
    const t = (e.target as HTMLElement).closest("button");
    if (!t) return;
    if (t.dataset.preset) return void this.loadPreset(t.dataset.preset);
    this.active = null;
    if (t.dataset.add) {
      this.recipe.steps.push(newStep(t.dataset.add as Step["op"]));
      this.render(`[data-i="${this.recipe.steps.length - 1}"]`); // focus moves to the new step's first control
      return this.emit();
    }
    if (t.dataset.remove) {
      const i = Number(t.dataset.remove), steps = this.recipe.steps;
      steps.splice(i, 1);
      // focus moves to the step that took its place, else the one above, else the first "add" button
      this.render(steps.length ? `[data-remove="${Math.min(i, steps.length - 1)}"]` : ".add-btn");
      return this.emit();
    }
  }

  private onInput(e: Event) {
    const el = e.target as HTMLInputElement;
    const i = Number(el.dataset.i), f = el.dataset.f;
    const s = this.recipe.steps[i];
    if (!s || !f) return;
    this.active = null;
    let rerender = false;
    if (f === "part") {
      s.parts = [...this.root.querySelectorAll<HTMLInputElement>(`input[data-i="${i}"][data-f="part"]:checked`)].map((x) => x.value as Kind);
    } else if (s.op === "int") {
      if (f === "bits") {
        // The ternary option only exists at 2 bits. It is shown or hidden in place: rebuilding the editor
        // here would replace the slider under the pointer and end the drag at 2.
        s.bits = Number(el.value);
        if (s.bits !== 2) s.ternary = false;
        const tern = el.closest(".step")?.querySelector<HTMLInputElement>('[data-f="ternary"]');
        if (tern) {
          tern.checked = !!s.ternary;
          tern.closest("label")!.hidden = s.bits !== 2;
        }
      }
      if (f === "gran") { s.granularity = el.value === "group" ? { kind: "group", size: 32 } : { kind: el.value as "tensor" | "channel" }; rerender = true; }
      if (f === "gsize") s.granularity = { kind: "group", size: Number(el.value) };
      if (f === "asym") s.symmetric = !el.checked;
      if (f === "ternary") s.ternary = el.checked;
      const out = el.closest(".step")?.querySelector("output");
      if (out && f !== "gran") out.textContent = s.ternary ? "1.58 (ternary)" : String(s.bits);
    } else if (s.op === "float") {
      if (f === "format") s.format = el.value as typeof s.format;
      if (f === "scaling") {
        s.scaling = el.value === "mx" ? { kind: "mx", block: 32 } : el.value === "nv" ? { kind: "nv", block: 16 } : { kind: el.value as "none" | "tensor" };
        const note = el.closest(".step")?.querySelector(".note");
        if (note) note.textContent = SCALING_NOTE[s.scaling.kind];
      }
    } else if (s.op === "prune") {
      if (f === "method") { s.method = el.value as typeof s.method; if (s.method === "channels") s.amount = Math.min(s.amount, 0.6); rerender = true; }
      if (f === "amount") { s.amount = Number(el.value) / 100; el.closest(".step")!.querySelector("output")!.textContent = `${el.value}%`; }
    } else if (s.op === "cluster" && f === "k") {
      s.k = 2 ** Number(el.value);
      el.closest(".step")!.querySelector("output")!.textContent = `${s.k} values (${el.value} bits)`;
    }
    if (rerender) this.render();
    else this.root.querySelectorAll(".preset[aria-pressed='true']").forEach((b) => b.setAttribute("aria-pressed", "false"));
    this.setHint(null);
    this.emit();
  }
}
