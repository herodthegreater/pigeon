/**
 * The YOLOv8n cage's recipe editor: presets, one precision setting for the
 * whole model, and a channel cut per backbone and neck stage. Vanilla DOM, the
 * same look as the Pigeon cage's editor (lib/ui/recipe.ts). The markup is drawn
 * once; edits update it in place, so keyboard focus never moves. Calls
 * `onChange(recipe, label)` after every edit (sliders settle for a moment first).
 */

import { fpGrid } from "../detector/quant";
import { describe, GROUP_SIZES, keyOf, layerName, MAX_MASK, nativeRecipe, PRESETS, type DetectorRecipe } from "../detector/recipe";
import type { PruneStage } from "../detector/engine";

export type Stage = PruneStage & { kind: string };

const BITS = [1, 2, 3, 4, 5, 6, 7, 8, 16]; // slider positions 1..9
const clone = (r: DetectorRecipe): DetectorRecipe => JSON.parse(JSON.stringify(r));
const pct = (v: number) => `${Math.round(v * 100)}%`;
const SETTLE_MS = 300;

const bitsText = (b: number) => (b >= 16 ? "16, native" : String(b));
const bitsSpoken = (b: number) => (b >= 16 ? "16 bits, the native fp16 weights" : `${b} bit${b > 1 ? "s" : ""} per weight`);

function precisionNote(r: DetectorRecipe): string {
  if (r.bits >= 16) return "At 16 bits nothing is rounded: these are the native fp16 weights. Move the slider to 8 or fewer to choose how weights are rounded.";
  if (r.bits === 1) return "At 1 bit each weight becomes plus or minus the mean size of the weights that share its scale, so the grid doesn't apply.";
  const levels = r.levels === "fp" ? fpGrid(r.bits).length * 2 - 1 : 2 ** r.bits - 1;
  const who = r.granularity === "tensor" ? "one scale for each whole tensor" : r.granularity === "channel" ? "a scale for each output channel" : `a scale for each block of up to ${r.group} weights in an output channel`;
  const grid = r.levels === "fp" ? "float levels, packed close together near zero" : "evenly spaced levels";
  return `Each weight rounds to one of ${levels} ${grid}, with ${who}. Every scale is stored too, as 16 bits.`;
}

export class YoloRecipeEditor {
  recipe = nativeRecipe();
  private timer: ReturnType<typeof setTimeout> | undefined;

  constructor(private root: HTMLElement, private stages: Stage[], private onChange: (r: DetectorRecipe, label: string) => void) {
    this.render();
    root.addEventListener("input", (e) => this.onInput(e));
    root.addEventListener("click", (e) => this.onClick(e));
    // Hovering or focusing a preset previews its explanation; leaving shows the active one again.
    const preview = (e: Event) => this.hint((e.target as HTMLElement).closest<HTMLElement>(".preset")?.dataset.preset ?? this.activePreset());
    root.addEventListener("mouseover", preview);
    root.addEventListener("focusin", preview);
    root.addEventListener("mouseleave", () => this.hint(this.activePreset()));
    root.addEventListener("focusout", () => this.hint(this.activePreset()));
  }

  /** Load a recipe as if each control had been set by hand, and score it. */
  load(r: DetectorRecipe) {
    this.recipe = clone(r);
    this.sync();
    this.emit();
  }

  /** Move keyboard focus to the first control (the bits slider), e.g. after a reset disables the button that had it. */
  focusFirst() {
    this.root.querySelector<HTMLElement>("#y-bits")?.focus();
  }

  /** Load a preset by id, as if its button were pressed. Returns false for an unknown id. */
  loadPreset(id: string): boolean {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return false;
    this.load(p.recipe);
    return true;
  }

  private activePreset(): string | null {
    const k = keyOf(this.recipe);
    return PRESETS.find((p) => keyOf(p.recipe) === k)?.id ?? null; // "native" is a preset too, with no button: Reset does its job
  }

  private hint(id: string | null) {
    const el = this.root.querySelector(".preset-hint");
    if (!el) return;
    const p = PRESETS.find((x) => x.id === id);
    el.textContent = p ? `${p.name}: ${p.hint}` : "Your own recipe. Pick a preset to start from one of these instead.";
  }

  private emit(settle = false) {
    clearTimeout(this.timer);
    const go = () => this.onChange(clone(this.recipe), describe(this.recipe));
    if (settle) this.timer = setTimeout(go, SETTLE_MS);
    else go();
  }

  private stageRow(s: Stage): string {
    const n = s.stage.split(".")[1];
    return `<div class="stage">
      <label for="cut-${n}"><span class="st">${layerName(s.stage)}</span> <span class="kind">${s.kind}, ${s.channels} ch</span>
        <span class="share num">${(s.flop_frac * 100).toFixed(1)}% of compute</span></label>
      <input id="cut-${n}" type="range" min="0" max="${MAX_MASK * 100}" step="10" value="0" data-stage="${s.stage}"/>
      <output class="num" for="cut-${n}" data-out="${s.stage}">0%</output>
    </div>`;
  }

  private render() {
    const backbone = this.stages.filter((s) => Number(s.stage.split(".")[1]) <= 9);
    const neck = this.stages.filter((s) => Number(s.stage.split(".")[1]) > 9);
    this.root.innerHTML = `
      <div class="presets" role="group" aria-labelledby="ypresets-label"><span class="sub" id="ypresets-label">Presets</span>
        <div class="preset-row">${PRESETS.filter((p) => p.id !== "native").map((p) => `<button type="button" class="preset" data-preset="${p.id}" aria-pressed="false" aria-describedby="ypreset-hint">${p.name}</button>`).join("")}</div>
        <p class="preset-hint" id="ypreset-hint"></p></div>

      <section class="step" aria-labelledby="prec-title">
        <div class="step-head"><h3 id="prec-title">Precision</h3></div>
        <div class="step-body">
          <label for="y-bits" class="field"><span>Bits per weight</span><output class="num" for="y-bits" id="y-bits-out"></output></label>
          <input id="y-bits" type="range" min="1" max="${BITS.length}" step="1" data-f="bits"/>
          <fieldset class="gran" id="y-gran"><legend class="sub">Who shares a scale</legend><div class="seg">${(["tensor", "channel", "group"] as const).map(
            (k) => `<label><input type="radio" name="y-gran" value="${k}" data-f="gran"/><span>${k === "tensor" ? "Per tensor" : k === "channel" ? "Per channel" : "Per group"}</span></label>`,
          ).join("")}</div></fieldset>
          <label class="field" for="y-group" id="y-group-field"><span>Weights per group</span><select id="y-group" data-f="group">${GROUP_SIZES.map((n) => `<option value="${n}">${n}</option>`).join("")}</select></label>
          <fieldset class="levels" id="y-levels"><legend class="sub">Grid</legend><div class="seg two">${(["int", "fp"] as const).map(
            (k) => `<label><input type="radio" name="y-levels" value="${k}" data-f="levels"/><span>${k === "int" ? "Integer" : "Float"}</span></label>`,
          ).join("")}</div></fieldset>
          <p class="note" id="y-prec-note"></p>
        </div>
      </section>

      <section class="step" aria-labelledby="struct-title">
        <div class="step-head"><h3 id="struct-title">Structure</h3><button type="button" class="btn secondary" data-clear>Clear the cuts</button></div>
        <div class="step-body">
          <p class="note">Each slider masks a share of one stage's channels, weakest first (the smallest weights). Masking zeroes them to preview a structural cut. Your copy of the file doesn't shrink, so size and compute count what removing those channels for real would leave.</p>
          <fieldset class="stages"><legend class="sub">Backbone, L0 to L9</legend>${backbone.map((s) => this.stageRow(s)).join("")}</fieldset>
          <fieldset class="stages"><legend class="sub">Neck, L12 to L21</legend>${neck.map((s) => this.stageRow(s)).join("")}</fieldset>
          <p class="note">The head, L22, is never cut: it holds the class scores and the box decoder. L10, L11, L13, L14, L17 and L20 have no weights (they resize and join feature maps).</p>
        </div>
      </section>`;
    this.sync();
  }

  /** Put every control in the state the recipe describes. */
  private sync() {
    const r = this.recipe, q = <T extends Element>(s: string) => this.root.querySelector<T>(s)!;
    const bits = q<HTMLInputElement>("#y-bits");
    bits.value = String(BITS.indexOf(r.bits) + 1);
    bits.setAttribute("aria-valuetext", bitsSpoken(r.bits));
    q("#y-bits-out").textContent = bitsText(r.bits);
    for (const el of this.root.querySelectorAll<HTMLInputElement>('input[name="y-gran"]')) el.checked = el.value === r.granularity;
    for (const el of this.root.querySelectorAll<HTMLInputElement>('input[name="y-levels"]')) el.checked = el.value === r.levels;
    q<HTMLSelectElement>("#y-group").value = String(r.group);
    // Scales and grids mean nothing at 16 bits, and the grid means nothing at 1 bit.
    q<HTMLFieldSetElement>("#y-gran").disabled = r.bits >= 16;
    q<HTMLSelectElement>("#y-group").disabled = r.bits >= 16;
    q("#y-group-field").toggleAttribute("hidden", r.granularity !== "group");
    q<HTMLFieldSetElement>("#y-levels").disabled = r.bits >= 16 || r.bits === 1;
    q("#y-prec-note").textContent = precisionNote(r);
    for (const el of this.root.querySelectorAll<HTMLInputElement>("input[data-stage]")) {
      const v = r.masks[el.dataset.stage!] ?? 0, s = this.stages.find((x) => x.stage === el.dataset.stage)!;
      el.value = String(Math.round(v * 100));
      el.setAttribute("aria-valuetext", v > 0 ? `${pct(v)} of ${s.channels} channels masked` : "nothing masked");
      this.root.querySelector(`output[data-out="${el.dataset.stage}"]`)!.textContent = pct(v);
    }
    const active = this.activePreset();
    for (const b of this.root.querySelectorAll<HTMLElement>(".preset")) b.setAttribute("aria-pressed", String(b.dataset.preset === active));
    const anyCut = Object.values(r.masks).some((v) => v > 0);
    q<HTMLButtonElement>("[data-clear]").disabled = !anyCut;
    this.hint(active);
  }

  private onClick(e: Event) {
    const t = (e.target as HTMLElement).closest("button");
    if (!t || t.disabled) return;
    if (t.dataset.preset) { this.loadPreset(t.dataset.preset); return; }
    if (t.dataset.clear !== undefined) {
      this.recipe.masks = {};
      this.sync();
      this.emit();
      this.root.querySelector<HTMLElement>("#cut-0")?.focus(); // the button disables itself, so focus moves to the first stage
    }
  }

  private onInput(e: Event) {
    const el = e.target as HTMLInputElement | HTMLSelectElement;
    const r = this.recipe;
    let settle = false;
    if (el.dataset.stage) {
      const v = Number(el.value) / 100;
      if (v > 0) r.masks[el.dataset.stage] = v; else delete r.masks[el.dataset.stage];
      settle = true;
    } else if (el.dataset.f === "bits") {
      r.bits = BITS[Number(el.value) - 1];
      settle = true;
    } else if (el.dataset.f === "gran") r.granularity = el.value as DetectorRecipe["granularity"];
    else if (el.dataset.f === "group") r.group = Number(el.value);
    else if (el.dataset.f === "levels") r.levels = el.value as DetectorRecipe["levels"];
    else return;
    this.sync();
    this.emit(settle);
  }
}
