import { describe, expect, it } from "vitest";
import { birdFor, type Sighting } from "../../lib/ui/birds";
import { drawBird, uniqueIds } from "./draw";
import { CONDITION_LABEL, CONDITION_WORD, keptPhrase, ratioPhrase, ratioShort } from "./format";
import { KEY, RULES, countLine, entryHTML, newestFirst, pigeonAccuracy, ruleOf, rowHTML, safeHref, scoreParts, summarize } from "./lifelist";
import { firstHealthyLine, noBirdLine, recipeOf, runBirds, runHref } from "./runbirds";
import { KEEP_RULE, PIGEON_RULE } from "../../lib/ui/birds";
import { CONDITION_WORD as MASCOT_WORD, conditionOf, starvation } from "../../lib/ui/mascot";

describe("bird words", () => {
  it("says size and health in plain numbers", () => {
    expect(ratioPhrase(6.21, "fp32")).toBe("6.2 times smaller than fp32");
    expect(ratioPhrase(0.9, "fp32")).toBe("1.1 times bigger than fp32");
    expect(ratioShort(7.7)).toBe("7.7× smaller");
    expect(keptPhrase(0.9987)).toBe("keeps 99.9% of its accuracy");
    expect(keptPhrase(1.004, "target_ap")).toBe("scores 100.4% of its target-class AP");
  });

  it("says bigger whenever a model stores more than its reference, and the same size only when it is", () => {
    // QLoRA at rank 12: 70.27 kbit of adapters and base against a 68.22 kbit student.
    expect(ratioPhrase(68.22 / 70.27, "student 2x32, fp32")).toBe("1.03 times bigger than student 2x32, fp32");
    expect(ratioShort(68.22 / 70.27)).toBe("1.03× bigger");
    expect(ratioShort(1 / 1.0004)).toBe("under 0.1% bigger");
    expect(ratioPhrase(1.02, "fp32")).toBe("1.02 times smaller than fp32");
    expect(ratioPhrase(1, "fp32")).toBe("the same size as fp32");
    expect(ratioShort(1)).toBe("same size");
    expect(ratioShort(NaN)).toBe("unknown");
    for (const r of [0.5, 0.97, 0.999, 0.99999]) expect(ratioShort(r)).toMatch(/bigger$/);
  });

  it("labels a condition the way the Coop's bird card does, and lower-cases it in running text", () => {
    expect(CONDITION_LABEL).toEqual(MASCOT_WORD);
    expect(CONDITION_LABEL.emaciated).toBe("Starving");
    expect(CONDITION_WORD.emaciated).toBe("starving");
  });
});

describe("drawings", () => {
  it("gives every drawing its own ids, so two healthy swifts don't share a clip path", () => {
    const a = drawBird(4, 0), b = drawBird(4, 0);
    const ids = (s: string) => [...s.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]);
    expect(ids(a).length).toBeGreaterThan(0);
    expect(ids(a).filter((x) => ids(b).includes(x))).toEqual([]);
    const refs = [...a.matchAll(/url\(#([^)]+)\)/g)].map((m) => m[1]);
    expect(refs.length).toBeGreaterThan(0);
    for (const r of refs) expect(ids(a)).toContain(r);
  });

  it("rewrites references along with the ids", () => {
    expect(uniqueIds('<clipPath id="c"/><g clip-path="url(#c)"/>', "p")).toBe('<clipPath id="p-c"/><g clip-path="url(#p-c)"/>');
  });

  it("hides a decorative drawing from screen readers", () => {
    expect(drawBird(1, 0, { decorative: true })).toContain('aria-hidden="true"');
    expect(drawBird(1, 0)).toContain('role="img" aria-label="Rock Pigeon, healthy"');
  });
});

const ptq = {
  metric: "acc" as const,
  rows: [
    { label: "fp32", acc: 0.915, kbits: 4481.664, ref: true },
    { label: "4-bit", acc: 0.9135, kbits: 721.536 },
    { label: "3-bit", acc: 0.8987, kbits: 582.272 },
    { label: "1-bit", acc: 0.4256, kbits: 303.744 },
  ],
};
const info = { methodId: "ptq", methodName: "Post-training quantization", href: "/pigeon/guide/ptq/#run", settings: "bits 4", lifeListHref: "/pigeon/life-list/" };

describe("a run's birds", () => {
  it("draws the best bird large, the rest one line each, and records a sighting per model", () => {
    const r = runBirds(ptq, info);
    expect(r.sightings.map((s) => s.recipe)).toEqual(["4-bit (run with bits 4)", "3-bit (run with bits 4)", "1-bit (run with bits 4)"]);
    expect(r.sightings[1]).toMatchObject({ model: "lab:ptq", modelName: "Post-training quantization", name: "Barn Swallow", condition: "healthy", href: "/pigeon/guide/ptq/#run" });
    const h = r.render({ ok: true, firstHealthy: ["Barn Swallow"] });
    expect(h.match(/<svg/g)).toHaveLength(3); // one drawing per model row, the reference has none
    expect(h).toContain("7.6 times smaller than fp32, keeps 98.2% of its accuracy.");
    expect(h).toContain("A bird stays healthy while its model keeps at least 95%");
    expect(h).toContain('href="/pigeon/life-list/"');
  });

  it("names the reference once, and counts the saved birds in words", () => {
    const kd = {
      metric: "acc" as const,
      rows: [
        { label: "teacher 3x256 (reference)", acc: 0.9, kbits: 7000, ref: true },
        { label: "pruned", acc: 0.89, kbits: 950 },
        { label: "one-shot", acc: 0.3, kbits: 950 },
      ],
    };
    const h = runBirds(kd, info).render({ ok: true, firstHealthy: [] });
    expect(h).toContain("7.3 times smaller than teacher 3x256, keeps 98.9% of its accuracy.");
    expect(h).toContain("Measured against teacher 3x256, this run's reference.");
    expect(h).toContain("Both birds are saved");
  });

  it("labels the best bird in sentence case, like the Coop's bird card", () => {
    const h = runBirds(ptq, info).render(null);
    expect(h).toMatch(/<span class="bird-cond c-healthy">Healthy<\/span>/);
    expect(h).toMatch(/<span class="c-skeleton">skeleton<\/span>/); // a comma phrase in a mini row stays lower case
  });

  it("shows labBirds' reason, with what to change, when no model got smaller", () => {
    // Low-rank at rank 128 factors nothing: every row is the reference's size.
    const same = { metric: "acc" as const, rows: [{ label: "teacher 3x256 (reference)", acc: 0.915, kbits: 4481.66, ref: true }, { label: "rank 128, no recovery", acc: 0.915, kbits: 4481.66 }, { label: "+ KD recovery", acc: 0.914, kbits: 4481.66 }] };
    const r = runBirds(same, info);
    expect(r.sightings).toEqual([]);
    const h = r.render(null);
    expect(h).toContain("No birds from this run");
    expect(h).toContain("No model in this run is smaller than teacher 3x256 (reference), so there&#39;s no bird.");
    expect(h).toContain("Change a setting so the model stores fewer bits than the reference, then run it again.");
    expect(h).not.toContain("<svg");
    // QLoRA at rank 12: the adapters make the model bigger than the student it started from.
    const grew = { metric: "acc" as const, rows: [{ label: "student 2x32, fp32", acc: 0.88, kbits: 68.22, ref: true }, { label: "+ rank-12 adapters", acc: 0.87, kbits: 70.27 }] };
    expect(runBirds(grew, info).sightings).toEqual([]);
    expect(runBirds(grew, info).render(null)).toContain("smaller than student 2x32, fp32, so there&#39;s no bird.");
    // A method that never shrinks anything gives its own reason, with no setting to change.
    expect(noBirdLine({ rows: [], bird: false }, "Same size, so no bird.")).toBe("Same size, so no bird.");
  });

  it("says why there's no bird, and when the list can't be saved", () => {
    const none = runBirds({ rows: [{ label: "a", acc: 0.7, kbits: 68 }], bird: false, bird_reason: "Same size, so no bird." }, info);
    expect(none.sightings).toEqual([]);
    expect(none.render(null)).toContain("Same size, so no bird.");
    expect(none.render(null)).toContain("No birds from this run");
    expect(runBirds(ptq, info).render(null)).toContain(">Birds from this run<");
    expect(none.render(null)).not.toContain("<svg");
    expect(runBirds(ptq, info).render({ ok: false, saved: 0 })).toContain("so this run can't be saved to your life list");
    expect(runBirds(ptq, info).render({ ok: false, saved: 1 })).toContain("Only 1 of 3 birds were saved");
  });

  it("names a species' first healthy bird once", () => {
    expect(firstHealthyLine([])).toBe("");
    expect(firstHealthyLine(["Barn Swallow", "Common Swift", "Barn Swallow"])).toBe("First healthy Barn Swallow and Common Swift on your life list.");
    expect(recipeOf("4-bit", "")).toBe("4-bit (run with default settings)");
  });

  it("links back to the method with the run's settings in the address", () => {
    expect(runHref("/pigeon/guide/gptq/", { bits: 3, act_order: true, scheme: "per row" })).toBe("/pigeon/guide/gptq/?bits=3&act_order=true&scheme=per+row#run");
    expect(runHref("/pigeon/guide/ptq/", {})).toBe("/pigeon/guide/ptq/#run");
    expect(safeHref(runHref("/pigeon/guide/gptq/", { bits: 3 }))).toBe("/pigeon/guide/gptq/?bits=3#run");
  });
});

describe("the life list page", () => {
  const s = (over: Partial<Sighting>): Sighting => ({
    ...birdFor(4, 0.99), model: "lab:ptq", modelName: "Post-training quantization", recipe: "4-bit (bits 4)", href: "/pigeon/guide/ptq/#run", when: "2026-10-09T10:00:00Z", ...over,
  });

  it("keys the five species in size order, then the dodo", () => {
    expect(KEY.map((k) => k.id)).toEqual(["pigeon", "dove", "swift", "swallow", "hummingbird", "dodo"]);
    expect(KEY[0].size).toBe("Uncompressed, or less than 1.8 times smaller");
    expect(KEY[4].size).toBe("14 times smaller or more");
  });

  it("counts species seen healthy, not sightings", () => {
    const list = [s({}), s({ recipe: "x" }), s({ ...birdFor(8, 0.97), recipe: "y" }), s({ ...birdFor(16, 0.5), recipe: "z" })];
    const sum = summarize(list);
    expect(countLine(sum)).toBe("2 of 5 species seen healthy, from 4 sightings.");
    expect(countLine(summarize([]))).toBe("0 of 5 species seen healthy: no sightings yet.");
  });

  it("draws only birds that were seen, and gives the best sighting of each", () => {
    const sum = summarize([s({})]);
    const swift = KEY.find((k) => k.id === "swift")!, dove = KEY.find((k) => k.id === "dove")!;
    expect(entryHTML(dove, sum)).not.toContain("<svg");
    expect(entryHTML(dove, sum)).toContain("Not seen yet");
    expect(entryHTML(swift, sum).match(/<svg/g)).toHaveLength(1);
    expect(entryHTML(swift, null)).not.toContain("<svg");
    expect(entryHTML(swift, sum)).toContain("Seen healthy, 1 sighting");
    expect(entryHTML(swift, sum)).toContain("Try this recipe again");
    // Before the list is read, the key claims nothing about what was seen.
    expect(entryHTML(swift, null)).not.toMatch(/Not seen|Seen/);
  });

  it("says which rule graded each sighting, and gives the numbers that rule reads", () => {
    const lab = s({});
    expect(ruleOf(lab)).toBe("keep");
    expect(scoreParts(lab)).toEqual(["keeps 99.0%"]);
    const labRow = rowHTML(lab);
    expect(labRow).toContain("keeps 99.0%");
    expect(labRow).toContain(`<td class="rule">${RULES.keep.name}</td>`);

    // Pigeon in the Coop: graded by its accuracy-and-pointing line, so its row gives pointing too.
    const pigeon = s({ ...birdFor(1.36, 0.617), condition: "skeleton", starve: 1, model: "pigeon", modelName: "Pigeon", recipe: "30% pruned · all", rule: "pigeon", pointing: 0.696 });
    expect(ruleOf(pigeon)).toBe("pigeon");
    // Skeleton with pointing above 35%: its accuracy broke it, so it is at most 60%, never the 61.7% kept.
    expect(scoreParts(pigeon)).toEqual(["accuracy 60% or less", "pointing 69.6%"]);
    expect(scoreParts({ ...pigeon, accuracy: 0.58 } as Sighting)).toEqual(["accuracy 58.0%", "pointing 69.6%"]);
    const row = rowHTML(pigeon);
    expect(row).toContain("pointing 69.6%");
    expect(row).toContain("Pigeon&#39;s line");
    // A Pigeon sighting saved before sightings carried a rule was graded by Pigeon's line all the same.
    expect(ruleOf({ model: "pigeon" })).toBe("pigeon");
    expect(scoreParts({ ...pigeon, rule: undefined, pointing: undefined })).toEqual(["keeps 61.7% of its accuracy", "pointing not saved"]);
    // Broken by pointing: its accuracy is unknown, so the share kept is said as a share.
    expect(scoreParts({ ...pigeon, pointing: 0.3 })).toEqual(["keeps 61.7% of its accuracy", "pointing 30.0%"]);
    expect(ruleOf({ model: "yolov8n" })).toBe("keep");

    // The species entry says it too.
    const sum = summarize([pigeon]);
    const entry = entryHTML(KEY.find((k) => k.id === "pigeon")!, sum);
    expect(entry).toContain("pointing 69.6%");
    expect(entry).toContain("Graded by Pigeon&#39;s line");
    expect(entryHTML(KEY.find((k) => k.id === "swift")!, summarize([lab]))).toContain("Graded by the share-kept rule");
    expect(RULES.keep.text).toBe(KEEP_RULE);
    expect(RULES.pigeon.text).toBe(PIGEON_RULE);
  });

  it("recovers Pigeon's accuracy from its line where the line pins it down, so no row contradicts it", () => {
    const at = (accuracy: number, pointing: number) => {
      const starve = starvation(accuracy, pointing);
      return s({ ratio: 2, kept: accuracy / 0.94, starve, condition: conditionOf(starve), species: "dove", name: "Mourning Dove", model: "pigeon", modelName: "Pigeon", rule: "pigeon", pointing });
    };
    // 89.6% accuracy keeps 95.3% of 94.0%, yet Pigeon's line starves it: the row says 89.6%.
    expect(scoreParts(at(0.896, 0.8))).toEqual(["accuracy 89.6%", "pointing 80.0%"]);
    expect(scoreParts(at(0.75, 0.7))).toEqual(["accuracy 75.0%", "pointing 70.0%"]);
    expect(scoreParts(at(0.93, 0.9))).toEqual(["accuracy 90% or more", "pointing 90.0%"]);
    // Starved by pointing more than by accuracy: the accuracy isn't known, only its share.
    expect(scoreParts(at(0.88, 0.5))[0]).toBe(`keeps ${(88 / 0.94).toFixed(1)}% of its accuracy`);
    expect(pigeonAccuracy(at(0.58, 0.696))).toBe("accuracy 60% or less");
  });

  it("calls a legacy sighting that grew bigger, not the same size", () => {
    expect(rowHTML(s({ ratio: 68.22 / 70.27 }))).toContain("1.03× bigger");
  });

  it("never links outside the site or trusts stored text", () => {
    expect(safeHref("/pigeon/coop/?preset=mixed")).toBe("/pigeon/coop/?preset=mixed");
    expect(safeHref("javascript:alert(1)")).toBeNull();
    expect(safeHref("//evil.test/")).toBeNull();
    const row = rowHTML(s({ modelName: "<b>x</b>", href: "javascript:alert(1)" }));
    expect(row).toContain("&lt;b&gt;x&lt;/b&gt;");
    expect(row).not.toContain("<a ");
  });

  it("lists sightings newest first, the last saved first when one run saved several at once", () => {
    const list = newestFirst([s({ when: "2026-10-01T00:00:00Z" }), s({ when: "2026-10-09T00:00:00Z" })]);
    expect(list[0].when).toBe("2026-10-09T00:00:00Z");
    const same = "2026-10-09T10:00:00.000Z";
    const run = newestFirst([s({ recipe: "old", when: "2026-10-08T00:00:00Z" }), s({ recipe: "a", when: same }), s({ recipe: "b", when: same })]);
    expect(run.map((x) => x.recipe)).toEqual(["b", "a", "old"]);
  });
});

describe("ratio wording never claims a bird the model hasn't reached (final review)", () => {
  it("cuts shrinkage instead of rounding it up to a species threshold", () => {
    expect(ratioShort(1.7669)).toBe("1.7× smaller"); // still a rock pigeon; a dove needs 1.8
    expect(ratioShort(1.8)).toBe("1.8× smaller");
    expect(ratioShort(1.2508)).toBe("1.2× smaller"); // a dodo needs under 1.3
    expect(ratioShort(6.21)).toBe("6.2× smaller");
    expect(ratioShort(1.0004)).toBe("under 0.1% smaller");
    expect(ratioShort(1)).toBe("same size");
    expect(ratioShort(68.224 / 70.272)).toBe("1.03× bigger"); // QLoRA at rank 12 grew 3%
  });
});
