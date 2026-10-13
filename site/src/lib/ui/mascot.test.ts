import { describe, expect, it } from "vitest";
import { CONDITION_WORD, DODO, mascotSVG, SPECIES, speciesFor, speciesNote } from "./mascot";

describe("the birds' notes", () => {
  it("calls only an untouched model full weight", () => {
    const pigeon = SPECIES[0];
    expect(speciesNote(pigeon, 1)).toMatch(/^Full weight/);
    for (const ratio of [1.05, 1.2, 1.79]) {
      expect(speciesFor(ratio).id).toBe("pigeon");
      const note = speciesNote(pigeon, ratio);
      expect(note).not.toMatch(/full[- ]precision|full weight|uncompressed/i);
    }
  });

  it("states the dove's real threshold in the rock pigeon's note", () => {
    expect(SPECIES[0].note).toContain(`${SPECIES[1].minRatio} times smaller`);
  });

  it("never calls a dodo's model uncompressed: it is drawn up to 1.3x smaller", () => {
    expect(DODO.note).not.toMatch(/uncompressed|full[- ]precision/i);
    expect(mascotSVG({ ratio: 1.25, accuracy: 0.5, pointing: 0.2 }).dodo).toBe(true);
    expect(mascotSVG({ ratio: 1.35, accuracy: 0.5, pointing: 0.2 }).dodo).toBe(false);
  });
});

describe("mascotSVG's two call shapes", () => {
  it("takes accuracy and pointing (Pigeon) or a starvation (any other model)", () => {
    const a = mascotSVG({ ratio: 1, accuracy: 0.95, pointing: 0.98 });
    expect([a.species.id, a.condition, a.starve, a.dodo]).toEqual(["pigeon", "healthy", 0, false]);
    const b = mascotSVG({ ratio: 4, starve: 0.5 });
    expect([b.species.id, b.condition, b.starve]).toEqual(["swift", "emaciated", 0.5]);
    const c = mascotSVG({ ratio: 8, starve: 3 });
    expect([c.species.id, c.condition, c.starve]).toEqual(["swallow", "skeleton", 1]);
  });

  it("names the condition with the words the page uses", () => {
    expect(Object.values(CONDITION_WORD)).toEqual(["Healthy", "Starving", "Skeleton"]);
    expect(mascotSVG({ ratio: 4, starve: 0.5 }).svg).toContain('aria-label="Common Swift, starving"');
    expect(mascotSVG({ ratio: 1, starve: 1 }).svg).toContain('aria-label="Dodo, skeleton"');
  });
});
