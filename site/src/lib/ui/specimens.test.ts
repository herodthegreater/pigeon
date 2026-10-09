import { describe, expect, it } from "vitest";
import testset from "../../../public/testset/testset.json";
import { creditText, decodeEntities, esc } from "./specimens";
import type { TestImage } from "../engine/testset";

describe("photo credits", () => {
  it("decodes the entities the test set stores, instead of deleting their characters", () => {
    expect(decodeEntities("R&#x27;lyeh Imaging")).toBe("R'lyeh Imaging");
    expect(decodeEntities("Andy Reago &amp; Chrissy McClarren")).toBe("Andy Reago & Chrissy McClarren");
    expect(decodeEntities("InAweofGod&#39;sCreation")).toBe("InAweofGod'sCreation");
    expect(decodeEntities("Tom &lt;3 &quot;birds&quot;")).toBe('Tom <3 "birds"');
    expect(decodeEntities("AT&T &bogus; & co")).toBe("AT&T &bogus; & co");
  });

  it("leaves no entity in any credit in the test set, and escapes the decoded text for HTML", () => {
    for (const img of (testset as { images: TestImage[] }).images) {
      const text = creditText(img);
      expect(text).not.toMatch(/&(#x?[0-9a-f]+|amp|lt|gt|quot|apos);/i);
      expect(esc(text)).not.toMatch(/[<>"]/);
    }
    const rlyeh = (testset as { images: TestImage[] }).images.find((i) => i.credit.author?.startsWith("R&"))!;
    expect(creditText(rlyeh)).toBe("Photo: R'lyeh Imaging");
  });
});
