import { describe, expect, it } from "vitest";
import { methods, method } from "../../lib/lab/data";
import { section, whenItBreaks } from "./breaks";

const page = (body: string) => `# Title\n\n## The idea\n\nWords.\n\n## When it breaks\n\n${body}\n\n## Papers\n\n- One`;

describe("whenItBreaks", () => {
  it("quotes the first point's first two sentences, verbatim", () => {
    const n = whenItBreaks(page("- First sentence here. Second one. Third one.\n- Another point."));
    expect(n).toEqual({ text: "First sentence here. Second one.", points: 2 });
  });

  it("keeps only the first sentence when the second would run long", () => {
    const long = "B".repeat(300);
    expect(whenItBreaks(page(`- Short start. ${long}.`))!.text).toBe("Short start.");
  });

  it("counts a bold title as a title, not a sentence", () => {
    const n = whenItBreaks(page("- **Outliers.** One weight sets the scale. So the rest round to zero. Then more."));
    expect(n!.text).toBe("**Outliers.** One weight sets the scale. So the rest round to zero.");
  });

  it("never splits inside code or on a decimal point", () => {
    const n = whenItBreaks(page("- The step is `w. Q` at 0.05 here. Next sentence. Third."));
    expect(n!.text).toBe("The step is `w. Q` at 0.05 here. Next sentence.");
  });

  it("joins an indented continuation line into its point", () => {
    expect(whenItBreaks(page("- A point that\n  wraps. Second.\n- Two"))!.text).toBe("A point that wraps. Second.");
  });

  it("reads a prose section as one point", () => {
    expect(whenItBreaks(page("It breaks on outliers. Always."))).toEqual({ text: "It breaks on outliers. Always.", points: 1 });
  });

  it("returns null when the write-up has no such section", () => {
    expect(whenItBreaks("# T\n\n## The idea\n\nWords.")).toBeNull();
    expect(section("## When it breaks\n\n## Next", "When it breaks")).toEqual([""]);
    expect(whenItBreaks("## When it breaks\n\n## Next")).toBeNull();
  });

  it("finds a note for every method write-up that has the section, and every note is in the write-up", () => {
    for (const s of methods()) {
      const m = method(s.id);
      const n = whenItBreaks(m.page);
      if (!m.page.includes("\n## When it breaks\n")) {
        expect(n).toBeNull();
        continue;
      }
      expect(n, s.id).not.toBeNull();
      expect(m.page.replace(/\n\s+/g, " "), s.id).toContain(n!.text);
      expect((n!.text.match(/`/g) ?? []).length % 2, s.id).toBe(0);
      expect((n!.text.match(/\*\*/g) ?? []).length % 2, s.id).toBe(0);
    }
  });
});
