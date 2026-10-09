import { describe, expect, it } from "vitest";
import { method, methods } from "../../lib/lab/data";
import { codeCounts, guideLede, type CodeCounts } from "./counts";

describe("the Field Guide's counts", () => {
  it("adds up, and every count matches the method pages", () => {
    const c = codeCounts();
    expect(c.runnable + c.cockpit + c.reference).toBe(c.total);
    expect(c.referenceWithCode + c.referenceWithoutCode).toBe(c.reference);
    const ref = methods().filter((m) => m.status === "reference");
    expect(ref.filter((m) => method(m.id).code.length === 0).length).toBe(c.referenceWithoutCode);
  });

  it("only says every running page shows its code when every one does", () => {
    const c = codeCounts();
    const { lede } = guideLede(c);
    expect(lede.includes("each shown with the source code that runs it")).toBe(c.runsWithCode === c.runnable + c.cockpit);
    expect(lede).not.toMatch(/each written up with the real source code/);
  });

  it("words the reference split honestly in every case", () => {
    const base: CodeCounts = { total: 43, runnable: 31, cockpit: 3, reference: 9, runsWithCode: 34, referenceWithCode: 7, referenceWithoutCode: 2 };
    expect(guideLede(base).lede).toBe(
      "43 compression methods. 31 run in your browser and 3 in the YOLOv8n cockpit on your own machine, each shown with " +
        "the source code that runs it. The other 9 are reference write-ups for kinds of models the lab doesn't have: " +
        "7 show the closest code that does run, and 2 have no code at all.",
    );
    expect(guideLede({ ...base, runsWithCode: 33 }).lede).toContain("33 of those 34 shown with the source code that runs them");
    expect(guideLede({ ...base, referenceWithCode: 9, referenceWithoutCode: 0 }).lede).toContain("each shows the closest code");
    expect(guideLede({ ...base, referenceWithCode: 0, referenceWithoutCode: 9 }).lede).toContain("none of them has code");
    expect(guideLede({ ...base, total: 34, reference: 0, referenceWithCode: 0, referenceWithoutCode: 0 }).lede).not.toContain("reference");
  });
});
