import { describe, expect, it } from "vitest";
import { experiment, experiments } from "../../lib/lab/data";
import { DENSE_FP32, denseKbits, experimentBirds, keyWords, rowKbits, type ExperimentLike } from "./birds";

const teacher = { params: 140052, acc: 0.9139, target_ap: 0.9885 };
const refs = { teacher, bayes: { acc: 0.9734 } };

describe("a row's storage", () => {
  it("is the table's kbits column when it has one", () => {
    expect(rowKbits("quantization", { params: 2132, kbits: 25.1 })).toBe(25.1);
    expect(rowKbits("anything", { params: 2132, kbits: 25.1 })).toBe(25.1);
    expect(rowKbits("quantization", { params: 2132, kbits: 0 })).toBeNull();
  });

  it("is 32 bits a parameter for a dense fp32 net, and unknown otherwise", () => {
    expect(rowKbits("size_ladder", { params: 2132 })).toBeCloseTo(68.224, 9);
    expect(rowKbits("weight_sharing", { params: 2132 })).toBeNull(); // a codebook format the table doesn't record
    expect(rowKbits("lottery_ticket", { params: 2132 })).toBeNull(); // masked
    expect(rowKbits("size_ladder", { acc: 0.8 })).toBeNull();
  });

  it("is unknown for a pruned row of an experiment that only builds some dense nets", () => {
    expect(rowKbits("prune_unstructured", { sparsity: 0.9, params: 14013 })).toBeNull();
    expect(rowKbits("prune_unstructured", { sparsity: 0.9, params: 14013, dense_width: 103 })).toBeCloseTo(448.416, 9);
  });

  it("is unknown when seeds disagree on the parameter count, which a dense net never does", () => {
    expect(rowKbits("size_ladder", { params: 2132, params_std: 3.5 })).toBeNull();
    expect(rowKbits("size_ladder", { params: 2132, params_std: 0 })).toBeCloseTo(68.224, 9);
  });

  it("matches the lab's own storage rule wherever a table records a dense fp32 row", () => {
    // compression_frontier and quantization record kbits from lab/compress.py storage_bits.
    // Dense fp32 rows: quantization's 32-bit rows, and the frontier's teacher and "2x32 fp32"-style
    // nets (its "90% fp32" rows are pruned, stored sparse, and must not match).
    const fp32 = (r: Record<string, number | string>) => r.bits === 32 || /^((\d+x)+\d+ )?fp32$/.test(String(r.config ?? ""));
    let checked = 0;
    for (const name of ["compression_frontier", "quantization"]) {
      for (const r of experiment(name).rows.filter(fp32)) {
        expect(denseKbits(r.params as number)).toBeCloseTo(r.kbits as number, 6);
        checked++;
      }
    }
    expect(checked).toBeGreaterThanOrEqual(8);
  });

  it("names only experiments the lab has", () => {
    const names = new Set(experiments().map((s) => s.name));
    for (const name of Object.keys(DENSE_FP32)) expect(names, name).toContain(name);
  });
});

describe("an experiment's birds", () => {
  const exp = (rows: ExperimentLike["rows"], r: ExperimentLike["refs"] = refs, name = "size_ladder"): ExperimentLike => ({ name, rows, refs: r });

  it("need a teacher to measure against", () => {
    expect(experimentBirds(exp([{ width: 32, params: 2132, acc: 0.85 }], { bayes: { acc: 0.97 } }))).toBeNull();
    expect(experimentBirds({ name: "size_ladder", rows: [{ width: 32, params: 2132, acc: 0.85 }] })).toBeNull();
  });

  it("need a model smaller than the teacher", () => {
    expect(experimentBirds(exp([{ width: 256, params: 140052, acc: 0.92 }]))).toBeNull();
    expect(experimentBirds(exp([{ width: 32, params: 2132, acc: 0.85 }], refs, "weight_sharing"))).toBeNull();
  });

  it("measure every smaller model against the teacher's storage and score", () => {
    const b = experimentBirds(exp([{ width: 256, params: 140052, acc: 0.92 }, { width: 32, params: 2132, acc: 0.85 }]))!;
    expect(b.reference).toEqual({ params: 140052, kbits: denseKbits(140052), score: 0.9139 });
    expect(b.models).toBe(1);
    expect(b.source).toBe("dense");
    const [{ bird, count }] = b.kinds;
    expect(count).toBe(1);
    expect(bird.ratio).toBeCloseTo(140052 / 2132, 9);
    expect(bird.kept).toBeCloseTo(0.85 / 0.9139, 9);
    expect(bird.label).toBe("Hidden width: 32");
  });

  it("count a row repeating another's size and score once, and draw each kind of bird once, for its best model", () => {
    const b = experimentBirds(
      exp([
        { width: 32, method: "KD", params: 2132, acc: 0.89 },
        { width: 32, method: "KD again", params: 2132, acc: 0.89 }, // same model, same score
        { width: 64, method: "KD", params: 6292, acc: 0.9 }, // healthy hummingbird, bigger
        { width: 16, method: "KD", params: 820, acc: 0.88 }, // healthy hummingbird, smallest
        { width: 8, method: "KD", params: 356, acc: 0.7 }, // starving
        { width: 8, method: "labels", params: 356, acc: 0.75 }, // starving, keeps more
      ]),
    )!;
    expect(b.models).toBe(5);
    expect(b.repeats).toBe(1);
    expect(b.kinds.map((k) => [k.bird.condition, k.count, k.bird.label])).toEqual([
      ["healthy", 3, "Hidden width: 16; method: KD"],
      ["emaciated", 2, "Hidden width: 8; method: labels"],
    ]);
  });

  it("score the one-class specialists by target-class AP", () => {
    const b = experimentBirds(exp([{ width: 16, method: "binary head", params: 400, target_ap: 0.95 }], refs, "specialist_size"))!;
    expect(b.metric).toBe("target_ap");
    expect(b.kinds[0].bird.kept).toBeCloseTo(0.95 / 0.9885, 9);
  });

  it("leave out, and count, the smaller models whose storage the table doesn't record", () => {
    const b = experimentBirds(
      exp(
        [
          { sparsity: 0.9, method: "magnitude", params: 14013, acc: 0.91 },
          { sparsity: 0.9, method: "dense small net", dense_width: 103, params: 14013, acc: 0.91 },
        ],
        refs,
        "prune_unstructured",
      ),
    )!;
    expect(b.models).toBe(1);
    expect(b.skipped).toBe(1);
    expect(keyWords(b).skipped).toMatch(/^1 other model has no bird: a magnitude-pruned net also stores where its remaining weights sit/);
  });

  it("on the lab's pages: only where the table can size them, never bigger than the teacher", () => {
    const withBirds: string[] = [];
    for (const s of experiments()) {
      const e = experiment(s.name);
      const b = experimentBirds(e);
      if (!b) continue;
      withBirds.push(s.name);
      expect(e.refs?.teacher?.params, s.name).toBe(b.reference.params);
      expect(b.kinds.length).toBeGreaterThan(0);
      for (const { bird } of b.kinds) expect(bird.ratio, `${s.name}: ${bird.label}`).toBeGreaterThan(1);
      expect(b.kinds.reduce((n, k) => n + k.count, 0)).toBe(b.models);
    }
    // Storage these tables don't record (codebooks, masks), or no teacher to measure against.
    for (const name of ["weight_sharing", "lottery_ticket", "capacity_gap", "born_again", "train_big_then_compress"]) {
      expect(withBirds, name).not.toContain(name);
    }
    for (const name of ["compression_frontier", "quantization", "prune_structured", "low_rank", "size_ladder"]) {
      expect(withBirds, name).toContain(name);
    }
    expect(experimentBirds(experiment("compression_frontier"))!.source).toBe("kbits");
  });
});

describe("the key's words", () => {
  const b = experimentBirds({
    name: "size_ladder",
    rows: [
      { width: 128, params: 20756, acc: 0.92 },
      { width: 32, params: 2132, acc: 0.85 },
    ],
    refs,
  })!;
  const w = keyWords(b);

  it("say how many models and kinds of bird there are, and which model a kind is drawn for", () => {
    expect(w.intro).toBe("The table's 2 smaller models hatch 2 kinds of bird.");
    const many = experimentBirds({
      name: "size_ladder",
      rows: [
        { width: 32, method: "KD", params: 2132, acc: 0.89 },
        { width: 32, method: "KD again", params: 2132, acc: 0.89 },
        { width: 16, method: "KD", params: 820, acc: 0.88 },
      ],
      refs,
    })!;
    expect(keyWords(many).intro).toBe(
      "The table's 2 smaller models hatch 1 kind of bird. Each kind is drawn once, for one model: the smallest if the bird is " +
        "healthy or bones, the one that keeps the most if it's starving. Rows with the same size and score count as one model.",
    );
    expect(keyWords(many).entries[0].count).toBe("Best of 2 models");
  });

  it("say 'scores' when a model beats the teacher and 'keeps' when it doesn't", () => {
    expect(w.entries.map((e) => e.nums)).toEqual([
      "6.7× smaller, scores 100.7% of the teacher's accuracy",
      "65.6× smaller, keeps 93.0% of the teacher's accuracy",
    ]);
    expect(w.entries.map((e) => e.count)).toEqual(["The only model of its kind", "The only model of its kind"]);
  });

  it("name the reference and where storage comes from", () => {
    expect(w.reference).toBe(
      "Measured against this experiment's teacher: 140,052 parameters, 4481.7 kbit, accuracy 0.914. " +
        "Storage is 32 bits a parameter, the lab's rule for a dense fp32 net, which every model drawn here is.",
    );
    expect(w.skipped).toBe("");
  });

  it("use no em dash", () => {
    for (const s of [w.intro, w.reference, ...w.entries.flatMap((e) => [e.nums, e.count, e.label])]) expect(s).not.toMatch(/—| -- /);
  });
});
