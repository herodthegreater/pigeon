# lab: distillation & compression simulations

The YOLOv8n cockpit asks *how much of a general model does one task need?* on a
real YOLO detector, where every experiment costs minutes of GPU time and one COCO subset.
This lab asks the same question in a **miniature world where everything is
known**, so that each way of making a model smaller, or a small model better,
can be run thousands of times on a laptop CPU, with seeds and error bars, and
so that the edge cases can be poked on purpose.

**Findings are in [`REPORT.md`](REPORT.md).** Raw tables and every figure are in
[`results/RESULTS.md`](results/RESULTS.md).

## The website

```bash
pip install numpy matplotlib fastapi uvicorn
python -m lab.web                 # http://127.0.0.1:8100
```

- **Small networks / Language models / Vision**: every method on its own page
  with a plain-language write-up, the exact source code that runs (read from
  the repo), a **Run it** panel that trains and compresses a real model (most
  runs take seconds to a minute), and the simulation figures. Methods are
  tagged *runs here*, *runs in the cockpit* (the YOLOv8n cockpit's code, needs
  torch) or *reference* (needs a model this repo doesn't have, e.g. a language
  model).
- **Build it yourself**: 10 coding lessons, from softmax and backprop to
  distillation, pruning, quantization, the SVD and GPTQ. You write the code in
  the page; it runs hidden tests against it (gradient checks, comparisons with
  the lab's implementation, accuracy targets) in a separate Python process on
  your machine. The server only listens on localhost for that reason.
- **Results**: the report and every experiment's figures and table.

### The same site on GitHub Pages (no server)

`python -m lab.web.build_static --out _site` builds a static copy in which all
the Python runs **in the visitor's browser**: Pyodide (CPython compiled to
WebAssembly, ~20 MB, self-hosted from the official release) runs the same
`lab/` files inside a Web Worker. Page data, source code and results are
pre-rendered JSON; the trained models ship as `.npz` files in
`lab_bundle.zip`; lesson checks run in the worker with a 30-second limit
(a runaway loop just restarts Python). Nothing needs a backend, so it's free to
host and safe to make public.

### The shipped models (`models/`)

The eight trained models behind the Run buttons (the 3x256 teacher at seeds
0 to 3, the plain and residual 8x128 teachers, the distilled 2x32 student and
the 4,000 inverted inputs) are committed in `lab/models/`. The static build
packs those files byte for byte and never trains, so CI can't drift: training
the same model on another machine or BLAS gives slightly different weights, and
every number a write-up quotes from its page's Run would move with them.
`common.cached` falls back to these files before training anything, so a fresh
clone and the browser use the same weights (LAB_QUICK models never come from
here). `lab/tests/test_quotes.py` re-runs a sample of the quoted runs against
them.

To change what the site ships, train or load the models and write them out,
then re-measure every quoted page run in Pyodide
(`lab/web/run_in_pyodide.cjs`) and update the write-ups:

```bash
python -m lab.web.build_static --save-models   # lab/.cache wins, then lab/models; the rest is trained
```

The experiments in `results/` were recorded with an earlier training of the
same 3x256 teacher (0.9139 test accuracy); the shipped one scores 0.9150.

`.github/workflows/pages.yml` runs the tests, builds the site and deploys it on
every push to `main` that touches `lab/`. One-time setup: in the repository's
Settings, under Pages, set the source to "GitHub Actions". To preview a build
locally:

```bash
python -m lab.web.build_static --out _site
python -m http.server -d _site 8300      # http://127.0.0.1:8300
```

Everything on the site comes from one registry, [`methods.py`](methods.py),
and the test suite is generated from the same registry, so a method can't be
on the site without being tested:

```bash
pip install pytest httpx
python -m pytest lab/tests        # ~5 min; quick-mode budgets
```

The tests check every method's metadata, source pointers and write-up, run
every runnable method at its defaults *and* at both ends of every parameter
range, check that every lesson's reference solution passes and its skeleton
fails, and hit every API route.

## The simulations

```bash
python -m lab.selftest            # finite-difference check of all backprop + losses
python -m lab.run --list          # 28 experiments in 6 groups
python -m lab.run                 # run everything (~35 min on 4 cores)
python -m lab.run loss_zoo transfer_set
python -m lab.run --group 6       # just the specialization experiments
LAB_QUICK=1 python -m lab.run     # 2-minute smoke test (numbers not meaningful)
```

Dependencies: `numpy` + `matplotlib` only. No torch, so it runs anywhere.

## The world (`world.py`)

| | |
|---|---|
| classes | 20, in 5 superclasses of 4. **Class 0 is the target** ("bird"); classes 1–3 are its siblings (the confusers) |
| inputs | 12-D. Each class is a mixture of 8 Gaussian modes, pushed through an invertible warp `x = R1 z + 0.5 sin(R2 z)` |
| ceiling | the warp is a bijection, so the **exact Bayes posterior** p(y\|x) is computable: 97.3 % accuracy is the best any model can do |
| splits | 8k labeled train · 20k unlabeled "pool" · 10k test (all experiments share the same test set) |
| ladder | linear 52 % · MLP 2x8 ≈ 40 % · 2x32 ≈ 78 % · 3x256 teacher ≈ 91 %: capacity really matters |

Superclass structure is what gives soft labels *dark knowledge*: "70 % bird,
25 % sibling, 0 % vehicle" carries information a one-hot label doesn't.

## The engine

| file | what |
|---|---|
| `nn.py` | numpy MLP with hand-written backprop: masks (sparsity), fake-quant with STE (QAT, same rounding as `harness/quant.py`), residual layers, linear layers, real neuron removal, layer deletion, linear-collapse, input gradients |
| `train.py` | batch sources (fixed / online-augmented teacher queries) × loss zoo (CE, KD: KL / reverse-KL / MSE / hard / top-k / grouped, FitNets hint, similarity-preserving), training loop (AdamW, cosine, EMA), metrics (acc, target AP, NLL, ECE, KL-to-Bayes, teacher fidelity) |
| `compress.py` | magnitude pruning, neuron importance (L1 / activation / Taylor / task-aware Taylor), structured pruning, head collapse, PTQ, k-means weight sharing, SVD low-rank, layer dropping |
| `layerwise.py` | LLM compression algorithms, one linear layer at a time on calibration data: RTN, GPTQ, AWQ, SmoothQuant (+ function-preserving outlier injection), Wanda, SparseGPT, QLoRA, ShortGPT block influence |
| `methods.py` | the registry behind the website and the tests: 43 methods, their parameters, run functions and source pointers |
| `lessons.py` | the 10 build-it-yourself lessons and the sandboxed checker |
| `web/` | FastAPI server + a dependency-free single-page front end; `web/content/methods/*.md` are the write-ups; `web/build_static.py` + `web/static/worker.js` make the in-browser (GitHub Pages) version |
| `tests/` | pytest suite generated from the registry |
| `common.py` | shared world, cached teachers (lab/.cache, then the shipped `models/`), step-budgeted `scratch` / `distill` helpers, registry |
| `models/` | the trained models the website ships, committed so CI never retrains them |
| `experiments/` | the 28 experiments |
| `report.py` | results → figures + `results/RESULTS.md` |
| `selftest.py` | gradient checks for every loss and layer type |

Students and teachers trained in different arms get the **same number of
optimizer steps**, so a bigger transfer set doesn't win just by training longer.

## Technique catalog

**Making a smaller model out of a larger one**

| technique | experiment(s) |
|---|---|
| Hinton KD (soft targets, temperature, KD/CE mix) | `size_ladder`, `temperature_alpha` |
| reverse-KL, logit MSE, hard pseudo-labels | `loss_zoo` |
| top-k-only teacher (API that exposes only top-k probs) | `loss_zoo` |
| feature distillation (FitNets hints), relational (similarity-preserving) | `loss_zoo` |
| distill on unlabeled data / few labels | `size_ladder`, `label_budget` |
| data-free distillation: random noise, teacher inversion | `transfer_set` |
| function matching (online mixup, patient training) | `function_matching`, `transfer_set` |
| teacher assistants (multi-step distillation) | `capacity_gap` |
| ensemble teachers (logit vs prob averaging) | `ensemble_teacher` |
| unstructured magnitude pruning (one-shot, gradual), KD recovery | `prune_unstructured` |
| structured / neuron pruning, 4 importance criteria, 3 recovery modes | `prune_structured` |
| lottery tickets (rewind to init, late rewind, random reinit, random mask) | `lottery_ticket` |
| post-training quantization (per-tensor vs group scales), QAT with labels vs KD | `quantization` |
| k-means weight sharing (Deep Compression) vs uniform quantization | `weight_sharing` |
| SVD low-rank factorization vs neuron pruning | `low_rank` |
| layer deletion in plain vs residual nets; init a student from teacher layers | `layer_drop` |
| combining prune + quantize + distill; order of operations | `compression_frontier` |
| task-aware pruning, head collapse to the target task | `specialist_prune` |

**Making a small model better**

| technique | experiment(s) |
|---|---|
| label smoothing, mixup, EMA, longer training | `small_tricks` |
| over-parameterize then collapse (ExpandNets / RepVGG) | `small_tricks` |
| self-distillation / born-again networks | `small_tricks`, `born_again` |
| train big then prune to size | `small_tricks`, `train_big_then_compress` |
| distill from a teacher, + unlabeled data, + online augmentation | `small_tricks`, `size_ladder` |
| specialize the output head (keep only what the task needs) | `specialist_size` |

**Teacher edge cases**: capacity gap (`capacity_gap`), early / over-trained /
data-starved teachers (`teacher_checkpoints`), label-smoothed teachers
(`teacher_smoothing`), a perfect Bayes teacher (`oracle_teacher`), noisy labels
(`label_noise`), a systematically biased teacher (`biased_teacher`), and
students that don't match their teachers (`fidelity_gap`).

**Specialist data edge case**: what if the recovery data only contains the
target class, like the cockpit's 400 all-bird fine-tune images? (`specialist_data`)

## Limits of a toy world

MLPs on 12-D synthetic data are not conv nets on images. The lab is good at
*relative* questions (does X beat Y, where does a curve bend, which edge case
breaks) and at isolating mechanisms. It says nothing about absolute numbers
on COCO. Effects that depend on spatial structure (channel-pruning dependency
groups, attention transfer on feature maps, BN-statistics-based data-free
distillation) are only represented by their closest MLP analogue, or not at all.
