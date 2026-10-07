# lab: distillation & compression simulations

pigeon asks *how much of a general model does one task need?* on a real YOLO
detector, where every experiment costs minutes of GPU time and one COCO subset.
This lab asks the same question in a **miniature world where everything is
known**, so that each way of making a model smaller, or a small model better,
can be run thousands of times on a laptop CPU, with seeds and error bars, and
so that the edge cases can be poked on purpose.

**Findings are in [`REPORT.md`](REPORT.md).** Raw tables and every figure are in
[`results/RESULTS.md`](results/RESULTS.md).

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
| `common.py` | shared world, cached teachers, step-budgeted `scratch` / `distill` helpers, registry |
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
target class, like pigeon's 400 all-bird fine-tune images? (`specialist_data`)

## Limits of a toy world

MLPs on 12-D synthetic data are not conv nets on images. The lab is good at
*relative* questions (does X beat Y, where does a curve bend, which edge case
breaks) and at isolating mechanisms. It says nothing about absolute numbers
on COCO. Effects that depend on spatial structure (channel-pruning dependency
groups, attention transfer on feature maps, BN-statistics-based data-free
distillation) are only represented by their closest MLP analogue, or not at all.
