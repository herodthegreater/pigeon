# What the simulations say

28 experiments on one synthetic task. In 26 of them, every arm that trains or
recovers a model is averaged over 2 or 3 seeds; rows that only score a fixed
model (a teacher or an ensemble as it is, or a teacher or the distilled
student cut or rounded with no recovery) come from one run. `weight_sharing`
and `compression_frontier` ran a single seed throughout.
Every toy number here comes from `lab/results/*.json`, apart from the shipped
teacher's own score in the setup paragraph. The detector numbers in
the last section are quoted from the YOLOv8n cockpit's `cockpit/STATUS.md` and
`cockpit/README.md`, where marked. The figures and full tables are in
[`results/RESULTS.md`](results/RESULTS.md) and on the website's Results page.

**The setup in one paragraph.** A 20-class problem in 12 dimensions (5
superclasses of 4; class 0 plays "bird"), built so that the best possible
classifier is known exactly: the Bayes-optimal accuracy is **0.973**. The
"big model" is a 3×256 MLP (140k parameters) at **0.914** accuracy and 0.989
bird AP. (That is the teacher these experiments trained. The website's Run
buttons load the copy committed in `lab/models`, the same recipe trained
again, which scores 0.915 and 0.989.) Students are mostly 2×16 (820
parameters) or 2×32 (2.1k). Unless an experiment is about temperature,
distillation uses $T = 1$. Pruned or quantized
models are recovered with the same optimizer and learning rate as models
trained from scratch, so comparisons differ only in where the weights start.

**How far to trust it.** This is a toy: MLPs on 12-D synthetic data, not conv
nets on images or transformers on text. It is good at *relative* questions and
at isolating one mechanism at a time; it says nothing about absolute numbers
on COCO. One more caveat: "recovery" after pruning or quantization is extra
distillation on 28k inputs, so a few recovered models land a hair above the
teacher (e.g. 0.919 vs 0.914). Read those as "fully recovered", not
"improved by compression". Where a result contradicts a well-known paper, that is more likely a
fact about this regime (tiny students, plenty of clean labels, an
over-parameterized teacher) than a refutation.

---

## The short version

1. **Distillation pays for itself through new inputs, not through soft labels
   on inputs the teacher was trained on.** The teacher memorized its training
   set (loss 0.0006, essentially one-hot outputs there), so distilling on the
   8k labeled inputs gives exactly what the labels give (2×32: 0.818 vs
   0.817). Adding 20k *unlabeled* inputs is what helps (0.832), and it helps a
   lot when labels are scarce: with 250 labels, from 0.476 to 0.701.
2. **Higher temperature hurt every small student here.** 2×16 student: $T = 1$
   0.663, $T = 4$ 0.640, $T = 16$ 0.582. The textbook $T = 4$ is a setting to tune, not
   a constant.
3. **A teacher's soft labels are worth at most a few points.** Distilling from
   the *exact* Bayes posterior beats labels by 2.3 points at 2×32 and 3.4 at
   2×128, and most of that is also captured by plain Bayes-argmax labels on the
   same inputs. Students are limited by their own size first.
4. **Pruning without recovery is a cliff; with recovery it's a non-event**,
   up to a point. One-shot 90% magnitude pruning drops the teacher to 0.286;
   1,500 steps of distillation bring it to 0.912 (teacher 0.914). At 98% it
   only recovers to 0.63 while a dense net with the same weight count reaches
   0.85.
5. **A pruned teacher is a head start, not a better destination.** At equal
   training budget a pruned-and-recovered 3×64 matches a 3×64 trained fresh
   (0.903–0.908 vs 0.904). With a short budget the pruned start is ahead
   (0.874–0.893 depending on the criterion, vs 0.869). At very aggressive cuts (3×16) even the head start
   disappears.
6. **Which neurons you cut barely matters once you retrain.** Random neuron
   selection recovers about as well as L1, activation or Taylor importance. It
   matters a lot *before* recovery and for one specific case: keeping one
   class (see 9).
7. **Quantization is nearly free for the big model and expensive for the small
   one.** The teacher survives 1-bit weights after quantization-aware training
   (0.919); the 2×32 student at 2 bits reaches only 0.49 even with QAT. Small
   models have no redundancy to quantize away.
8. **Per bit stored, a small dense student wins.** A KD-trained 2×64 at 4 bits
   is 35 kbit and scores 0.901. The best sparse or pruned versions of the
   teacher at similar accuracy cost 58–360 kbit.
9. **For one class, specialize the head.** A bird-vs-rest network with 82
   parameters reaches 0.878 bird AP; the 20-way network needs 820 parameters
   for the same (0.862). Keeping extra "what else could it be" outputs costs a
   tiny specialist more capacity than the dark knowledge gives back.
10. **Recovery data needs negatives.** Recovering a bird-specialist pruned
    from the teacher on *bird inputs only* gives 0.797 bird AP; the same birds
    plus an equal number of random non-birds give 0.962, and even pure noise
    as negatives gives 0.941. The YOLOv8n cockpit's recovery set is 400
    images that all contain birds; see the last section.

---

## 1 · Distillation fundamentals

**Student size** (`size_ladder`). KD on the unlabeled pool helps at every
width, by 0.5–1.5 points of accuracy; KD on the labeled inputs alone is
indistinguishable from labels at every width.

| width | labels | KD, labeled inputs | KD, + 20k unlabeled |
|---|---|---|---|
| 2×8 | 0.443 | 0.443 | 0.456 |
| 2×16 | 0.664 | 0.663 | 0.671 |
| 2×32 | 0.817 | 0.818 | 0.832 |
| 2×64 | 0.882 | 0.882 | 0.896 |
| 2×128 | 0.904 | 0.904 | 0.916 |

**Temperature and KD weight** (`temperature_alpha`, 2×16, labeled inputs
only). Accuracy falls monotonically with $T$ at every mixing weight; at $T = 1$ KD
equals labels. A capacity-starved student spends its few parameters fitting
the teacher's softened tail.

**Which form of teacher signal** (`loss_zoo`, 2×16, labeled + unlabeled
inputs). Hard pseudo-labels (0.671), KD at $T = 1$ (0.671), and a teacher that
reveals only its top-1 or top-3 probabilities (0.673, 0.671) are
interchangeable. KD at $T = 4$ (0.640) and reverse KL (0.629) are worse; MSE on
logits is far worse (0.509) because it asks a tiny model to reproduce a
memorizing teacher's huge logit gaps. Adding a FitNets hint or
similarity-preserving loss to $T = 1$ KD changes nothing (0.672, 0.667); a hint
*instead* of KD is worse (0.622).

**Few labels** (`label_budget`, 2×32). The big teacher plus the unlabeled pool
is worth a lot when labels are scarce and nothing when they aren't:

| labels | labels only | KD + pool, big teacher | KD + pool, teacher trained on the same labels |
|---|---|---|---|
| 250 | 0.476 | 0.701 | 0.523 |
| 1000 | 0.667 | 0.752 | 0.706 |
| 8000 | 0.817 | 0.818 | 0.818 |

The teacher has to know something the student's labels don't: a teacher
trained on the same 250 labels only adds 5 points.

**Label noise** (`label_noise`, 2×32). Distillation beats the noisy labels at
every noise level, and a teacher trained *briefly* (10 epochs) is a much
better teacher under noise because it hasn't memorized the wrong labels yet:
at 60% noise, labels 0.649, KD from the fully trained teacher 0.673, KD from
the short-trained one 0.735.

## 2 · Teacher edge cases

- **Bigger teachers never hurt** (`capacity_gap`). A 2×32 student gains up to a
  64-wide teacher (0.827) and is flat after (0.835 from a 512-wide one). A
  2×8 student is flat across all teachers (0.455–0.470): it is limited by its
  own width. **Teacher assistants** ($512 \to 128 \to 8$, $512 \to 32 \to 8$) are within
  noise of distilling directly. A *weak* teacher hurts: pure KD from a
  16-wide teacher (0.679) gives a 2×32 student 0.674, against 0.817 from
  labels.
- **Checkpoints** (`teacher_checkpoints`). Student accuracy tracks the teacher
  until the teacher plateaus at epoch 16 (0.831) and stays there; an
  over-trained, unregularized teacher with worse calibration (KL to the Bayes
  posterior 0.485 vs 0.309) makes an equally good student (0.831). No
  "early-stopped teachers teach better" effect here.
- **Label-smoothed teachers** (`teacher_smoothing`). Smoothing makes a better
  *model* (0.914 to 0.926 at 0.4) and, at $T = 4$, a much worse *teacher* (student
  0.640 to 0.555), matching Müller et al. At $T = 1$ the damage is small until
  smoothing 0.4 (0.671 to 0.653).
- **The perfect teacher** (`oracle_teacher`). See point 3 above:
  posterior-KD beats labels by 1.4 / 2.3 / 2.7 / 3.4 points at widths
  16 / 32 / 64 / 128, and Bayes-argmax labels on the same inputs get almost all
  of that. Most of the teacher's value is *correct labels on more inputs*.
- **Self-distillation** (`born_again`). It makes small students worse every
  generation (2×32: 0.816 to 0.792 after three) and the teacher-sized model
  slightly better (0.917 to 0.923 with the unlabeled pool).
- **Students don't match their teachers** (`fidelity_gap`). Even a student
  with the *teacher's exact architecture* agrees with it on only 95–97% of
  test inputs after distillation; a 2×64 student reaches 93% after 12,000
  steps; a 2×16 student 75%.
- **A biased teacher** (`biased_teacher`). Half its "bird" training labels
  said "sibling", and it passes the bias on: bird AP 0.895 from pure KD vs 0.952
  from clean labels. Mixing clean labels back in repairs it (KD weight 0.25:
  0.947).
- **Ensembles** (`ensemble_teacher`). Eight teachers make a better ensemble
  (0.914 to 0.926) and the same student (0.832 to 0.834), whether logits or
  probabilities are averaged. The student, not the teacher, is the
  bottleneck.

## 3 · What the student sees (transfer data)

**Which inputs carry the knowledge** (`transfer_set`, 2×32 student, KD at
$T = 1$):

| transfer inputs | accuracy |
|---|---|
| labels only, 8k (reference) | 0.817 |
| real: the 8k labeled inputs | 0.818 |
| real: 8k + 20k unlabeled | 0.832 |
| real: 8k + online mixup / jitter | 0.813 / 0.813 |
| labels only, 500 (reference) | 0.587 |
| real: 500 | 0.593 |
| real: 500 + online mixup / mixup + jitter | 0.738 / 0.756 |
| fake: Gaussian noise / uniform noise (28k) | 0.680 / 0.679 |
| fake: teacher inversion, no real data | 0.675 |
| fake: inversion + 500 real | 0.698 |

With no real data at all, distilling on random noise gets a 2×32 student to
0.68, well above what 500 real labels give (0.59). Synthesizing inputs by
inverting the teacher was no better than noise here: in 12 dimensions noise
already covers the input space; in image space it wouldn't, which is why
DeepInversion needs its batch-norm prior. When real data is scarce, mixing it
up and querying the teacher on the mixtures is the single biggest lever
(500 real inputs: 0.756).

**Patient, consistent teaching** (`function_matching`, 2×32, only the 8k
labeled inputs). Querying the teacher on freshly mixed-up inputs every step
*loses* at short budgets (1,500 steps: 0.753 vs 0.776 for fixed targets), ties
at 6,000, and wins clearly at 24,000 (0.872 vs 0.838; labels alone 0.839).
Fixed targets stop improving after 6,000 steps; the online teacher keeps
giving new information. This is Beyer et al.'s result, reproduced.

## 4 · Shrinking a trained model

**Unstructured magnitude pruning** (`prune_unstructured`).

| sparsity | no recovery | + labels | + KD | gradual (5 steps) + KD | dense net, same #weights |
|---|---|---|---|---|---|
| 50% | 0.896 | 0.915 | 0.919 | 0.917 | 0.920 |
| 90% | 0.286 | 0.902 | 0.912 | 0.904 | 0.911 |
| 95% | 0.121 | 0.874 | 0.889 | 0.858 | 0.898 |
| 98% | 0.060 | 0.621 | 0.629 | 0.618 | 0.853 |
| 99% | 0.054 | 0.080 | 0.088 | 0.401 | 0.779 |

Gradual pruning only helps at the extreme (99%). Past about 95%, a *dense* network
with the same number of nonzero weights is better, and it needs no sparse
kernels to be fast.

**Lottery tickets** (`lottery_ticket`, 2×64, five rounds of 50%). Rewinding
the surviving weights to their initial (or early) values and retraining beats
fine-tuning the pruned weights, re-initializing them, or using a random mask,
and the gap grows with sparsity: at 96.9% sparsity, rewind 0.612, fine-tune
0.487, same mask with fresh init 0.480, random mask 0.322.

**Structured pruning** (`prune_structured`, teacher $3 \times 256 \to 3 \times W$; "short" =
750 steps of recovery, "full" = 3,000, the same as training fresh).

| width | no recovery | pruned + KD, short | fresh + KD, short | pruned + KD, full | fresh + KD, full |
|---|---|---|---|---|---|
| 128 | 0.30–0.43 | 0.915–0.917 | 0.911 | 0.915–0.918 | 0.918 |
| 64 | 0.11–0.17 | 0.874–0.893 | 0.869 | 0.903–0.908 | 0.904 |
| 32 | 0.08–0.11 | 0.721–0.776 | 0.744 | 0.836–0.860 | 0.847 |
| 16 | 0.04–0.06 | 0.504–0.551 | 0.548 | 0.652–0.686 | 0.701 |

(Ranges span the four importance criteria: L1, activation, Taylor, random.)
Without recovery every criterion falls off a cliff. With recovery the
criterion matters little: random selection is within a point of the best.
L1 was best or tied at every width; Taylor (loss-gradient) importance was no
better than random. A plausible reason: the importance was measured on the
teacher's own training data, where its loss is 0.0006 and the gradients that
Taylor importance multiplies are about 200× smaller than on held-out data.
KD beat label-only recovery at every width (e.g. 3×32, full budget: 0.836–0.860
vs 0.820–0.843).

**Quantization** (`quantization`).

| bits | teacher PTQ (per tensor / group-16) | teacher QAT + KD | student PTQ (per tensor / group-16) | student QAT + KD |
|---|---|---|---|---|
| 8 | 0.914 / 0.914 | n/a | 0.828 / 0.828 | n/a |
| 4 | 0.900 / 0.913 | 0.920 | 0.723 / 0.794 | 0.836 |
| 3 | 0.777 / 0.897 | 0.919 | 0.485 / 0.640 | 0.791 |
| 2 | 0.071 / 0.610 | 0.917 | 0.069 / 0.168 | 0.488 |
| 1 | 0.401 / 0.436 | 0.919 | 0.152 / 0.168 | 0.529 |

8 bits is free. Per-group scales are the cheapest fix for low bits. QAT
recovers the over-parameterized teacher completely, even at 1 bit, and cannot
save the small student below 3 bits.

**Low-rank** (`low_rank`). Replacing each $256 \times 256$ matrix by a rank-$r$ product
and distilling: rank 32 (41.8k parameters) matches the teacher (0.920), rank
16 (24.7k) is at 0.911, and below that it collapses (rank 8: 0.735). At about
10k parameters, deleting neurons is better (3×64: 0.893, 10.5k parameters).

**Weight sharing** (`weight_sharing`). At the same bits per weight, a
per-layer k-means codebook beats uniform rounding wherever bits are scarce
(teacher at 2 bits: 0.849 vs 0.071; student at 3 bits: 0.645 vs 0.485), and
fine-tuning with re-snapping adds 1–3 more points. One edge case worth
knowing: symmetric 2-bit uniform quantization has only three levels
$\{-s, 0, +s\}$ with $s$ set by the largest weight, so most weights round to zero
and it does *worse* than 1-bit sign quantization (teacher 0.071 vs 0.401).

**Deleting layers** (`layer_drop`, 8×128 teachers). Removing even one layer
from a plain network collapses it (0.910 to 0.062), but a short distillation
restores it completely, even with 6 of 8 layers gone (0.914). A residual
network degrades gracefully without any recovery (1 layer gone: 0.867, 2:
0.804, 4: 0.532), which is why layer dropping is practical for transformers.
Initializing a 4-layer student from every other teacher layer gave no benefit
over random init (0.916 vs 0.919): this teacher's depth was surplus.

**Everything on one chart** (`compression_frontier`). By storage:

| model | kbit | accuracy |
|---|---|---|
| teacher, fp32 | 4,482 | 0.914 |
| 90%-sparse teacher, 4-bit (prune, then QAT) | 359 | 0.914 |
| structured-pruned 3×64, 4-bit | 58 | 0.900 |
| **KD student 2×64, 4-bit** | **35** | **0.901** |
| KD student 2×32, 4-bit | 13 | 0.836 |
| structured-pruned 3×32, 4-bit | 19 | 0.824 |

Sparse storage pays a bitmap per weight position, which is why the sparse
teacher stays large. **Order matters**: at 97% sparsity, prune-then-quantize
(0.858) beats quantize-then-prune (0.794).

## 5 · Making small models better

**Bag of tricks** (`small_tricks`). For a fixed 2×32 network (labels 0.817):

| trick | 2×16 | 2×32 |
|---|---|---|
| baseline | 0.664 | 0.817 |
| label smoothing 0.1 | 0.670 | 0.823 |
| mixup | 0.631 | 0.799 |
| EMA of weights | 0.648 | 0.807 |
| 4× longer training | 0.715 | 0.841 |
| over-parameterize, then collapse | 0.710 | 0.842 |
| self-distillation | 0.623 | 0.801 |
| train 2×128, prune to size, self-KD | 0.657 | 0.823 |
| KD from teacher + unlabeled pool | 0.671 | 0.832 |
| KD + pool + online mixup + EMA, 4× longer | 0.710 | 0.861 |

The two tricks that work for a capacity-starved model both change
*optimization*, not regularization: train longer, or train an
over-parameterized version and multiply the extra matrices back in (the
collapsed network is exactly the same size). Regularizers (mixup, EMA,
self-distillation) hurt at a fixed budget, because the model is underfitting,
not overfitting. (EMA used a 0.999 decay over 3,000 steps, which still leans on
fairly early weights; a decay matched to the run length would hurt less.) The best 2×32 combines the teacher, the unlabeled pool,
consistent online augmentation and a longer schedule (0.861).

**Train big, then compress** (`train_big_then_compress`). Not in this regime:
for a final 2×16, training it directly (0.686) beats pruning down from 32,
64, 128 or 256 wide (0.61–0.64, with or without self-distillation). Li et
al.'s result is about compute-limited training of large transformers; here
the small model is not compute-limited.

## 6 · Specialization (the bird-detector question)

**How small can a one-class model be?** (`specialist_size`, bird AP)

| width | 20-way, KD | bird vs rest, labels | bird vs rest, KD | bird + 3 siblings + rest, KD | bird + siblings + 4 superclasses, KD |
|---|---|---|---|---|---|
| 2×4 | 0.648 | 0.859 | 0.878 | 0.599 | 0.600 |
| 2×8 | 0.740 | 0.924 | 0.935 | 0.855 | 0.756 |
| 2×16 | 0.862 | 0.953 | 0.969 | 0.932 | 0.902 |
| 2×32 | 0.954 | 0.974 | 0.983 | 0.978 | 0.965 |

A binary head is the right shape for a tiny specialist; KD adds about a point
over labels. Extra outputs that preserve "which non-bird is it" only stop
hurting at 2×32.

**Carving the specialist out of the generalist** (`specialist_prune`, teacher
$3 \times 256 \to 3 \times W$, bird AP). Scoring neurons by their effect on a *bird-vs-rest*
loss keeps a usable bird detector even before any recovery (0.50 at 3×32,
against 0.08 for all-class importance), and the neurons it keeps differ from
the generally important ones: only 16% overlap at 3×32 and none at 3×4.
After recovery, collapsing the head to the task and distilling only the task
beats keeping the 20-way head (3×16: 0.895 vs 0.846; 3×32: 0.975 vs 0.961).
Task-aware importance mattered after recovery only at 3×8 (0.729 vs 0.641).
At these widths a fresh specialist trained the same way is about as good
(3×16: 0.908; 3×32: 0.967), except at the extreme 3×4, where the pruned start
is clearly better (0.488 vs 0.380).

**What the recovery data must contain** (`specialist_data`, bird AP):

| recovery inputs | pruned teacher 3×32 | fresh 2×16 specialist |
|---|---|---|
| everything (28k) | 0.968 | 0.893 |
| **birds only (1,403)** | **0.797** | **0.721** |
| birds + their 3 sibling classes | 0.886 | 0.857 |
| birds + an equal number of random others | 0.962 | 0.925 |
| birds + random-noise negatives | 0.941 | 0.883 |
| 2,806 random inputs, no bird filter | 0.927 | 0.873 |

Recovering on birds alone loses 17 points: the model is never shown what a
non-bird looks like, so it stops rejecting them. Adding the same number of
random non-birds recovers almost everything, and even noise is a decent
stand-in for negatives.

---

## What this means for the cockpit

The YOLOv8n cockpit's loop is: cut YOLOv8n (remove channels, round weights to
fewer bits, or both), then fine-tune on 400 COCO images that all contain birds, with the
quantizer inside the training loop (QAT) when the recipe quantizes, and
measure bird mAP@50. The toy can't promise numbers for YOLO, but it points at
concrete things to try, roughly in order of expected payoff:

1. **Put non-bird images in the recovery set.** The single biggest effect in
   section 6 (0.797 to 0.962 bird AP) comes from adding negatives to an
   all-positive recovery set. A detector sees background inside bird images,
   which is partial protection the toy doesn't have; `build_finetune_set.py`
   keeps only the bird boxes, so other objects in those photos already count
   as background. Whole images without birds are what teach it not to fire on
   cats and kites. Adding an equal number of random COCO images is cheap.
2. **Recover by distilling from the unpruned model, on more images than you
   have labels for.** KD from the original YOLOv8n's outputs needs no labels,
   so any unlabeled images (COCO train, video frames) become recovery data.
   That was worth 1.5 points with 8k labels and 22 points with 250. The
   Ultralytics trainer `recover_job.py` drives has no distillation term yet;
   that is the code to add.
3. **Cut in small steps and give each step a real recovery budget.** Here,
   short recovery made pruning look up to 15 points worse than it is. The
   cockpit's own runs (cockpit/STATUS.md) point the same way, more strongly: one 20% cut plus
   40 epochs recovered only to 0.0706, while three 7.2% cuts with 15 epochs
   after each kept 0.2874 (native 0.4284) with 31% fewer params. The toy's
   gradual unstructured pruning only helped at 99% sparsity, so on this
   question trust the detector. Each step lost more than the last
   (0.3943, 0.3377, 0.2874), and cockpit/STATUS.md names more data and epochs per step
   as the next lever. Those recovered numbers lean optimistic: `best.pt` is
   picked on the same 125 images they are scored on.
4. **Collapse the head to the task.** Keeping 80 class outputs costs capacity
   a small model doesn't have (section 6). A bird-only (or bird +
   "everything else") head, distilled from the teacher's grouped scores, is
   the detector version of the specialist head. The cockpit still trains the
   80-class head.
5. **Expect quantization to be the cheap win and structure to be the hard
   one.** 8 bits was free everywhere in the toy; low bits need group scales
   and QAT, and hurt small models far more than big ones. The detector agrees:
   with group-128 scales 8-bit costs 0.5% of YOLOv8n's bird mAP@50 and plain
   4-bit rounding collapses, while a one-shot 5% channel cut scores 0.000
   before fine-tuning (cockpit/README.md). Better schemes push the precision cliff
   back: FP4 levels with one scale per output channel keep 79% of native mAP,
   92% with stages 0-2 left at 8 bits (4.04 bits per weight), and QAT took a
   3.16-bit recipe from about 0% to 70% of native in 3 epochs
   (cockpit/README.md, measured on a stand-in bird set, so read them as
   relative). The toy's order
   holds: Regrow prunes first and then fine-tunes with the quantizer in the
   loop.
6. **Don't trust gradient-based importance computed on training images.** The
   model's loss there is near zero; measure importance on held-out images, or
   use L1, which was best or tied here. The cockpit's Export already ranks channels
   by weight magnitude, and its structure scan measures each stage's mAP drop
   on evaluation images rather than training images.

---

## Mistakes caught while running this

The first full run had four problems, found by reading the results critically
and by the agents that wrote the method pages; all are fixed and every number
above comes from the corrected code:

- `magnitude_prune` counted sparsity relative to the weights still alive, so
  repeated calls compounded: "gradual pruning to 75%" actually ended near 98%,
  and lottery-ticket rounds were far sparser than labelled.
- Pruned models were recovered with half the steps and a third of the
  learning rate of the fresh baselines, which made "train small from scratch"
  look up to 15 points better than "prune and recover". At equal budget the gap
  mostly disappears (section 4).
- The default distillation temperature ($T = 4$) was hurting every small student;
  experiments now use $T = 1$ except where temperature is the variable.
- Ensemble probability-averaging went through `log(p + 1e-8)`, which reshaped
  the targets; it now uses an exact log-mean-exp.
