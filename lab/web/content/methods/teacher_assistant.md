## The idea

A very small student can fail to absorb a very large teacher: the teacher's outputs come from a function the student has no way to represent, a mismatch called the *capacity gap*. Mirzadeh et al. proposed bridging it with a *teacher assistant* (TA), a mid-sized network that is first distilled from the teacher and then serves as the teacher for the small student. The TA's outputs come from a simpler function, which should be easier for the student to imitate.

## How it works

1. Train the large teacher on labels.
2. Distill a mid-sized assistant from the teacher with ordinary KD (soft targets at temperature `T`).
3. Distill the small student from the assistant, not from the teacher.
4. Optionally chain more assistants, each smaller than the last.
5. Compare against distilling the student directly from the teacher with the same budget.

```python
ta = distill(width=64, teacher=big)
student_direct = distill(width=8, teacher=big)
student_via_ta = distill(width=8, teacher=ta)
```

The method adds one full distillation run per assistant. Whether it pays depends on whether the gap, not the student's own capacity, is what limits the student.

## The code, line by line

`capacity_gap` in `lab/experiments/teacher.py` runs two experiments.

First, the gap itself. `teachers = {w: teacher(hidden=[w, w, w]) for w in [16, 32, 64, 128, 256, 512]}` builds six three-layer teachers of growing width. `teacher()` in `lab/common.py` caches each one on disk, so all experiments reuse the same trained models. For each teacher, `tr = ev(t)` records its own test metrics, then `distill([sw, sw], t, s)` trains a 2x8 and a 2x32 student from it for every seed (`distill` defaults: KD at `T = 1` on the 8k labeled plus 20k unlabeled inputs). Each row stores `teacher_acc`, so the student's accuracy can be plotted against the teacher's.

Second, the chains. `big = teachers[512]`. For every seed it records `"512 -> 8 (direct)"` from `distill([8, 8], big, s)`, then for `ta in [128, 32]` it builds `mid = distill([ta, ta], big, s)` and records `distill([8, 8], mid, s)`. Every link uses the same step budget, so the chained student gets no extra training of its own; the extra cost is the assistant's run.

`agg(gap, ["teacher_width", "student"])` and `agg(chain, ["path"])` average over seeds and add `*_std` fields with the standard deviation.

`run_ta` in `lab/methods.py` does one chain: `mid = distill(W(p["assistant"]), t, 0, steps=STEPS)` builds a 2-layer assistant from the standard 3x256 teacher, then trains the student both directly from `t` and from `mid`. `W(w)` is shorthand for two hidden layers of width `w`.

## What the simulations show

`capacity_gap`, chains to a 2x8 student (3 seeds, 3000 steps, KD at `T = 1`):

| path | accuracy |
|---|---|
| 512 -> 8 (direct) | 0.4556 ± 0.015 |
| 512 -> 128 -> 8 | 0.4579 ± 0.015 |
| 512 -> 32 -> 8 | 0.4680 ± 0.014 |

The best chain is ahead by less than one standard deviation. The gap experiment shows why there is little to fix:

| teacher width | teacher accuracy | 2x8 student | 2x32 student |
|---|---|---|---|
| 16 | 0.6788 | 0.4608 | 0.6744 |
| 32 | 0.8161 | 0.4696 | 0.7964 |
| 64 | 0.8822 | 0.4601 | 0.8268 |
| 256 | 0.9139 | 0.4556 | 0.8318 |
| 512 | 0.9228 | 0.4556 | 0.8351 |

The 2x8 student gets about the same from a 16-wide teacher as from a 512-wide one; its best result comes from the 32-wide teacher, which is also the assistant in the best chain. The 2x32 student keeps improving up to the biggest teacher. The 2x8 student is limited by its own size: `size_ladder` gives 0.4428 for a 2x8 trained on labels alone. A TA cannot add capacity.

This page's runs (1000 steps, seed 0) agree. Student 2x8, assistant 2x64: direct 0.3718, via assistant 0.3745. Student 2x4, assistant 2x32: 0.2258 and 0.2369. Student 2x16, assistant 2x128: 0.5850 and 0.5888. Single-seed gaps of this size are within noise. The assistant is weaker than the teacher (2x64 reached 0.8596 against the teacher row's 0.9150; that teacher is the copy the Run button loads, trained again with the same recipe, while the experiment's earlier training scored 0.9139), so the student learns from a worse but simpler model. The toy cannot test the regime the paper studied, convolutional students many layers shallower than their teachers, where the mismatch is much larger.

## When it breaks

- If the student is limited by its own capacity, not by the gap, no intermediate teacher helps.
- Each link loses accuracy: the assistant is worse than the teacher, and the student inherits those errors on top of its own.
- Every assistant costs a full training run, and the best assistant size has to be searched.
- A badly trained assistant (too few steps) is worse than none.
- With a short budget, any differences are smaller than seed noise; compare over several seeds before concluding.

## Try this

- Keep **student width** at 8 and try **assistant width** 16, 32, 64 and 256. Look for a size that helps consistently, and for whether the assistant's own accuracy predicts it.
- Set **student width** to 2 or 4: the gap is largest there. Check whether the via-assistant row ever pulls clearly ahead.
- Set **assistant width** equal to 256 (as wide as the teacher, but two layers) and compare the assistant row with the teacher row: that is what the student loses before it even starts.

## Going further

Teacher assistants appear in practice mainly when the gap is extreme, for example compressing a large transformer into a model many times smaller, sometimes through a chain of intermediate sizes. A simpler alternative that targets the same problem is to use a *weaker* teacher, such as an early checkpoint (see `teacher_checkpoints`), or to distill at a lower temperature so the student is not asked to copy structure it cannot represent. `lab/experiments/teacher.py:capacity_gap` is the closest analogue here; to try the idea on a detector, the chain would be YOLOv8x to YOLOv8m to YOLOv8n, each link a full detector distillation run.
