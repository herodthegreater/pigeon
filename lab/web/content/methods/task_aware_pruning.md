## The idea

Ordinary importance scores ask which neurons matter for everything the network does. *Task-aware pruning* scores neurons by how much they matter for the kept task alone, here a bird-vs-rest loss instead of 20-class cross-entropy, so neurons that only separate two kinds of vehicle are cut first. The output layer is then collapsed to a coarser label space, and the network is recovered by distilling only that grouped task. It is the toy version of the YOLOv8n cockpit's job: carving a bird detector out of an 80-class YOLO.

## How it works

1. Group the 20 classes, each grouping a `C x K` 0/1 matrix `G`. Scoring uses `binary` (bird | rest). The head uses `supers` (bird | 3 siblings | 4 other superclasses).
2. Score each hidden neuron with the Taylor criterion, `sum over examples of |a * dL/da|`, where `L = -log q[true group]` and `q = softmax(z) @ G`.
3. Keep the top `w` neurons per layer (structured pruning).
4. Collapse the 20-way output layer to `K` outputs, each the mean of its member classes' weights.
5. Recover with grouped knowledge distillation (KD): the teacher's probabilities summed into the same `K` groups.

## The code, line by line

`neuron_importance` in `lab/compress.py`, with `method="taylor"` and a `group` matrix, starts like plain Taylor. It then builds the gradient of the grouped loss with respect to the 20 logits. `yg = group[y].argmax(1)` maps each example's class to its group. `Gy = group[:, yg].T` is an `N x C` matrix with 1 for every class in the example's true group. `qy = (p * Gy).sum(1, keepdims=True)` is the probability of that group, and `dz = (p - p * Gy / qy) / len(z)` is the gradient of `-log qy`. For a non-bird example the true group is all 19 other classes, and for each of them the gradient is `p_c * (1 - 1 / qy)`, proportional to `p_c`. It never reshuffles probability inside the group, so only the bird-vs-rest boundary produces a signal. The score is then `np.abs(h * g).sum(0)` as usual.

`collapse_head` copies the net. `norm = G / G.sum(0, keepdims=True)` divides each column of `G` by its group size, so `net.W[-1] @ norm` turns the 256x20 output matrix into a 256xK one whose column `k` averages its members' columns. The bias is treated the same way. The bird is a group of one, so its *logit* is copied exactly. Its *probability* still changes, because the softmax now runs over 8 averaged logits. Averaging logits is not summing probabilities, so merged groups need recovery.

`run_task_prune` in `lab/methods.py` scores with `group=G["binary"]` ("bird") or `None` ("all classes"), prunes the 3x256 teacher to `[w] * 3`, collapses to the 8-output `supers` head, and recovers a copy with `S.spec_distill(cut.copy(), t, G["supers"], 0, steps=FT, lr=FT_LR)` (grouped KD at T=1). The `.copy()` matters: `spec_distill` trains in place, so without it the "no recovery" row would show the recovered network.

## What the simulations show

From `specialist_prune` (bird AP; recovered arms are means of 2 seeds with the same 3,000 steps as the fresh specialist; teacher AP 0.9885):

| keep width | all-class, none | bird, none | all-class, 20-way KD | all-class, collapsed + grouped KD | bird, collapsed + grouped KD | fresh 3xW specialist |
|---|---|---|---|---|---|---|
| 128 | 0.3660 | 0.7581 | 0.9901 | 0.9900 | 0.9864 | 0.9889 |
| 64 | 0.1487 | 0.5333 | 0.9834 | 0.9879 | 0.9862 | 0.9873 |
| 32 | 0.0823 | 0.4997 | 0.9611 | 0.9755 | 0.9679 | 0.9667 |
| 16 | 0.0963 | 0.5097 | 0.8464 | 0.8950 | 0.8916 | 0.9079 |
| 8 | 0.0980 | 0.5308 | 0.7040 | 0.6406 | 0.7289 | 0.7485 |
| 4 | 0.0807 | 0.4403 | 0.5308 | 0.4885 | 0.4850 | 0.3796 |

Before recovery, task-aware scoring does what it promises: the bird-scored cut keeps 0.44-0.76 bird AP at every width, against 0.08-0.37 for the all-class cut. The criteria pick different neurons. In the second hidden layer the kept sets overlap by 55% at width 128, 33% at 64, 16% at 32 and 0% at 4. After full recovery the advantage mostly disappears. From width 16 up the all-class carve recovers as well or slightly better, the bird-scored carve wins clearly only at width 8 (0.7289 against 0.6406), and at 8 and 16 a fresh specialist beats both. Collapsing the head beats 20-way KD at widths 16-64 (0.8950 against 0.8464 at 16), but not below.

Page runs (500 recovery steps, one seed; "no recovery" measured after the head collapse): at width 16, bird importance goes from 0.4964 to 0.7735 and all-class from 0.0628 to 0.6328. At width 64, bird goes from 0.5490 to 0.9715 and all-class from 0.1221 to 0.9722.

The toy's answer: task-aware importance matters when recovery is short or impossible. With enough recovery, *how many* neurons you keep matters more than *which*.

## When it breaks

- **Scored for a different question.** Neurons were scored for 2 outputs but the head keeps 8, so the sibling outputs start with little support.
- **Rare target, noisy scores.** Only 101 of the 2,000 probe examples are birds.
- **Collapsed heads start uncalibrated.** Averaged logits are only an initialization for merged groups.
- **Fresh specialists can win.** Check that baseline.

## Try this

- Set **keep width** 64 and compare **importance for** bird and all classes, first in the "no recovery" rows, then after recovery.
- Repeat at **keep width** 16, a 16x cut per layer.
- Step **keep width** from 128 down to 4 with **importance for** bird and find where recovery stops reaching the teacher's 0.9885.

## Going further

On a detector the recipe means scoring channels with a loss that counts only the target class, for example a Taylor importance on detection loss restricted to class 14, in place of the cockpit's `GroupMagnitudeImportance`, which like `l1` ignores the task. The cockpit already measures task sensitivity directly: its sensitivity scan cuts 25% of each stage, records the drop in bird mAP@50, and ranks stages by FLOPs per unit of sensitivity. Head collapse is still open: the cockpit's README notes that specializing to pigeons specifically, rather than all birds, would need "head collapse to one class".
