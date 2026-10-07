## The idea

Ordinary importance scores ask "which neurons matter for everything this network does?" A specialist only needs one thing. *Task-aware pruning* scores neurons by how much they matter for the kept task alone. Here that means a bird-vs-rest loss instead of 20-class cross-entropy. Neurons that only separate, say, two kinds of vehicle get low scores and are cut first. After pruning, the output layer is collapsed to a coarser label space, and the network is recovered by distilling only the grouped task from the original. This is the toy version of pigeon's goal: carve a bird detector out of an 80-class YOLO.

## How it works

1. Group the 20 classes. For scoring, `binary`: bird | rest. For the head, `supers`: bird | 3 siblings | 4 other superclasses. Each grouping is a `C x K` 0/1 matrix `G`.
2. Score each hidden neuron with the Taylor criterion, `sum over examples of |a * dL/da|`. Here `L` is cross-entropy in the *grouped* space, `-log q[true group]` with `q = softmax(z) @ G`.
3. Keep the top `w` neurons per layer (structured pruning).
4. Collapse the 20-way output layer to `K` outputs. Each new output's weights are the mean of its member classes' weights.
5. Recover with grouped knowledge distillation (KD): the teacher's softened probabilities summed into the same `K` groups.

## The code, line by line

`neuron_importance` in `lab/compress.py`, with `method="taylor"` and a `group` matrix, starts like plain Taylor. `z, hid, cache = net.forward(X, cache=True)` and `p = softmax(z)`. The task-aware branch then builds the gradient of the grouped loss with respect to the 20 logits. `yg = group[y].argmax(1)` maps each example's class to its group: 0 for bird and 1 for rest under `binary`. `Gy = group[:, yg].T` is an `N x C` matrix holding 1 for every class in that example's true group. `qy = (p * Gy).sum(1, keepdims=True)` is the probability the model gives to the true group. The gradient of `-log qy` is `dz = (p - p * Gy / qy) / len(z)`. For a non-bird example, the true group is all 19 non-bird classes. Moving probability *between* those classes leaves `qy` unchanged. The gradient reflects that: for each class `c` in the group it is `p_c * (1 - 1 / qy)`, proportional to `p_c`. It lifts the whole group together and has no part that reshuffles probability inside it. Only the bird-vs-rest boundary produces a training signal. `net.backward(cache, dz, hidden_grads=hg)` carries it back to each hidden layer, and the score is `np.abs(h * g).sum(0)`, as in Structured pruning.

`collapse_head` copies the net. `norm = G / G.sum(0, keepdims=True)` divides each column of `G` by its group size, so `net.W[-1] @ norm` replaces the 256x20 output matrix with a 256xK one whose column `k` is the average of its member classes' weight columns. The bias is treated the same way. `net.masks[-1] = None` drops any output mask and `net.sizes[-1] = G.shape[1]` records the new width. For a group of one class (the bird) the column is copied unchanged, so the bird *logit* is exact. The bird *probability* still changes, because the softmax now runs over 8 averaged logits instead of 20. Averaging logits is not the same as summing probabilities, so merged groups start out wrong and need the recovery step.

`run_task_prune` in `lab/methods.py` computes `sc` with `group=G["binary"]` (criterion "bird") or `None` ("all classes"). It prunes the 3x256 teacher to `[w] * 3`, collapses it to the 8-output `supers` head, and recovers a copy with `S.spec_distill(cut.copy(), t, G["supers"], 0, steps=FT, lr=FT_LR)`, which is KD at T=4 with the teacher's probabilities grouped. The `.copy()` matters: `spec_distill` trains its argument in place, so without it the "no recovery" row would be evaluated on the already-recovered network.

## What the simulations show

From `specialist_prune` (bird AP on the full test set; recovered arms are mean of 2 seeds and get the same 3,000 steps as the fresh specialist; teacher AP 0.9885):

| keep width | all-class, none | bird, none | all-class, 20-way KD | all-class, collapsed + grouped KD | bird, collapsed + grouped KD | fresh 3xW specialist |
|---|---|---|---|---|---|---|
| 128 | 0.3660 | 0.7581 | 0.9901 | 0.9893 | 0.9866 | 0.9891 |
| 64 | 0.1487 | 0.5333 | 0.9834 | 0.9863 | 0.9862 | 0.9866 |
| 32 | 0.0823 | 0.4997 | 0.9611 | 0.9676 | 0.9566 | 0.9649 |
| 16 | 0.0963 | 0.5097 | 0.8464 | 0.8723 | 0.8616 | 0.8931 |
| 8 | 0.0980 | 0.5308 | 0.7040 | 0.4917 | 0.7560 | 0.7391 |
| 4 | 0.0807 | 0.4403 | 0.5308 | 0.4317 | 0.4012 | 0.3903 |

Task-aware scoring does exactly what it promises *before* recovery: at every width the bird-scored cut keeps around 0.44-0.76 bird AP, against 0.08-0.37 for the all-class cut. The two criteria pick different neurons. In the second hidden layer, the kept sets overlap by 55% at width 128, 33% at 64, 16% at 32, and 0% at 4. After a full recovery, though, the advantage mostly disappears. The bird-scored carve wins clearly only at width 8. At 16 and 32 the all-class carve recovers slightly better, and at 16 a fresh specialist of the same shape beats both. Collapsing the head and distilling only the grouped task helps over 20-way KD at widths 16-64, but not at the smallest widths.

Page runs (500 recovery steps, one seed; "no recovery" here is measured after the head collapse): width 16 with bird importance goes from 0.5068 to 0.7303, and with all-class importance from 0.1015 to 0.5404. At width 64: bird 0.5095 → 0.9653, all classes 0.1538 → 0.9547. With a short recovery budget, the better starting point still shows.

So the toy's answer: task-aware importance matters most when recovery is short or impossible. With enough recovery data and steps, which neurons you kept matters less than how many.

## When it breaks

- **Scored for the wrong question.** A task-aware score can't see what the recovery loss needs. Here the head keeps 8 outputs, but neurons were scored only for 2 (bird vs rest), so the sibling outputs start with little support.
- **Rare target, noisy scores.** Birds are 1/20 of the probe set. Of the 2,000 probe examples, 101 are birds, so scores for bird-specific neurons rest on few examples.
- **Collapsed heads start wrong.** Averaged logits are a rough initialization for merged groups. Without recovery, a collapsed model's probabilities are not calibrated.
- **One-shot scores.** As with all one-shot structured pruning, scores are computed on the full net and not updated as layers shrink.
- **Fresh specialists can win.** At small widths, a randomly initialised specialist distilled with the same grouped loss may beat the carved one. Check that baseline before trusting a carve.

## Try this

- Set **keep width** to 64 and compare **importance for** bird and all classes. Look at the "no recovery" rows: task-aware scoring should keep more of the bird AP before any training.
- Repeat at **keep width** 16. See whether the task-aware advantage survives recovery when the cut is 16x per layer.
- Step **keep width** down from 128 to 4 with **importance for** bird and find where recovery stops reaching the teacher's 0.9885 AP.

## Going further

On a detector, the same recipe means scoring channels with a loss that only counts the target class. A loss-gradient (Taylor) importance on detection loss restricted to class 14 would replace pigeon's current `GroupMagnitudeImportance`, which, like `l1` here, ignores the task. pigeon's cockpit already measures task sensitivity directly: its sensitivity scan cuts 25% of each stage and records the drop in bird mAP@50, then ranks stages by FLOPs per unit of sensitivity. That is a per-stage, measured version of task-aware importance. The head-collapse half is still open: README notes that specializing to pigeons would need "head collapse to one class".
