## The idea

A generalist cut down to a bird specialist has to be fine-tuned back to health. pigeon does that on 400 COCO images that all contain birds. This page asks whether "only images of the target" is enough. A bird-vs-rest model must learn what is *not* a bird as much as what is, and a set of positives never shows it a negative. Recovery is knowledge distillation (KD) on whatever inputs it gets, so the input set is the only variable.

## How it works

1. Build recovery sets from the 28,000 available inputs (8,000 labeled plus 20,000 unlabeled), using the true class only to filter: everything; birds only (the 1,403 inputs of class 0); birds plus the 3 sibling classes; birds plus an equal number of random non-birds; birds plus the same number of Gaussian noise vectors; and 2 x 1,403 random inputs with no filter.
2. Recover a specialist on each set with grouped KD: the teacher's probabilities (T=1) summed into 8 groups (bird, 3 siblings, 4 other superclasses).
3. Two specialists: the teacher pruned to 3x32 with bird-aware importance and a collapsed head, and a fresh 2x16 net.
4. Score both by bird average precision (AP) on the full 10,000-example test set.

## The code, line by line

`specialist_data` in `lab/experiments/specialist.py` takes `X_all, _ = transfer_X(d)` and `y_all = np.concatenate([d.train.y_clean, d.pool.y_clean])`. `bird = X_all[y_all == TARGET]` has 1,403 rows (398 of them from the labeled set). `pick = lambda m, n: X_all[m][rng.permutation(m.sum())[:n]]` draws `n` random rows matching a mask, for the equal-count negatives.

In `sets`, "birds + siblings" uses `np.isin(y_all, sib)` with `sib = d.world.siblings(TARGET)`, classes 1-3. "birds + noise negatives" appends `rng.standard_normal((n_b, d.world.spec.d))`. Inputs are standardized to mean 0 and standard deviation 1, so the noise lands where real data lives (average length 3.38 against 3.40): fake but plausible inputs.

`sc = neuron_importance(t, probe.X, probe.y, "taylor", group=Gs["binary"])` is computed once (see Task-aware pruning). For each set and seed, `pruned = collapse_head(structured_prune(t, [32] * 3, sc), Gm)` rebuilds the carved specialist and `spec_distill(pruned, t, Gm, s, X=X)` recovers it on that set alone. `spec_distill` wraps `X` in `FixedSource(X, None, K, teacher=t)`: no labels, only the teacher's outputs. The fresh arm is `spec_distill(spec_student(16, Gm.shape[1], s), t, Gm, s, X=X)`. Both get the default step budget, so a small set is seen for many more epochs than a large one.

`run_spec_data` in `lab/methods.py` trains the fresh 2x16 arm for one seed and 1,000 steps on the chosen set. Unless the choice is "everything", it also trains the same net on everything for comparison.

## What the simulations show

From `specialist_data` (bird AP, mean of 2 seeds; the teacher gets 0.9885):

| recovery inputs | pruned teacher 3x32 (recovered) | fresh specialist 2x16 |
|---|---|---|
| everything (28,000) | 0.9679 | 0.8933 |
| birds only (1,403) | 0.7971 | 0.7213 |
| birds + siblings | 0.8856 | 0.8571 |
| birds + equal number of random others (2,806) | 0.9620 | 0.9246 |
| birds + noise negatives (2,806) | 0.9406 | 0.8835 |
| random 2,806, no bird filter | 0.9271 | 0.8733 |

Bird-only recovery is the worst set for both models. The carved model drops from 0.9679 to 0.7971, and the fresh one from 0.8933 to 0.7213 (seed spread 0.092). Negatives are what pin down where "bird" ends. The 3 sibling classes, the closest confusers, recover about half of the carved model's loss (0.8856). Birds plus an equal number of random non-birds nearly matches all 28,000 inputs for the carved model (0.9620) and beats it for the fresh one (0.9246), with a tenth of the data. A random sample of the same size (about 5% birds) does worse, so the ratio of positives matters too. Even Gaussian noise negatives get the carved model to 0.9406, because standardized noise overlaps the real inputs.

Page runs (fresh 2x16, 1,000 steps, one seed): everything 0.8406, birds only 0.665, birds + random others 0.9055, birds + noise 0.7968.

The toy's "birds only" is purer than any real image set: every input is a positive with nothing else in it.

## When it breaks

- **No negatives, no boundary.** On a bird-only set nothing pins down the outputs away from birds, so what the model says there is an accident.
- **Inherited boundaries drift too.** The carved model starts with a boundary, but bird-only data drags it: it lost as much AP (0.17) as the fresh one.
- **Fake negatives work only where they land.** Standardized noise overlaps real inputs here. Random noise images look nothing like real non-bird scenes, so don't expect that result to transfer.

## Try this

- Run **recovery inputs** = birds only and compare with the "everything" row the page adds.
- Run birds + random others: same birds plus an equal number of negatives. Does it beat everything?
- Run birds + noise and compare with birds + random others, fake negatives against real ones.

## Going further

pigeon's recovery set (`build_finetune_set.py`) is 400 COCO images containing birds, labelled in the model's 80-class space with bird = class 14. A detection image is not a toy "bird input": it also contains background, and a detector learns from the locations where no bird is, so pigeon's set is not purely positive. The toy does say that the non-bird content matters, and that a balanced mix of positives and negatives from everything else worked better than near-confusers alone. For pigeon, adding a similar number of COCO images without birds is the analogue of "birds + random others". STATUS.md lists more training data (2k+ bird images) as the first lever for stronger recovery. The toy suggests checking what those images contain besides birds.
