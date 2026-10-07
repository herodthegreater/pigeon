## The idea

After you cut a generalist down to a bird specialist, it has to be fine-tuned back to health. pigeon does that on 400 COCO images that all contain birds. This page asks whether "only images of the target" is enough recovery data. A bird-vs-rest model has to learn what is *not* a bird as much as what is, and a set made only of positives never shows it a negative. The recovery here is knowledge distillation (KD): the student learns from the teacher's probabilities on whatever inputs it is given, so the content of the input set is the only thing that varies.

## How it works

1. Build candidate recovery sets from the 28,000 available inputs (8,000 labeled training plus 20,000 unlabeled pool), using the true class only to filter:
   - everything: all 28,000.
   - birds only: the 1,403 inputs whose true class is 0.
   - birds + siblings: birds plus the 3 confuser classes from the same superclass.
   - birds + an equal number of random non-birds.
   - birds + noise: birds plus the same number of random Gaussian vectors.
   - random: 2 x 1,403 inputs drawn without looking at the class.
2. For each set, recover a specialist with grouped KD. The teacher's probabilities (at T=1) are summed into 8 groups (bird, 3 siblings, 4 other superclasses), and the student matches them.
3. Two specialists: the teacher pruned to 3x32 with bird-aware importance and a collapsed head, and a fresh 2x16 specialist trained only on the set.
4. Score both by bird average precision (AP) on the shared 10,000-example test set, which has every class.

## The code, line by line

`specialist_data` in `lab/experiments/specialist.py` starts with `X_all, _ = transfer_X(d)`, the 28,000 inputs, and `y_all = np.concatenate([d.train.y_clean, d.pool.y_clean])`, their true classes. Even the "unlabeled" pool has a true class in this toy world, so the filters can be exact. `bird = X_all[y_all == TARGET]` has 1,403 rows (398 from the labeled set, 1,005 from the pool). `pick = lambda m, n: X_all[m][rng.permutation(m.sum())[:n]]` draws `n` random rows matching a boolean mask, and is used for the equal-count negatives.

`sets` builds the six inputs. "birds + siblings" uses `np.isin(y_all, sib)` with `sib = d.world.siblings(TARGET)`, which is classes 1-3. "birds + noise negatives" appends `rng.standard_normal((n_b, d.world.spec.d))`. The world's inputs are standardized to mean 0 and standard deviation 1, so these noise vectors land in the same region as real data (average length 3.38 against 3.40 for real inputs). They are fake but plausible inputs, not far-away garbage.

Scoring and pruning are done once: `sc = neuron_importance(t, probe.X, probe.y, "taylor", group=Gs["binary"])` is the task-aware Taylor score (see Task-aware pruning). For each set and seed, `pruned = collapse_head(structured_prune(t, [32] * 3, sc), Gm)` rebuilds the carved specialist, and `spec_distill(pruned, t, Gm, s, X=X)` recovers it on that set alone. `spec_distill` wraps `X` in `FixedSource(X, None, K, teacher=t)`: no labels, only the teacher's outputs on those inputs. The fresh arm is `spec_distill(spec_student(16, Gm.shape[1], s), t, Gm, s, X=X)`. Both get the default step budget, so a small set is repeated for many more epochs than a large one. Both are scored with `spec_eval`, the bird AP of output 0.

`run_spec_data` in `lab/methods.py` runs the fresh 2x16 arm for one seed and 1,000 steps on four of the sets. Its "birds + random others" set draws the negatives with `rng.permutation` over all non-birds.

## What the simulations show

From `specialist_data` (bird AP on the full test set, mean of 2 seeds, KD at T=1; the teacher gets 0.9885):

| recovery inputs | pruned teacher 3x32 (recovered) | fresh specialist 2x16 |
|---|---|---|
| everything (28,000) | 0.9679 | 0.8933 |
| birds only (1,403) | 0.7971 | 0.7213 |
| birds + siblings | 0.8856 | 0.8571 |
| birds + equal number of random others (2,806) | 0.9620 | 0.9246 |
| birds + noise negatives (2,806) | 0.9406 | 0.8835 |
| random 2,806, no bird filter | 0.9271 | 0.8733 |

Bird-only recovery is the worst set for both models: the carved model drops from 0.9679 to 0.7971, and the fresh one from 0.8933 to 0.7213 (with a large seed spread, 0.092). Negatives are what pins down where "bird" ends. Adding the 3 sibling classes, the closest confusers, recovers about half of the carved model's loss (0.8856). A balanced set of birds plus the same number of random non-birds nearly matches all 28,000 inputs for the carved model (0.9620) and beats it for the fresh one (0.9246 against 0.8933), with a tenth of the data. A random sample of the same size without the bird filter (about 5% birds) does worse than the balanced set, so the ratio of positives matters, not only the presence of negatives. Even Gaussian noise as negatives gets the carved model to 0.9406, because standardized noise falls in the same region as the real inputs.

Page runs (fresh 2x16, 1,000 steps, one seed): everything 0.8406, birds only 0.665, birds + random others 0.9055, birds + noise 0.7968.

The toy's "birds only" is purer than any real image set. Every input is a positive, with nothing else in it.

## When it breaks

- **No negatives, no boundary.** Recovery trains the model only where data exists. On a bird-only set, nothing pins down the outputs for inputs far from birds, so what the model says about them is an accident of the init and the pruned weights.
- **Inherited boundaries drift too.** A pruned teacher starts with a decision boundary, but bird-only data drags it with nothing to hold it in place. In the experiment the carved model lost as much bird AP (0.17) as the fresh one.
- **Fake negatives only work where they land.** Here standardized Gaussian noise happens to fall where real inputs live, so it works almost as well as real negatives for the carved model. In image space, random noise looks nothing like a real non-bird scene, so this result should not be expected to transfer.
- **Many epochs on a small set.** With a fixed step budget, 1,403 inputs are seen far more often than 28,000, which can overfit the teacher's outputs on those exact points.

## Try this

- Run **recovery inputs** = everything, then birds only. This page trains a fresh 2x16 model. Compare its drop with the experiment's numbers above.
- Run birds + random others. It has the same bird count as birds only plus an equal number of negatives. Check whether it matches or beats everything with a tenth of the data.
- Run birds + noise and compare with birds + random others. Same count, but fake negatives instead of real ones.

## Going further

pigeon's recovery set (`build_finetune_set.py`) is 400 COCO images that contain birds, labelled in the model's 80-class space with bird = class 14. A detection image is not the same as a toy "bird input": every image also contains background, and a detector learns from the locations where no bird is. So pigeon's images are not purely positive in the way the toy set is. What the toy does say is that the non-bird content matters. Here the best recipe was a balanced mix of positives and negatives drawn from everything else, not only near-confusers. For pigeon, adding a similar number of COCO images with no birds is the analogue of "birds + random others". STATUS.md lists more training data (2k+ bird images) as the first lever for stronger recovery. The toy suggests checking what those images contain besides birds.
