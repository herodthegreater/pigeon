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
2. For each set, recover a specialist with grouped KD. The teacher's probabilities at T=4 are summed into 8 groups (bird, 3 siblings, 4 other superclasses), and the student matches them.
3. Two specialists: the teacher pruned to 3x32 with bird-aware importance and a collapsed head, and a fresh 2x16 specialist trained only on the set.
4. Score both by bird average precision (AP) on the shared 10,000-example test set, which has every class.

## The code, line by line

`specialist_data` in `lab/experiments/specialist.py` starts with `X_all, _ = transfer_X(d)`, the 28,000 inputs, and `y_all = np.concatenate([d.train.y_clean, d.pool.y_clean])`, their true classes. Even the "unlabeled" pool has a true class in this toy world, so the filters can be exact. `bird = X_all[y_all == TARGET]` has 1,403 rows (398 from the labeled set, 1,005 from the pool). `pick = lambda m, n: X_all[m][rng.permutation(m.sum())[:n]]` draws `n` random rows matching a boolean mask, and is used for the equal-count negatives.

`sets` builds the six inputs. "birds + siblings" uses `np.isin(y_all, sib)` with `sib = d.world.siblings(TARGET)`, which is classes 1-3. "birds + noise negatives" appends `rng.standard_normal((n_b, d.world.spec.d))`. The world's inputs are standardized to mean 0 and standard deviation 1, so these noise vectors land in the same region as real data (average length 3.38 against 3.40 for real inputs). They are fake but plausible inputs, not far-away garbage.

Scoring and pruning are done once: `sc = neuron_importance(t, probe.X, probe.y, "taylor", group=Gs["binary"])` is the task-aware Taylor score (see Task-aware pruning). For each set and seed, `pruned = collapse_head(structured_prune(t, [32] * 3, sc), Gm)` rebuilds the carved specialist, and `spec_distill(pruned, t, Gm, s, X=X)` recovers it on that set alone. `spec_distill` wraps `X` in `FixedSource(X, None, K, teacher=t)`: no labels, only the teacher's outputs on those inputs. The fresh arm is `spec_distill(spec_student(16, Gm.shape[1], s), t, Gm, s, X=X)`. Both get the default step budget, so a small set is repeated for many more epochs than a large one. Both are scored with `spec_eval`, the bird AP of output 0.

`run_spec_data` in `lab/methods.py` runs the fresh 2x16 arm for one seed and 1,000 steps on four of the sets. Its "birds + random others" set draws the negatives with `rng.permutation` over all non-birds.

## What the simulations show

SIMS-TODO

## When it breaks

- **No negatives, no boundary.** Recovery trains the model only where data exists. On a bird-only set, nothing pins down the outputs for inputs far from birds, so what the model says about them is an accident of the init and the pruned weights.
- **The carved model keeps old habits.** A pruned teacher starts with a decision boundary. Positive-only data can shift that boundary in ways nothing corrects, which a fresh model with no prior boundary does not have to unlearn.
- **Fake negatives cover the wrong region.** Noise covers the input space evenly, but real confusers sit close to birds. Noise teaches "not everywhere", not "not a sibling".
- **Many epochs on a small set.** With a fixed step budget, 1,403 inputs are seen far more often than 28,000, which can overfit the teacher's outputs on those exact points.

## Try this

- Run **recovery inputs** = everything, then birds only. Compare `target_ap` and note how much the drop is for a fresh 2x16 model.
- Run birds + random others. It has the same bird count as birds only plus an equal number of negatives. Check whether it matches or beats everything with a tenth of the data.
- Run birds + noise and compare with birds + random others. Same count, but fake negatives instead of real ones.

## Going further

pigeon's recovery set (`build_finetune_set.py`) is 400 COCO images that contain birds, labelled in the model's 80-class space with bird = class 14. A detection image is not the same as a toy "bird input": every image also contains background, and a detector learns from the locations where no bird is. So pigeon's images are not purely positive in the way the toy set is. What the toy does say is that the choice of non-bird content matters, and that real negatives near the target are the most useful kind. For pigeon, adding COCO images with no birds, especially ones with other animals, is the analogue of "birds + random others". STATUS.md lists more training data (2k+ bird images) as the first lever for stronger recovery. The toy suggests checking what those images contain besides birds.
