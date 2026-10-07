## The idea

A generalist model spends part of its capacity telling apart 20 classes. If you only need "bird or not", a *specialist* can drop that work by shrinking its output layer (the *head*) to a coarser set of labels. The open question is how coarse. A 2-output head (bird, rest) is cheapest. Hinton's argument for distillation suggests keeping some of the teacher's "what else could it be" knowledge: an output for each of bird's 3 sibling classes, the ones most often confused with it, and maybe one per other superclass. This page compares those heads at very small widths, where every output competes for capacity.

## How it works

1. Pick a grouping of the 20 classes into `K` outputs, written as a `20 x K` 0/1 matrix `G` with exactly one 1 per row. Output 0 is always the bird alone.
   - `binary` (K=2): bird | everything else.
   - `siblings` (K=5): bird | each of its 3 siblings | everything else.
   - `supers` (K=8): bird | each of the 3 siblings | each of the 4 other superclasses.
2. Build a student with a `K`-wide output layer.
3. Distill: compute the teacher's probabilities `p_t = softmax(z_t / T)` over 20 classes (here `T = 1`), sum them into groups `q_t = p_t @ G`, and train the student's `K` outputs to match `q_t` with the usual `T^2 * KL(q_t || p_s)` loss.
4. Score by average precision (AP) of output 0 on the test set, the same single-class ranking measure pigeon uses for birds.

## The code, line by line

`groups` in `lab/experiments/specialist.py` builds the three matrices. `sib = world.siblings(TARGET)` is `[1, 2, 3]`, the other members of class 0's superclass, and `others` lists the 4 other superclasses. Each grouping is an assignment list, one output index per class. In `a_sup`, the bird maps to 0, a sibling `c` maps to `1 + sib.index(c)`, and any other class maps to `4 + others.index(superclass(c))`. `build` turns the list into the matrix with `Gm[np.arange(C), assign] = 1`.

`spec_distill` trains a student in that label space. If no inputs are given, `X` defaults to `transfer_X(data())`: the 8,000 labeled training inputs plus the 20,000-input unlabeled pool, 28,000 in total. Labels are not used. `FixedSource(X, None, Gm.shape[1], teacher=t)` runs the teacher once over all inputs and stores its logits. The loss is `KD(T=1, group=Gm)`.

Inside `KD._teacher_probs` (`lab/train.py`), `p = softmax(tl, T)` softens the teacher, then `p = p @ self.G` sums each group's probabilities. Grouping happens *after* the temperature. At the T=1 used here that makes no difference, but with `T > 1` softening spreads probability towards the many unlikely classes, so a large group, such as the 19-class "rest" of the binary head, would receive more mass than the teacher actually gives it. The student's side is unchanged: `ls = log_softmax(z, T)` over its `K` outputs, and the gradient `T * (ps - pt) / N`.

`spec_student(width, k, seed)` is a plain `[12, width, width, k]` MLP. `spec_eval` calls `evaluate(..., full_space=False)`, which reports only `target_ap` from output 0, since accuracy over 20 classes means nothing for a 2-output model.

`run_specialist_head` in `lab/methods.py` adds two comparison arms. "generalist" is a 20-way `2xW` student distilled normally with `distill` (KD over all 20 classes, also at T=1). "binary labels" trains the binary head with plain cross-entropy on `yb = (d.train.y != 0)`, where 0 means bird, with no teacher at all.

## What the simulations show

From `specialist_size` (bird AP, mean of 3 seeds; all KD at T=1; the teacher gets 0.9885 and the exact Bayes posterior 0.9992):

| width | generalist 20-way, KD | binary, labels only | binary, KD | siblings head, KD | supers head, KD |
|---|---|---|---|---|---|
| 2 | 0.1286 | 0.5149 | 0.5156 | 0.3428 | 0.2149 |
| 4 | 0.6475 | 0.8592 | 0.8781 | 0.5989 | 0.5998 |
| 8 | 0.7397 | 0.9243 | 0.9346 | 0.8555 | 0.7561 |
| 16 | 0.8619 | 0.9530 | 0.9689 | 0.9324 | 0.9018 |
| 32 | 0.9544 | 0.9743 | 0.9827 | 0.9779 | 0.9648 |

Specializing the head is a large win: a width-8 binary specialist (194 parameters) beats a width-16 generalist (820 parameters). But the richer heads lose to the plain binary head at every width, and the more outputs, the worse. At these sizes the hidden layers cannot afford to serve extra outputs, and the dark knowledge those outputs carry doesn't pay for the capacity they take. The gap closes as width grows: at 32 the siblings head is within 0.005 of binary. Distillation adds a little over labels for the binary head (0.9346 vs 0.9243 at width 8, 0.9689 vs 0.9530 at 16). Width 2 is unstable: the standard deviation of the binary arms is about 0.33.

Page runs at width 4, one seed: generalist 0.6148, binary with labels 0.8073, binary with KD 0.8134, siblings 0.6516, supers 0.3029. The supers number is far below the 3-seed mean of 0.5998, which shows how much one seed varies at this size.

Superclasses in this world are weak structure. Class centres are spread 1.0 around superclass centres spread 1.2, and each class's modes are spread 2.5. So "sibling" knowledge may matter less here than it does for real birds against other animals.

## When it breaks

- **Too many outputs for the width.** Every extra output needs features. At widths 2-8 that cost exceeds the benefit.
- **No negatives in the transfer set.** The head learns "rest" only from the inputs it sees (see Recovery data for a specialist).
- **Grouping after temperature.** At `T > 1`, summing softened probabilities inflates big groups, so a 19-class "rest" output gets a larger share of the target than the teacher actually gives it.
- **Losing the other classes.** A specialist cannot later be asked about anything else, and its output-0 score is calibrated only against the classes it was shown.
- **Tiny widths are seed lotteries.** Report several seeds before trusting a head choice.

## Try this

- At **width** 4, run all five **head** settings. Rank them by `target_ap`, then rerun at **width** 16 to see the ranking compress.
- Compare **head** binary and binary labels at **width** 8. Look at whether the teacher adds anything for a 2-output task.
- Run **head** generalist at **width** 32 and binary at **width** 8. Compare `target_ap` against the `params` column.

## Going further

In a detector, the classification branch of the head predicts one score per class at every location. pigeon keeps YOLOv8n's 80-class head: its fine-tune set is built "in the model's 80-class space (bird = 14)", and the evaluator reads class 14. README's limitations note that specializing to pigeons would need a labeled dataset "and head collapse to one class". On this toy, collapsing to bird-vs-rest beats keeping outputs for the confusers. A collapsed head is also what makes task-aware pruning meaningful: once the head asks only about birds, channels that served other classes have no output left to serve.
