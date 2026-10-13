## The idea

A convolution layer outputs a stack of feature maps called *channels*, one per filter. Deleting a channel removes that filter and one input slice of every layer that reads it, so the model gets smaller and does less arithmetic. With residual adds and concatenations, one channel is read by several layers, and all of them must lose it together or the shapes stop matching. That set is a *dependency group*. The torch-pruning library finds the groups by tracing the network; the YOLOv8n cockpit adds one rewrite so YOLOv8's main block can be traced.

## How it works

1. Rewrite: replace each Ultralytics `C2f` block with `C2f_v2`, which computes the same thing in a form the tracer can follow.
2. Trace: mark every weight as trainable, switch to training mode, and run one example image through the model. torch-pruning follows the autograd graph to record which layer feeds which.
3. Group: for each output channel, collect everything tied to it: the conv filter, its batch-norm (BN) entries, the input slice of each consumer, and anything joined by a residual add or a concatenation.
4. Score each channel $c$ in a group by the size of its weights, an L2 (sum-of-squares) criterion $\lVert W_c \rVert_2^2$ on each layer's slice $W_c$, combined across all the group's layers. A small score means little signal flows through it.
5. Remove the lowest-scoring fraction of every group by slicing the tensors, so they physically shrink. The fraction is either one number for the whole model or one per stage.
6. Never remove output channels of the Detect head (the layer that emits boxes and class scores).

Example: YOLOv8's Bottleneck computes `x + cv2(cv1(x))`. Cutting channel 7 from `cv2`'s output forces cutting channel 7 from `x`, and so from the layer that produced `x` and every layer that reads it.

## The code, line by line

`C2f` splits one $1 \times 1$ conv's output in half with `chunk(2, 1)`; the module docstring records that the tracer can't follow it, so a generic pruner removes zero channels.

`C2f_v2.__init__` builds two convs instead, `self.cv0` and `self.cv1`, each producing `self.c = int(c2 * e)` channels. `self.cv2` maps the concatenation of $2 + n$ branches back to `c2` channels. `forward` builds `y = [self.cv0(x), self.cv1(x)]`, runs each Bottleneck on the newest entry (`y.extend(m(y[-1]) for m in self.m)`) and returns `self.cv2(torch.cat(y, 1))`. Every split is now a separate layer the tracer can see.

`transfer_weights` makes the swap exact: it reuses the old `cv2` and `m`, gives `old_weight[:half]` of `cv1.conv.weight` to `cv0` and `old_weight[half:]` to `cv1`, splits the BN tensors the same way, and copies plain attributes such as `c` and Ultralytics' routing tags `f` and `i`. `make_prunable` swaps every `C2f` in `yolo.model`.

Both prune functions build their pruner with `_pruner`:

- `model.requires_grad_(True)` comes first. Ultralytics checkpoints load with `requires_grad=False`, and the tracer only sees layers autograd tracks. The docstring records the two failures this fixes: a reloaded checkpoint pruned nothing, and a fresh model never pruned stages 0-1, which sit before the first newly built `C2f_v2`.
- `ignored` lists every `Detect` module, so the head's outputs are never cut.
- `getattr(tp.pruner, "GroupNormPruner", tp.pruner.MetaPruner)` uses whichever class the installed torch-pruning has. `GroupMagnitudeImportance()` does the scoring.
- `model.train()` runs before construction, because in training mode the head returns raw feature maps, which trace cleanly. The `finally` block restores the old mode with `model.train(was)`.
- `iterative_steps=1` means the whole ratio comes off in one `step()`.

`structured_prune(yolo, ratio)` is one line, `_pruner(yolo.model, example, pruning_ratio=ratio).step()`: every group loses the same fraction. The headless scripts (`prune_sweep.py`, `finetune.py`, `iterative_prune.py`) use it.

`structured_prune_per_stage(yolo, stage_fracs)` is what the cockpit's Export and Regrow call. `_stage_pruner` collects `head`, the ids of every module inside `Detect`. For each other `Conv2d`, `top_stage(name)` maps `model.5.cv1.conv` to `model.5`; if that stage has a fraction, the conv goes into `ratio_dict` with `min(float(stage_fracs[st]), 0.9)`, a cap that keeps a stage from vanishing. `pruning_ratio=0.0` leaves every other group alone. Groups cross stage boundaries, so cutting one stage's outputs also shrinks the next stage's inputs. `removal_plan` builds the same pruner to list the channels without removing them (see prune_and_regrow).

## What the simulations show

The toy-MLP analogue, not YOLO, is `prune_structured` (lab/results/prune_structured.json). A 3x256 MLP teacher loses whole neurons, ranked by the L1 norm of their weights, $\lVert w_j \rVert_1$, then gets no recovery, fine-tuning on labels, or distillation from the teacher (KD) for 3,000 steps, the same budget as training fresh. Accuracy on 20 classes; recovered numbers are 2-seed means.

| hidden width | no recovery | labels | KD | fresh net, KD |
|---|---|---|---|---|
| 128 | 0.4247 | 0.9091 | 0.9155 | 0.9176 |
| 64 | 0.1685 | 0.8909 | 0.90825 | 0.9042 |
| 32 | 0.0816 | 0.8426 | 0.86035 | 0.8469 |
| 16 | 0.0443 | 0.68005 | 0.68565 | 0.70075 |

Halving the width with no recovery drops accuracy to 0.4247, the same kind of cliff the detector shows. After recovery the ranking rule mattered little: at width 128 with KD, L1, activation, Taylor and random ranking land between 0.9154 and 0.9182. With 750 steps instead, the pruned start was ahead at width 64 (0.8927 against 0.86855 for a fresh net). An MLP has no adds or concatenations, so the toy says nothing about dependency groups.

## When it breaks

- One-shot cuts fall off a cliff. STATUS.md records bird mAP@50 of 0.000 after a uniform 5% prune, still 0.000 at 5, 10, 20 and 30% after the `requires_grad` fix. Always fine-tune afterwards.
- Where you cut matters. Unfine-tuned on 125 images, stage 0 at 25% scores 0.4100 and stage 9 at 30% scores 0.1265 (STATUS.md).
- Fewer operations did not mean faster on a Mac. STATUS.md records a 30% prune that cut MACs (multiply-accumulates) by 35% yet made MPS latency 6% worse. After the `requires_grad` fix the same prune gives 1.81M params and 2.78G MACs, from 3.16M and 4.40G.
- Only YOLOv8's `C2f` has a traceable rewrite. YOLO11's `C3k2` and `C2PSA` blocks have none.
- `transfer_weights` reads `cv1.bn`, so `make_prunable` must run before `fuse` folds BN into the convs. server.py does it in that order.
- Coupling makes a cut's saving hard to guess: STATUS.md records stages 15 and 21 at 25% once estimated at 2% of compute saved, while Export saved 8%.

## Going further

To run it, install `requirements.txt`, run `python build_eval_set.py`, then `uvicorn server:app --port 8000 --app-dir .` and open http://localhost:8000. In Structure, press Scan (it masks 10% of each stage and measures the drop), then "Apply cut" or click nodes (each is 5% of a stage's channels). "Export real pruned model" posts to `/api/export`, which prunes a fresh `yolov8n.pt` with `structured_prune_per_stage`, reports params, GMACs, latency and real mAP before regrow, and saves `runs/pigeon_pruned.pt`. The toy version is `lab/compress.py:structured_prune`.
