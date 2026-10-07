## The idea

A convolution layer outputs a stack of feature maps called *channels*, one per filter. Deleting a channel removes that filter and one input slice of every layer that reads it, so the model gets smaller and does less arithmetic. With residual adds and concatenations, one channel is read by several layers, and all of them must lose it together or the shapes stop matching. That set is a *dependency group*. The torch-pruning library finds the groups by tracing the network; pigeon adds one rewrite so YOLOv8's main block can be traced.

## How it works

1. Trace: run one example image through the model and record which layer feeds which.
2. Group: for each output channel, collect everything tied to it: the conv filter, its batch-norm (BN) entries, the input slice of each consumer, and anything joined by a residual add or a concatenation.
3. Score each group by the magnitude of its weights. A small score means little signal flows through it.
4. Remove the lowest-scoring fraction by slicing the tensors, so they physically shrink.
5. Leave the output channels of the Detect head (the layer that emits boxes and class scores) alone.

Example: YOLOv8's Bottleneck computes `x + cv2(cv1(x))`. Cutting channel 7 from `cv2`'s output forces cutting channel 7 from `x`, and so from the layer that produced `x` and every layer that reads it.

## The code, line by line

`C2f` is Ultralytics' main building block. Its forward splits one 1x1 conv's output in half with `chunk(2, 1)`. The torch-pruning tracer can't follow `chunk`; the module docstring records that a generic pruner then removes zero channels.

`C2f_v2` does the same math with two convs, `self.cv0` and `self.cv1`, each producing `self.c` channels. `forward` builds `y = [self.cv0(x), self.cv1(x)]`, runs each Bottleneck in `self.m` on the newest entry (`y.extend(m(y[-1]) for m in self.m)`), and feeds `torch.cat(y, 1)` to `self.cv2`. Every split is now a separate layer the tracer can see.

`transfer_weights` makes the new block compute exactly what the old one did. It reuses `cv2` and `m`, then splits the old `cv1.conv.weight` by output row: `old_weight[:half]` goes to `cv0`, `old_weight[half:]` to `cv1`. The loop over `"weight", "bias", "running_mean", "running_var"` splits the BN the same way. The last loop copies plain attributes such as `c`.

`replace_c2f_with_c2f_v2` walks the module tree and swaps each `C2f`, reading its settings from the live module: `shortcut` from the Bottleneck's `add` flag (`infer_shortcut`), groups from `child.m[0].cv2.conv.groups`, and the width ratio `e = child.c / child.cv2.conv.out_channels`. `make_prunable` runs it on `yolo.model`.

`structured_prune(yolo, ratio)`: `ignored` lists the `Detect` modules. `model.train()` comes before building the pruner because in training mode the head returns raw feature maps, which trace cleanly; `model.train(was_training)` restores the mode afterwards. `getattr(tp.pruner, "GroupNormPruner", tp.pruner.MetaPruner)` uses whichever pruner class the installed torch-pruning has. `GroupMagnitudeImportance()` scores each group by weight magnitude. With `iterative_steps=1`, the whole `ratio` comes off in one `pruner.step()`, which edits the model in place. The same ratio applies to every group.

`structured_prune_per_stage(yolo, stage_fracs)` is what the cockpit's Export calls. It builds `ratio_dict` with one entry per `Conv2d` outside the head. `top_stage(name)` maps `model.5.cv1.conv` to `model.5`, and if the UI asked for a cut in that stage the conv gets `min(float(stage_fracs[st]), 0.9)`; the cap stops a stage from disappearing. Every other layer gets `pruning_ratio=0.0`. Groups cross stage boundaries, so cutting stage 5's outputs also shrinks stage 6's inputs.

## What the simulations show

The toy-MLP analogue, not YOLO, is `prune_structured` (lab/results/prune_structured.json). A 3x256 MLP teacher (accuracy 0.9139 on 20 classes) loses whole neurons, ranked by L1 weight norm, then gets no recovery, fine-tuning on labels, or distillation from the teacher (KD). Recovered numbers are 2-seed means.

| hidden width | no recovery | labels | KD | fresh net, KD |
|---|---|---|---|---|
| 128 | 0.4247 | 0.909 | 0.9108 | 0.915 |
| 64 | 0.1685 | 0.8727 | 0.84895 | 0.8993 |
| 32 | 0.0816 | 0.7297 | 0.638 | 0.83475 |
| 16 | 0.0443 | 0.49645 | 0.4388 | 0.6704 |

Halving the width with no recovery drops accuracy from 0.9139 to 0.4247, the same kind of cliff the detector shows. After recovery the ranking rule mattered little: at width 128, L1, activation, Taylor and random ranking all land between 0.90285 and 0.909 with labels. At narrow widths a fresh net of the same shape trained with KD won. An MLP has no adds or concatenations, so the toy says nothing about dependency groups.

## When it breaks

- One-shot cuts fall off a cliff. STATUS.md records that a 5% one-shot prune took bird mAP@50 to 0.000. Always fine-tune afterwards (see prune_and_regrow).
- Fewer operations did not mean faster on a Mac. A 30% prune cut params from 3.16M to 1.82M and MACs (multiply-accumulates) from 4.40G to 2.86G, yet MPS latency rose 6%.
- Only YOLOv8's `C2f` has a traceable rewrite. YOLO11's `C3k2` and `C2PSA` blocks have none.
- `transfer_weights` reads `cv1.bn`, so a model already fused by `predict` (BN folded into the conv) can't be made prunable.
- Re-pruning a reloaded checkpoint currently does nothing: params stay at 2.775M across steps (STATUS.md).
- On YOLOv8n the head is `model.22`, but `stage_table` excludes `model.23`, so the head shows up as a column in the cockpit grid. Cuts there change the live mAP, but Export ignores them because `Detect` is in `ignored`.

## Going further

To run it, install `requirements.txt`, run `python build_eval_set.py`, then `uvicorn server:app --port 8000` and open http://localhost:8000. In the Structure module, click nodes in a column (each node is 5% of that stage's channels), then press "Export real pruned model". `/api/export` loads a fresh `yolov8n.pt`, calls `make_prunable` and `structured_prune_per_stage`, reports params, MACs (shown as GFLOP) and latency, and saves `runs/pigeon_pruned.pt`. Headless: `real_prune_test.py` (one 30% cut) and `prune_sweep.py` (several ratios). The toy version is `lab/compress.py:structured_prune`.
