## The idea

If a pruned network recovers well, is that because its trained weights were good, or because its *mask* (which connections survive) picked a sub-network that was trainable all along? The lottery ticket experiment separates the two. Find a mask by pruning a trained network. Then reset the surviving weights to the values they had at initialisation, or very early in training, and retrain with the mask fixed. If this "ticket" trains as well as the dense net, while the same mask with fresh random weights does not, then the initial values mattered. Frankle and Carbin called such a sub-network a winning ticket.

## How it works

1. Initialise a network and save its weights `W0`. Train it, and also save a snapshot `Wk` from early in training (the *rewind point*).
2. Prune the trained net by magnitude to get a 0/1 mask `m`.
3. Reset: `W = W0 * m` (rewind to init) or `W = Wk * m` (late rewinding).
4. Retrain with the mask fixed.
5. Repeat steps 2-4, pruning more each round. This is *iterative magnitude pruning*.
6. Controls, at the same sparsity: the same mask with a new random init, and a random mask on the original init.

## The code, line by line

`lottery_ticket` in `lab/experiments/shrink.py` uses a 2x128 net with 20,480 weights. `init = net(hid, seed=s)` is `W0`. `scratch(init.copy(), s, callback=...)` trains a copy for 3,000 steps. The callback runs after every epoch, and `snap.setdefault("early", n.copy()) if ep == 1` keeps the weights at the end of the second epoch. With 8,000 examples and batch size 128 that is step 126, about 4% of training.

`branches` maps each arm to its starting weights: `init`, `early`, or `None`, meaning "keep the trained weights and fine-tune". `state` starts every arm from the trained net. Each round sets `sp = 1 - 0.5 ** rnd` (50%, 75%, 87.5%, ...) and calls `mask_src = magnitude_prune(state[name], sp)` on that arm's retrained net from the previous round. `magnitude_prune` counts `sp` over all weights, already-pruned ones included, so each round removes half of the weights still alive, and the mask of round `r` is always a subset of the mask of round `r - 1`.

`m = mask_src.copy() if start is None else start.copy()` picks the weights. `m.masks = [...]` attaches the new mask and `m.apply_masks()` zeroes the pruned weights. `scratch(m, s, lr=FT_LR if start is None else 3e-3)` retrains for 3,000 steps. `FT_LR` in `shrink.py` is now also 3e-3, so every arm uses the same optimizer settings and differs only in its starting weights. Each arm keeps its own mask lineage through `state[name] = m`.

The two controls follow. "ticket mask, fresh init" puts the rewind-to-init arm's mask on `net(hid, seed=999 + s)`. "random mask, original init" draws `rr.random(mk.shape) < mk.mean()` for each layer, keeping that layer's density but not its pattern.

`run_lottery` in `lab/methods.py` is a shorter version for this page: 2x64, 1,000 steps per round, and only the rewind point you choose. It calls `magnitude_prune(m, 1 - 0.5 ** r)` on the previous round's net, exactly as the experiment does.

## What the simulations show

From `lottery_ticket` (2x128, mean of 2 seeds; dense accuracy 0.904). The saved results were produced by an earlier `magnitude_prune` that measured sparsity as a fraction of the *surviving* weights, so the rounds compounded: round 2 removed 75% of the 50% left. The real sparsity, from `macs` out of 20,480, is in the second column:

| label | real sparsity | rewind to init | rewind to early | fine-tune trained | ticket mask, fresh init | random mask, original init |
|---|---|---|---|---|---|---|
| 0.5 | 50% | 0.9038 | 0.9022 | 0.9005 | 0.8921 | 0.8966 |
| 0.75 | 87.5% | 0.8500 | 0.8522 | 0.8125 | 0.8303 | 0.8337 |
| 0.875 | 98.4% | 0.1392 | 0.1333 | 0.1227 | 0.1310 | 0.0759 |

At 87.5% sparsity the rewound tickets beat both controls by about 0.02 accuracy, a real but small effect. Fine-tuning the trained weights (at a lower learning rate) did worst there. By 98.4% (320 weights left) every arm is close to chance, and the rows past that have 20, then 1, nonzero weights.

Page runs with the current code (2x64, one seed; dense 0.8511):

| round (removed) | rewind to early | rewind to init | random reinit |
|---|---|---|---|
| 1 (50%) | 0.8501 | 0.8403 | 0.7875 |
| 2 (75%) | 0.7968 | 0.7771 | 0.7633 |
| 3 (87.5%) | 0.6505 | 0.6493 | 0.6380 |
| 4 (93.8%) | 0.4159 | 0.4747 | 0.4470 |
| 6 (98.4%) | 0.2299 | 0.2098 | 0.1911 |

Rewinding beats a random reinit by 0.06 at 50% and the gap narrows to about 0.01 by 87.5%. At 93.8% the order changes, which at one seed is within noise.

The toy cannot test the regime where the hypothesis is most interesting: large vision models, where Frankle et al. found that rewinding to init stops working and rewinding to an early point is needed.

## When it breaks

- **Compounding schedules.** If the pruning function ranks only survivors, a schedule of total sparsities overshoots fast, as in the saved results above. Always check the nonzero count.
- **Tickets are not cheaper to find.** You need the full training run to get the mask, then a full retrain per round. It is a scientific probe, not a compression shortcut.
- **Small or capacity-starved nets** show little difference between ticket and random mask, as the 2x64 runs do.
- **Masks are unstructured**, so tickets bring no speedup without sparse kernels (see Magnitude pruning).
- **Learning-rate confounds.** Fine-tuning trained weights and retraining rewound ones are sensitive to the learning rate in different ways. A comparison is only fair when both use the same settings, as they do here.

## Try this

- Set **prune rounds** to 1 and switch **rewind to** between early and random reinit. At a true 50% sparsity, look at how much the original values matter.
- Set **prune rounds** to 3 (87.5%) and compare all three rewind points. Check the `params` column to confirm the real size: 916, of which 148 are biases.
- Run **prune rounds** 6 with each rewind. Watch the accuracy fall as the network approaches 244 parameters, most of them biases.

## Going further

On real networks, Frankle et al. (2020) showed that rewinding to an early iteration rather than to init is what makes tickets work at scale. They linked this to *linear mode connectivity*: the point after which retraining from the same weights with different data order always ends in the same basin. pigeon never rewinds. Its loop keeps the trained weights and fine-tunes them (the "fine-tune trained" arm), which is the practical choice when the original training run cannot be repeated, as with a pretrained YOLOv8n. Running a ticket experiment on it would need the COCO training recipe and many full trainings.
