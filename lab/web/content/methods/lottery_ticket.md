## The idea

When a pruned network recovers well, is that because its trained weights were good, or because its *mask* (the set of surviving connections) picked a sub-network that was trainable from the start? The lottery ticket experiment separates the two. Find a mask by pruning a trained network, reset the surviving weights to their values at initialisation (or very early in training), and retrain with the mask fixed. If this "ticket" trains well while the same mask with fresh random weights does not, the initial values mattered. Frankle and Carbin called it a winning ticket.

## How it works

1. Initialise a network, save its weights `W0`, train it, and also save an early snapshot `Wk` (the *rewind point*).
2. Prune the trained net by magnitude to get a 0/1 mask `m`.
3. Reset: `W = W0 * m` (rewind to init) or `W = Wk * m` (late rewinding).
4. Retrain with the mask fixed.
5. Repeat steps 2-4, pruning more each round (*iterative magnitude pruning*).
6. Controls at the same sparsity: the ticket mask with a new random init, and a random mask on the original init.

## The code, line by line

`lottery_ticket` in `lab/experiments/shrink.py` uses a 2x128 net with 20,480 weights. `init = net(hid, seed=s)` is `W0`. `scratch(init.copy(), s, callback=...)` trains a copy for 3,000 steps, and `snap.setdefault("early", n.copy()) if ep == 1` keeps the weights after the second epoch: step 126, about 4% of training.

`branches` maps each arm to its starting weights: `init`, `early`, or `None` (keep the trained weights and fine-tune). Each round sets `sp = 1 - 0.5 ** rnd` and calls `mask_src = magnitude_prune(state[name], sp)` on that arm's net from the previous round. `magnitude_prune` counts `sp` over all weights, so each round halves the survivors and each mask is a subset of the last.

`m = mask_src.copy() if start is None else start.copy()` picks the weights, `m.masks = [...]` attaches the mask, and `m.apply_masks()` zeroes the pruned entries. `scratch(m, s, lr=FT_LR if start is None else 3e-3)` retrains for 3,000 steps. Both learning rates are 3e-3, so the arms differ only in their starting weights. `state[name] = m` carries each arm's lineage into the next round.

The controls follow. "ticket mask, fresh init" puts the rewind-to-init mask on `net(hid, seed=999 + s)`. "random mask, original init" draws `rr.random(mk.shape) < mk.mean()` per layer, keeping each layer's density but not its pattern.

`run_lottery` in `lab/methods.py` is a short version for this page: 2x64, 1,000 steps per round, one rewind point.

## What the simulations show

From `lottery_ticket` (2x128, mean of 2 seeds; dense accuracy 0.904):

| sparsity | rewind to init | rewind to early | fine-tune trained | ticket mask, fresh init | random mask, original init |
|---|---|---|---|---|---|
| 50% | 0.9038 | 0.9022 | 0.9013 | 0.8921 | 0.8966 |
| 75% | 0.8986 | 0.8951 | 0.8847 | 0.8758 | 0.8778 |
| 87.5% | 0.8690 | 0.8690 | 0.8506 | 0.8344 | 0.8361 |
| 93.75% | 0.7884 | 0.7887 | 0.7332 | 0.7196 | 0.7083 |
| 96.9% | 0.6118 | 0.6115 | 0.4866 | 0.4801 | 0.3219 |

This is the lottery-ticket pattern. Rewound tickets hold up best at every sparsity, and the gap grows as the net gets sparser. At 96.9% (640 weights), with the original init, the ticket mask scores 0.6118 and a random mask 0.3219. With the ticket mask, the original init scores 0.6118 and a fresh one 0.4801. So both the mask and its initial values matter. Rewinding to init and to the early snapshot are indistinguishable here. Fine-tuning the trained weights, what most pruning pipelines do, lands with the controls once sparsity passes 90%.

Page runs (2x64, one seed, **prune rounds** 6; dense 0.8514):

| round (removed) | rewind to early | rewind to init | random reinit |
|---|---|---|---|
| 1 (50%) | 0.8502 | 0.8417 | 0.7872 |
| 2 (75%) | 0.7934 | 0.7757 | 0.7586 |
| 3 (87.5%) | 0.6421 | 0.6621 | 0.6489 |
| 4 (93.8%) | 0.4524 | 0.4751 | 0.4551 |
| 6 (98.4%) | 0.1889 | 0.2219 | 0.2030 |

In the smaller, shorter-trained net, rewinding beats a random reinit by 0.055 to 0.063 at 50% and by 0.017 to 0.035 at 75%. From 87.5% on the three arms sit within 0.035 of each other and swap order from round to round: one seed cannot separate them.

The toy cannot test large vision models, where Frankle et al. found that rewinding to init stops working and an early rewind point is needed.

## When it breaks

- **Compounding schedules.** If a pruning function ranks only survivors, a schedule of total sparsities overshoots. An earlier version of this experiment did that: its "75%" round was really 87.5%. Check the nonzero count.
- **Tickets are expensive to find.** You need a full training run per round. It is a probe, not a shortcut.
- **Small or briefly trained nets** show little ticket effect where it should matter most, at high sparsity: in the page runs, rewound and randomly reinitialized tickets sit within 0.035 of each other from 87.5% on.
- **Masks are unstructured**, so there is no speedup without sparse kernels.

## Try this

- Set **prune rounds** to 1 and switch **rewind to** between early and random reinit.
- Set **prune rounds** to 3 (87.5%) and compare all three rewind points. The `params` column reads 916, of which 148 are biases.
- Run **prune rounds** 6 with each rewind and watch accuracy fall towards a 244-parameter net.

## Going further

Frankle et al. (2020) showed that rewinding to an early iteration makes tickets work at scale, and linked it to *linear mode connectivity*: past that point, retraining with a different data order ends in the same basin. The YOLOv8n cockpit never rewinds. Its loop fine-tunes the trained weights (the "fine-tune trained" arm), the practical choice when the original training run cannot be repeated, as with a pretrained YOLOv8n. A ticket experiment on it would need the COCO training recipe and many full trainings.
