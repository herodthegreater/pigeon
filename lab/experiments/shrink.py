"""Group 4 -- shrinking a trained model: pruning (unstructured, structured,
lottery tickets), quantization, low-rank, weight sharing, depth removal, and
the end-to-end size/accuracy frontier across all of them."""

from __future__ import annotations

import numpy as np

from ..common import (SEEDS, STEPS, STUDENT, agg, cached, data, distill, ev,
                      experiment, net, refs, scratch, teacher)
from ..compress import (drop_layers, kmeans_share, low_rank, magnitude_prune, neuron_importance,
                        quantize, structured_prune)
from ..nn import MLP

G = "4 · Shrinking a trained model"
FT = STEPS // 2          # recovery budget after a cut
FT_LR = 1e-3             # gentler LR when continuing from trained weights


def storage_bits(m: MLP, bits: int = 32, group: int | None = None, sparse: bool = False) -> int:
    """Bytes-on-disk proxy: nonzero weights x bits, + fp16 scales, + fp32 biases,
    + a 1-bit/position occupancy bitmap if stored sparse."""
    nnz = sum(int((m.eff(i) != 0).sum()) for i in range(m.n_layers))
    dense = sum(w.size for w in m.W)
    total = (nnz if sparse else dense) * bits + sum(b.size for b in m.b) * 32
    if bits < 32:
        total += (int(np.ceil(dense / group)) if group else m.n_layers) * 16
    if sparse:
        total += dense
    return int(total)


def kd_ft(m, t, s, steps=None, **kw):
    return distill(m, t, s, steps=steps or FT, lr=FT_LR, **kw)


def ce_ft(m, s, steps=None):
    return scratch(m, s, steps=steps or FT, lr=FT_LR)


def dense_width_for(nnz: int, d=12, C=20) -> int:
    """Width w of a 2-hidden-layer MLP with ~nnz weights: d*w + w*w + w*C = nnz."""
    b = d + C
    return max(2, int(round((-b + np.sqrt(b * b + 4 * nnz)) / 2)))


# ---------------------------------------------------------------------------
@experiment("prune_unstructured", G, "Magnitude pruning: the cliff and the regrow",
            "How sparse can the teacher get? Compare one-shot cuts with and without "
            "recovery, recovery by labels vs. by distilling from the unpruned self, "
            "gradual pruning, and a dense small model with the same weight count.")
def prune_unstructured():
    t = teacher()
    rows = []
    for sp in [0.5, 0.75, 0.9, 0.95, 0.98, 0.99]:
        for s in SEEDS[:2]:
            cut = magnitude_prune(t, sp)
            nnz = cut.macs()
            arms = {
                "one-shot, no recovery": lambda: cut,
                "one-shot + label fine-tune": lambda: ce_ft(cut.copy(), s),
                "one-shot + KD from unpruned self": lambda: kd_ft(cut.copy(), t, s),
            }
            for name, mk in arms.items():
                rows.append(dict(sparsity=sp, method=name, seed=s, **ev(mk())))
            # gradual: 5 equal steps up to the target sparsity, KD between each
            m = t
            for k in range(1, 6):
                m = kd_ft(magnitude_prune(m, 1 - (1 - sp) ** (k / 5)), t, s, steps=FT // 5)
            rows.append(dict(sparsity=sp, method="gradual (5 steps) + KD", seed=s, **ev(m)))
            w = dense_width_for(nnz)
            rows.append(dict(sparsity=sp, method="dense small net, same #weights, KD", seed=s,
                             dense_width=w, **ev(distill([w, w], t, s))))
    return dict(rows=agg(rows, ["sparsity", "method"]), refs=refs(t), plots=[
        dict(x="sparsity", y="acc", group="method", hlines=["teacher"]),
        dict(x="macs", y="acc", group="method", xlog=True, hlines=["teacher"]),
    ])


# ---------------------------------------------------------------------------
@experiment("prune_structured", G, "Structured (neuron) pruning: criteria x recovery",
            "Delete whole neurons so the net is genuinely smaller. Which importance "
            "criterion picks the right ones, and is a pruned teacher a better start "
            "than training the small shape from scratch?")
def prune_structured():
    d = data()
    t = teacher()
    probe = d.train.take(2000)
    rows = []
    rng = np.random.default_rng(0)
    scores = {c: neuron_importance(t, probe.X, probe.y, c, rng=rng)
              for c in ["l1", "act", "taylor", "random"]}
    for keep in [0.5, 0.25, 0.125, 0.0625]:
        width = int(256 * keep)
        for crit, sc in scores.items():
            cut = structured_prune(t, keep, sc)
            rows.append(dict(width=width, criterion=crit, recovery="none", seed=0, **ev(cut)))
            for s in SEEDS[:2]:
                rows.append(dict(width=width, criterion=crit, recovery="labels", seed=s,
                                 **ev(ce_ft(cut.copy(), s))))
                rows.append(dict(width=width, criterion=crit, recovery="KD", seed=s,
                                 **ev(kd_ft(cut.copy(), t, s))))
        for s in SEEDS[:2]:
            hid = [width] * 3
            rows.append(dict(width=width, criterion="(fresh init)", recovery="labels", seed=s,
                             **ev(scratch(hid, s))))
            rows.append(dict(width=width, criterion="(fresh init)", recovery="KD", seed=s,
                             **ev(distill(hid, t, s))))
    rows = agg(rows, ["width", "criterion", "recovery"])
    for r in rows:
        r["arm"] = f"{r['criterion']} / {r['recovery']}"
    return dict(rows=rows, refs=refs(t), plots=[
        dict(x="width", y="acc", group="arm", xlog=True, hlines=["teacher"],
             filter={"recovery": ["KD"]}, title="recovered with KD (+ fresh-init baseline)"),
        dict(x="width", y="acc", group="arm", xlog=True, hlines=["teacher"],
             filter={"criterion": ["taylor", "(fresh init)"]}, title="recovery method"),
        dict(x="width", y="acc", group="arm", xlog=True, filter={"recovery": ["none"]},
             title="no recovery: the cliff"),
    ])


# ---------------------------------------------------------------------------
@experiment("lottery_ticket", G, "Lottery tickets: is the mask or the init the secret?",
            "Iterative magnitude pruning, then retrain the sparse net from (a) its original "
            "init, (b) an early-training rewind point, (c) a fresh random init, (d) its "
            "trained weights. Plus a random mask control.")
def lottery_ticket():
    d = data()
    hid = [128, 128]
    rows = []
    for s in SEEDS[:2]:
        init = net(hid, seed=s)
        # train once, snapshotting the weights after epoch 2 (~4% of training):
        # the "late rewinding" point of Frankle et al.
        snap: dict = {}
        trained = scratch(init.copy(), s, callback=lambda ep, n: (
            snap.setdefault("early", n.copy()) if ep == 1 else None))
        early = snap["early"]
        branches = {"rewind to init": init, "rewind to early": early, "fine-tune trained": None}
        state = {k: trained for k in branches}
        for rnd in range(1, 6):
            sp = 1 - 0.5 ** rnd
            for name, start in branches.items():
                mask_src = magnitude_prune(state[name], sp)
                m = (mask_src.copy() if start is None else start.copy())
                m.masks = [None if mk is None else mk.copy() for mk in mask_src.masks]
                m.apply_masks()
                m = scratch(m, s, lr=FT_LR if start is None else 3e-3)
                state[name] = m
                rows.append(dict(sparsity=sp, method=name, seed=s, **ev(m)))
            # fresh init with the winning-ticket mask
            m = net(hid, seed=999 + s)
            m.masks = [None if mk is None else mk.copy() for mk in state["rewind to init"].masks]
            m.apply_masks()
            rows.append(dict(sparsity=sp, method="ticket mask, fresh init", seed=s, **ev(scratch(m, s))))
            # random mask, same per-layer sparsity
            m = net(hid, seed=s)
            rr = np.random.default_rng(rnd + 10 * s)
            m.masks = [(rr.random(mk.shape) < mk.mean()).astype(np.float32)
                       for mk in state["rewind to init"].masks]
            m.apply_masks()
            rows.append(dict(sparsity=sp, method="random mask, original init", seed=s, **ev(scratch(m, s))))
        rows.append(dict(sparsity=0.0, method="dense", seed=s, **ev(trained)))
    return dict(rows=agg(rows, ["sparsity", "method"]), refs=refs(), plots=[
        dict(x="sparsity", y="acc", group="method"),
    ])


# ---------------------------------------------------------------------------
@experiment("quantization", G, "Quantization: PTQ vs. QAT, labels vs. KD, big vs. small",
            "How few bits per weight survive? Does a per-group scale rescue low bits, does "
            "quantization-aware training rescue more, and do small models tolerate it worse?")
def quantization():
    t = teacher()
    models = {"teacher 3x256": t,
              "student 2x32 (KD-trained)": cached("q_student32", lambda: distill(STUDENT, t, 0))}
    rows = []
    for mname, m0 in models.items():
        r0 = ev(m0)
        rows.append(dict(model=mname, bits=32, method="fp32", seed=0, **r0,
                         kbits=storage_bits(m0) / 1000))
        for bits in [8, 6, 4, 3, 2, 1]:
            for g, gname in [(None, "per-tensor"), (16, "group-16")]:
                q = quantize(m0, bits, g)
                rows.append(dict(model=mname, bits=bits, method=f"PTQ {gname}", seed=0,
                                 kbits=storage_bits(m0, bits, g) / 1000, **ev(q)))
            if bits <= 4:
                for how in ["labels", "KD"]:
                    for s in SEEDS[:2]:
                        m = m0.copy()
                        m.qbits, m.qgroup = bits, 16
                        m = ce_ft(m, s) if how == "labels" else kd_ft(m, t, s)
                        rows.append(dict(model=mname, bits=bits, method=f"QAT group-16 + {how}",
                                         seed=s, kbits=storage_bits(m0, bits, 16) / 1000, **ev(m)))
    rows = agg(rows, ["model", "bits", "method"])
    for r in rows:
        r["series"] = f"{r['model']}: {r['method']}"
    return dict(rows=rows, refs=refs(t), plots=[
        dict(x="bits", y="acc", group="series", xlog=True, filter={"model": ["teacher 3x256"]},
             title="teacher"),
        dict(x="bits", y="acc", group="series", xlog=True,
             filter={"model": ["student 2x32 (KD-trained)"]}, title="small student"),
    ])


# ---------------------------------------------------------------------------
@experiment("low_rank", G, "Low-rank factorization (SVD)",
            "Replace each 256x256 matrix with a rank-r product. How low can r go, and how "
            "does it compare per-parameter with deleting neurons?")
def low_rank_exp():
    t = teacher()
    d = data()
    probe = d.train.take(2000)
    sc = neuron_importance(t, probe.X, probe.y, "taylor")
    rows = []
    for r in [128, 64, 32, 16, 8, 4, 2]:
        lr_ = low_rank(t, r)
        rows.append(dict(rank_or_width=r, method="SVD, no recovery", seed=0, **ev(lr_)))
        for s in SEEDS[:2]:
            rows.append(dict(rank_or_width=r, method="SVD + KD", seed=s, **ev(kd_ft(lr_.copy(), t, s))))
    for w in [128, 64, 32, 16]:
        for s in SEEDS[:2]:
            cut = structured_prune(t, [w, w, w], sc)
            rows.append(dict(rank_or_width=w, method="neuron prune (taylor) + KD", seed=s,
                             **ev(kd_ft(cut, t, s))))
    return dict(rows=agg(rows, ["rank_or_width", "method"]), refs=refs(t), plots=[
        dict(x="params", y="acc", group="method", xlog=True, hlines=["teacher"]),
    ])


# ---------------------------------------------------------------------------
@experiment("weight_sharing", G, "Weight sharing (k-means codebooks) vs. uniform bits",
            "Deep-Compression style: each layer keeps only k distinct values. At the same "
            "bits/weight, does a learned, non-uniform codebook beat uniform rounding?")
def weight_sharing():
    t = teacher()
    st = cached("q_student32", lambda: distill(STUDENT, t, 0))
    rows = []
    for mname, m0 in {"teacher 3x256": t, "student 2x32": st}.items():
        for k in [2, 4, 8, 16, 32]:
            bits = int(np.log2(k))
            rows.append(dict(model=mname, bits=bits, method="uniform (per-tensor)", seed=0,
                             **ev(quantize(m0, bits + 1 if bits < 16 else bits))))
            km = kmeans_share(m0, k)
            rows.append(dict(model=mname, bits=bits, method="k-means codebook", seed=0, **ev(km)))
            # fine-tune, re-snap, repeat (what Deep Compression's centroid tuning approximates)
            m = km
            for rnd in range(3):
                m = kmeans_share(kd_ft(m.copy(), t, rnd, steps=FT // 3), k)
            rows.append(dict(model=mname, bits=bits, method="k-means + KD + re-snap", seed=0, **ev(m)))
    rows = agg(rows, ["model", "bits", "method"])
    for r in rows:
        r["series"] = f"{r['model']}: {r['method']}"
    return dict(rows=rows, refs=refs(t), notes=(
        "Uniform at 'b bits' here uses b+1-bit symmetric quantization so both schemes store "
        "2^b-ish levels (symmetric signed quant at b bits has 2^b - 1 levels)."), plots=[
        dict(x="bits", y="acc", group="series", filter={"model": ["teacher 3x256"]}, title="teacher"),
        dict(x="bits", y="acc", group="series", filter={"model": ["student 2x32"]}, title="student"),
    ])


# ---------------------------------------------------------------------------
def _greedy_drop(m: MLP, k: int, X, y) -> list[int]:
    """Pick k hidden->hidden layers to delete, greedily, by accuracy on (X, y)."""
    dropped: list[int] = []
    cand = [i for i in range(1, m.n_layers - 1) if m.sizes[i] == m.sizes[i + 1]]
    for _ in range(k):
        best, best_acc = None, -1
        for c in cand:
            if c in dropped:
                continue
            trial = drop_layers(m, dropped + [c])
            acc = (trial.logits(X).argmax(1) == y).mean()
            if acc > best_acc:
                best, best_acc = c, acc
        dropped.append(best)
    return dropped


@experiment("layer_drop", G, "Deleting whole layers: plain vs. residual",
            "Remove k of 8 hidden layers from a deep teacher. Residual nets are said to "
            "behave like ensembles of shallow paths -- does that make depth removable? "
            "And is initializing a shallow student from the teacher's layers worth it?")
def layer_drop():
    d = data()
    probe = d.train.take(2000)
    deep = [128] * 8
    rows = []
    teachers = {"plain": teacher(hidden=deep), "residual": teacher(hidden=deep, res=True)}
    for kind, t in teachers.items():
        rows.append(dict(kind=kind, dropped=0, recovery="none", seed=0, **ev(t)))
        for k in [1, 2, 4, 6]:
            drop = _greedy_drop(t, k, probe.X, probe.y)
            cut = drop_layers(t, drop)
            rows.append(dict(kind=kind, dropped=k, recovery="none", seed=0, **ev(cut)))
            for s in SEEDS[:2]:
                rows.append(dict(kind=kind, dropped=k, recovery="KD", seed=s,
                                 **ev(kd_ft(cut.copy(), t, s))))
    # DistilBERT-style init: a 4-layer residual student from every other teacher layer
    t = teachers["residual"]
    init_rows = []
    for s in SEEDS[:2]:
        fromt = drop_layers(t, [2, 4, 6, 7])  # keeps hidden layers 0,1,3,5 (+ head)
        rand = net([128] * 4, seed=s, res=True)
        init_rows.append(dict(init="every-other teacher layer", seed=s, **ev(distill(fromt, t, s))))
        init_rows.append(dict(init="random", seed=s, **ev(distill(rand, t, s))))
    rows = agg(rows, ["kind", "dropped", "recovery"])
    for r in rows:
        r["series"] = f"{r['kind']}, {r['recovery']}"
    return dict(rows=rows + agg(init_rows, ["init"]), refs=refs(teachers["plain"]), plots=[
        dict(x="dropped", y="acc", group="series", filter={"kind": ["plain", "residual"]}),
        dict(x="init", y="acc", kind="bar", filter={"init": ["every-other teacher layer", "random"]},
             title="4-layer residual student, KD, different init"),
    ])


# ---------------------------------------------------------------------------
@experiment("compression_frontier", G, "Every route to small, on one size/accuracy chart",
            "Storage bits vs. accuracy for: small dense students, pruned teachers (sparse "
            "and structured), quantized versions of each, and the order of operations.")
def compression_frontier():
    t = teacher()
    rows = []
    s = 0
    rows.append(dict(family="teacher", config="fp32", kbits=storage_bits(t) / 1e3, **ev(t)))
    for w in [8, 16, 32, 64]:
        m = distill([w, w], t, s)
        rows.append(dict(family="small dense student (KD)", config=f"2x{w} fp32",
                         kbits=storage_bits(m) / 1e3, **ev(m)))
        for b in [8, 4]:
            q = m.copy(); q.qbits, q.qgroup = b, 16
            q = kd_ft(q, t, s)
            rows.append(dict(family="small dense student (KD) + QAT", config=f"2x{w} {b}b",
                             kbits=storage_bits(m, b, 16) / 1e3, **ev(q)))
    d = data()
    probe = d.train.take(2000)
    sc = neuron_importance(t, probe.X, probe.y, "taylor")
    for w in [64, 32, 16]:
        m = kd_ft(structured_prune(t, [w] * 3, sc), t, s)
        rows.append(dict(family="structured-pruned teacher (KD)", config=f"3x{w} fp32",
                         kbits=storage_bits(m) / 1e3, **ev(m)))
        q = m.copy(); q.qbits, q.qgroup = 4, 16
        q = kd_ft(q, t, s)
        rows.append(dict(family="structured-pruned teacher (KD) + QAT", config=f"3x{w} 4b",
                         kbits=storage_bits(m, 4, 16) / 1e3, **ev(q)))
    for sp in [0.9, 0.97, 0.99]:
        m = kd_ft(magnitude_prune(t, sp), t, s)
        rows.append(dict(family="sparse teacher (KD)", config=f"{int(sp*100)}% fp32",
                         kbits=storage_bits(m, sparse=True) / 1e3, **ev(m)))
        # prune -> quantize
        q = m.copy(); q.qbits, q.qgroup = 4, 16
        q = kd_ft(q, t, s)
        rows.append(dict(family="sparse teacher: prune -> QAT 4b", config=f"{int(sp*100)}% 4b",
                         kbits=storage_bits(m, 4, 16, sparse=True) / 1e3, **ev(q)))
        # quantize -> prune: QAT the dense teacher first, then prune + KD with quant on
        qd = t.copy(); qd.qbits, qd.qgroup = 4, 16
        qd = kd_ft(qd, t, s)
        p = magnitude_prune(qd, sp)
        p.qbits, p.qgroup = 4, 16
        p = kd_ft(p, t, s)
        rows.append(dict(family="sparse teacher: QAT 4b -> prune", config=f"{int(sp*100)}% 4b",
                         kbits=storage_bits(p, 4, 16, sparse=True) / 1e3, **ev(p)))
    return dict(rows=rows, refs=refs(t), plots=[
        dict(x="kbits", y="acc", group="family", kind="scatter", xlog=True, hlines=["teacher"],
             label="config", connect=True),
    ])

