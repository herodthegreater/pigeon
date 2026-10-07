"""Group 5 -- making a small model better, with and without a bigger model."""

from __future__ import annotations

from ..common import (SEEDS, STEPS, STUDENT, TINY, agg, data, distill, ev, experiment, net,
                      refs, scratch, teacher, train)
from ..compress import neuron_importance, structured_prune
from ..nn import MLP
from ..train import CE, KD, OnlineSource

G = "5 · Making small models better"


def overparam(hidden: list[int], seed: int, expand: int = 4) -> MLP:
    """ExpandNets / RepVGG idea: train every weight matrix as a product of two
    (a -> expand*max(a,b) -> b, no nonlinearity between), then multiply them
    back together. Same inference-time network, different optimization path."""
    d = data()
    base = [d.world.spec.d] + list(hidden) + [d.world.C]
    sizes, lin = [base[0]], set()
    for i in range(len(base) - 1):
        lin.add(len(sizes) - 1)
        sizes += [expand * max(base[i], base[i + 1]), base[i + 1]]
    return MLP(sizes, seed=seed, linear=lin)


@experiment("small_tricks", G, "Bag of tricks for a small model",
            "Fixed small architecture. What actually moves its accuracy: regularizers, "
            "longer training, weight averaging, over-parameterize-then-collapse, "
            "self-distillation, a teacher, or train-big-then-shrink?")
def small_tricks():
    d = data()
    t = teacher()
    C = d.world.C
    rows = []
    for hid in [TINY, STUDENT]:
        w = hid[0]
        for s in SEEDS:
            def wide_then_prune():
                big = scratch([128, 128], s)
                sc = neuron_importance(big, d.train.X[:2000], d.train.y[:2000], "taylor")
                return distill(structured_prune(big, [w, w], sc), big, s, transfer="train",
                               alpha=0.5)

            def self_distill():
                return distill(hid, scratch(hid, s), s + 100, transfer="train", T=2, alpha=0.5)

            def everything():
                src = OnlineSource(d.train.X, d.train.y, C, t, aug="mixup", extra_X=d.pool.X)
                return train(net(hid, s), src, [KD(T=1)], seed=s, steps=4 * STEPS, ema=0.999)

            arms = {
                "baseline (labels, CE)": lambda: scratch(hid, s),
                "label smoothing 0.1": lambda: scratch(hid, s, smoothing=0.1),
                "mixup": lambda: train(net(hid, s), OnlineSource(d.train.X, d.train.y, C, None),
                                       [CE()], seed=s),
                "EMA of weights": lambda: scratch(hid, s, ema=0.999),
                "4x longer training": lambda: scratch(hid, s, steps=4 * STEPS),
                "over-parameterize, then collapse": lambda: scratch(overparam(hid, s), s).collapse_linear(),
                "self-distillation (born-again)": self_distill,
                "train 2x128, prune to size, self-KD": wide_then_prune,
                "KD from teacher (labeled inputs)": lambda: distill(hid, t, s, transfer="train", alpha=0.5),
                "KD from teacher + pool": lambda: distill(hid, t, s),
                "KD + pool + online mixup + EMA, 4x long": everything,
            }
            for name, mk in arms.items():
                m = mk()
                assert m.sizes == [d.world.spec.d] + hid + [C], (name, m.sizes)
                rows.append(dict(student="x".join(map(str, hid)), method=name, seed=s, **ev(m, t)))
    return dict(rows=agg(rows, ["student", "method"]), refs=refs(t), plots=[
        dict(x="method", y="acc", kind="bar", filter={"student": ["16x16"]}, title="2x16 student"),
        dict(x="method", y="acc", kind="bar", filter={"student": ["32x32"]}, title="2x32 student"),
    ])


@experiment("train_big_then_compress", G, "Train big, then compress",
            "To end up with a 2xW network, is it better to train 2xW directly or to train "
            "something much wider and prune down to 2xW? (Li et al. 2020)")
def train_big_then_compress():
    d = data()
    probe = d.train.take(2000)
    rows = []
    for target in [16, 32]:
        for mult in [1, 2, 4, 8, 16]:
            start = target * mult
            for s in SEEDS[:2]:
                big = scratch([start, start], s)
                if mult == 1:  # baseline: the same extra budget, no pruning
                    # (no self-KD arm here: distilling a net into an exact copy of
                    # itself has zero gradient)
                    m = scratch(big, s, steps=STEPS // 2)
                    rows.append(dict(target=f"final 2x{target}", start_width=start, method="labels",
                                     seed=s, **ev(m)))
                    continue
                sc = neuron_importance(big, probe.X, probe.y, "taylor")
                cut = structured_prune(big, [target, target], sc)
                rows.append(dict(target=f"final 2x{target}", start_width=start, method="labels", seed=s,
                                 **ev(scratch(cut.copy(), s, steps=STEPS // 2))))
                rows.append(dict(target=f"final 2x{target}", start_width=start, method="self-KD", seed=s,
                                 **ev(distill(cut.copy(), big, s, steps=STEPS // 2))))
    rows = agg(rows, ["target", "start_width", "method"])
    for r in rows:
        r["series"] = f"{r['target']}, recover by {r['method']}"
    return dict(rows=rows, refs=refs(), plots=[
        dict(x="start_width", y="acc", group="series", xlog=True),
    ])

