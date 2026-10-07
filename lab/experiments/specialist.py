"""Group 6 -- the pigeon question: keep ONE capability (class 0, "bird") out of
a 20-class generalist. How small can a specialist get, how should its head be
shaped, and what data must the recovery step see?"""

from __future__ import annotations

import numpy as np

from ..common import (SEEDS, STEPS, agg, data, distill, ev, experiment, refs, teacher, train,
                      transfer_X)
from ..compress import collapse_head, neuron_importance, structured_prune
from ..nn import MLP
from ..train import CE, KD, FixedSource, evaluate
from ..world import TARGET

G = "6 · Specialization (the pigeon question)"


def groups(world) -> dict[str, np.ndarray]:
    """Label-space collapses, all with the target as output 0.

    binary    bird | everything else
    siblings  bird | its 3 superclass siblings (the confusers) | rest
    supers    bird | 3 siblings | each of the 4 other superclasses
    """
    C = world.C
    sib = world.siblings(TARGET)
    tsup = int(world.superclass(TARGET))
    others = [k for k in range(world.spec.n_super) if k != tsup]

    def build(assign):
        Gm = np.zeros((C, max(assign) + 1), np.float32)
        Gm[np.arange(C), assign] = 1
        return Gm
    a_bin = [0 if c == TARGET else 1 for c in range(C)]
    a_sib = [0 if c == TARGET else (1 + sib.index(c) if c in sib else 4) for c in range(C)]
    a_sup = [0 if c == TARGET else (1 + sib.index(c) if c in sib
                                    else 4 + others.index(int(world.superclass(c)))) for c in range(C)]
    return {"binary": build(a_bin), "siblings": build(a_sib), "supers": build(a_sup)}


def spec_eval(m: MLP) -> dict:
    return evaluate(m, data().test, full_space=False)


def spec_student(width: int, k: int, seed: int) -> MLP:
    d = data()
    return MLP([d.world.spec.d, width, width, k], seed=seed)


def spec_distill(m: MLP, t: MLP, Gm: np.ndarray, seed: int, X=None, steps=None, lr=3e-3):
    """KD in a collapsed label space: teacher probs are summed into groups."""
    if X is None:
        X, _ = transfer_X(data())
    return train(m, FixedSource(X, None, Gm.shape[1], teacher=t), [KD(T=4, group=Gm)],
                 steps=steps, seed=seed, lr=lr)


@experiment("specialist_size", G, "How small can a one-class specialist be?",
            "Target-class AP vs. size for a 20-way generalist student and for specialists "
            "whose heads keep less and less of the teacher's label space.")
def specialist_size():
    d = data()
    t = teacher()
    Gs = groups(d.world)
    yb = (d.train.y != TARGET).astype(np.int64)  # 0 = bird, 1 = rest
    rows = []
    for w in [2, 4, 8, 16, 32]:
        for s in SEEDS:
            arms = {
                "generalist 20-way, KD": lambda: distill([w, w], t, s),
                "binary head, labels only": lambda: train(
                    spec_student(w, 2, s), FixedSource(d.train.X, yb, 2), [CE()], seed=s),
                "binary head, KD": lambda: spec_distill(spec_student(w, 2, s), t, Gs["binary"], s),
                "bird+siblings+rest head, KD": lambda: spec_distill(
                    spec_student(w, 5, s), t, Gs["siblings"], s),
                "bird+siblings+superclasses head, KD": lambda: spec_distill(
                    spec_student(w, 8, s), t, Gs["supers"], s),
            }
            for name, mk in arms.items():
                m = mk()
                r = ev(m) if name.startswith("generalist") else spec_eval(m)
                rows.append(dict(width=w, method=name, seed=s, **r))
    return dict(rows=agg(rows, ["width", "method"]), refs=refs(t), plots=[
        dict(x="width", y="target_ap", group="method", xlog=True, hlines=["teacher", "bayes"],
             ref_metric="target_ap"),
        dict(x="params", y="target_ap", group="method", xlog=True, hlines=["teacher"],
             ref_metric="target_ap"),
    ])


@experiment("specialist_prune", G, "Carving a specialist out of the generalist",
            "Prune the teacher's neurons for one class. Does a task-aware importance score "
            "(bird-vs-rest) keep a better bird detector than an all-class score, and does "
            "collapsing the head help recovery?")
def specialist_prune():
    d = data()
    t = teacher()
    Gs = groups(d.world)
    Gm = Gs["supers"]
    probe = d.train.take(2000)
    sc_all = neuron_importance(t, probe.X, probe.y, "taylor")
    sc_bird = neuron_importance(t, probe.X, probe.y, "taylor", group=Gs["binary"])
    overlap = []
    rows = []
    for w in [128, 64, 32, 16, 8, 4]:
        a = set(np.argsort(-sc_all[1])[:w]); b = set(np.argsort(-sc_bird[1])[:w])
        overlap.append(dict(width=w, overlap=len(a & b) / w))
        cut_all = structured_prune(t, [w] * 3, sc_all)
        cut_bird = structured_prune(t, [w] * 3, sc_bird)
        rows.append(dict(width=w, method="all-class importance, no recovery", seed=0, **ev(cut_all)))
        rows.append(dict(width=w, method="bird importance, no recovery", seed=0, **ev(cut_bird)))
        for s in SEEDS[:2]:
            ft = dict(steps=STEPS)  # same budget as the fresh specialist
            arms = {
                "all-class importance, 20-way KD": lambda: distill(cut_all.copy(), t, s, **ft),
                "all-class importance, collapsed head + grouped KD": lambda: spec_eval(
                    spec_distill(collapse_head(cut_all, Gm), t, Gm, s, **ft)),
                "bird importance, collapsed head + grouped KD": lambda: spec_eval(
                    spec_distill(collapse_head(cut_bird, Gm), t, Gm, s, **ft)),
                "fresh 3xW specialist, grouped KD": lambda: spec_eval(spec_distill(
                    MLP([d.world.spec.d, w, w, w, Gm.shape[1]], seed=s), t, Gm, s)),
            }
            for name, mk in arms.items():
                out = mk()
                r = out if isinstance(out, dict) else ev(out)
                rows.append(dict(width=w, method=name, seed=s, **r))
    return dict(rows=agg(rows, ["width", "method"]) + overlap, refs=refs(t), plots=[
        dict(x="width", y="target_ap", group="method", xlog=True, hlines=["teacher"],
             ref_metric="target_ap", filter={"method": [
                 "all-class importance, 20-way KD", "all-class importance, collapsed head + grouped KD",
                 "bird importance, collapsed head + grouped KD", "fresh 3xW specialist, grouped KD"]}),
        dict(x="width", y="target_ap", group="method", xlog=True, ref_metric="target_ap",
             filter={"method": ["all-class importance, no recovery", "bird importance, no recovery"]},
             title="no recovery"),
        dict(x="width", y="overlap", filter={"overlap": None}, xlog=True,
             title="fraction of kept layer-2 neurons shared by both criteria"),
    ])


@experiment("specialist_data", G, "Recovery data for a specialist: are bird images enough?",
            "pigeon's recovery set is 400 images that *all contain birds*. If the recovery "
            "data only shows positives, can the model still learn what is NOT a bird?")
def specialist_data():
    d = data()
    t = teacher()
    Gs = groups(d.world)
    Gm = Gs["supers"]
    X_all, _ = transfer_X(d)
    y_all = np.concatenate([d.train.y_clean, d.pool.y_clean])
    sib = d.world.siblings(TARGET)
    rng = np.random.default_rng(0)
    bird = X_all[y_all == TARGET]
    n_b = len(bird)
    pick = lambda m, n: X_all[m][rng.permutation(m.sum())[:n]]
    sets = {
        "everything (28k)": X_all,
        f"birds only ({n_b})": bird,
        "birds + siblings": X_all[(y_all == TARGET) | np.isin(y_all, sib)],
        "birds + equal # random others": np.concatenate([bird, pick(y_all != TARGET, n_b)]),
        "birds + noise negatives": np.concatenate([bird, rng.standard_normal(
            (n_b, d.world.spec.d)).astype(np.float32)]),
        f"random {2 * n_b} (no bird filter)": X_all[rng.permutation(len(X_all))[:2 * n_b]],
    }
    probe = d.train.take(2000)
    sc = neuron_importance(t, probe.X, probe.y, "taylor", group=Gs["binary"])
    rows = []
    for name, X in sets.items():
        for s in SEEDS[:2]:
            pruned = collapse_head(structured_prune(t, [32] * 3, sc), Gm)
            r = spec_eval(spec_distill(pruned, t, Gm, s, X=X))
            rows.append(dict(data=name, model="pruned teacher 3x32 (recovered)", seed=s, **r))
            r = spec_eval(spec_distill(spec_student(16, Gm.shape[1], s), t, Gm, s, X=X))
            rows.append(dict(data=name, model="fresh specialist 2x16", seed=s, **r))
    return dict(rows=agg(rows, ["data", "model"]), refs=refs(t), plots=[
        dict(x="data", y="target_ap", group="model", kind="bar", hlines=["teacher"],
             ref_metric="target_ap"),
    ])
