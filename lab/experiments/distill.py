"""Group 1 -- distillation fundamentals: when does a teacher's soft output beat
plain labels, and which knobs matter?"""

from __future__ import annotations

from ..common import (QUICK, SEEDS, STUDENT, TINY, agg, data, distill, ev, experiment, refs,
                      scratch, teacher)
from ..train import SP, Hint

G = "1 · Distillation fundamentals"


@experiment("size_ladder", G, "KD across the student-size ladder",
            "How much does distillation help at each student size, and how much of "
            "that comes from the teacher's soft labels vs. from extra unlabeled inputs?")
def size_ladder():
    t = teacher()
    rows = []
    for w in [4, 8, 16, 32, 64, 128]:
        hid = [w, w]
        for s in SEEDS:
            arms = {
                "labels only (scratch)": lambda: scratch(hid, s),
                "KD, labeled inputs": lambda: distill(hid, t, s, transfer="train"),
                "KD, + unlabeled pool": lambda: distill(hid, t, s, transfer="pool"),
            }
            for name, make in arms.items():
                rows.append(dict(width=w, method=name, seed=s, **ev(make(), t)))
    return dict(rows=agg(rows, ["width", "method"]), refs=refs(t), plots=[
        dict(x="width", y="acc", group="method", xlog=True, hlines=["teacher", "bayes"]),
        dict(x="width", y="target_ap", group="method", xlog=True, hlines=["teacher", "bayes"]),
    ])


@experiment("temperature_alpha", G, "Temperature x KD-weight grid",
            "Which temperature T and what mix of KD vs. ground-truth CE (alpha) works "
            "best when the student only sees the labeled set?")
def temperature_alpha():
    t = teacher()
    rows = []
    for s in SEEDS:
        base = ev(scratch(TINY, s), t)
        for T in [1, 2, 4, 8, 16]:
            rows.append(dict(T=T, alpha="0 (labels only)", seed=s, **base))
            for a in [0.5, 0.9, 1.0]:
                m = distill(TINY, t, s, transfer="train", T=T, alpha=a)
                rows.append(dict(T=T, alpha=str(a), seed=s, **ev(m, t)))
    return dict(rows=agg(rows, ["T", "alpha"]), refs=refs(t), plots=[
        dict(x="T", y="acc", group="alpha", xlog=True, hlines=["teacher"]),
        dict(x="T", y="nll", group="alpha", xlog=True),
    ])


@experiment("loss_zoo", G, "The distillation-loss zoo",
            "Same student, same teacher, same inputs: which *form* of teacher signal "
            "transfers the most? (logits, probabilities, top-k only, hard labels, "
            "hidden features, relational structure)")
def loss_zoo():
    t = teacher()
    rows = []
    for s in SEEDS:
        w = TINY[-1]
        arms = {
            "labels only (scratch)": lambda: scratch(TINY, s),
            "hard pseudo-labels": lambda: distill(TINY, t, s, kind="hard"),
            "KD T=1": lambda: distill(TINY, t, s, T=1),
            "KD T=4": lambda: distill(TINY, t, s, T=4),
            "reverse-KL T=4": lambda: distill(TINY, t, s, T=4, kind="rkl"),
            "MSE on logits": lambda: distill(TINY, t, s, kind="mse"),
            "top-3 probs only": lambda: distill(TINY, t, s, kind="topk"),
            "top-1 prob + uniform rest": lambda: distill(TINY, t, s, kind="topk", k=1),
            "KD + FitNets hint": lambda: distill(TINY, t, s, extra_terms=[
                Hint(len(TINY) - 1, 2, w, 256, weight=0.05, seed=s)]),
            "KD + similarity-preserving": lambda: distill(TINY, t, s, extra_terms=[
                SP(len(TINY) - 1, 2, weight=50.0)]),
            "hint only (+labels)": lambda: distill(TINY, t, s, alpha=0.0, extra_terms=[
                Hint(len(TINY) - 1, 2, w, 256, weight=0.05, seed=s)]),
        }
        for name, make in arms.items():
            rows.append(dict(method=name, seed=s, **ev(make(), t)))
    return dict(rows=agg(rows, ["method"]), refs=refs(t), plots=[
        dict(x="method", y="acc", kind="bar", hlines=["teacher"]),
        dict(x="method", y="fidelity", kind="bar"),
    ])


@experiment("label_budget", G, "Distillation as a substitute for labels",
            "With only a few hundred labels, how much do a strong teacher and "
            "unlabeled inputs buy -- and does the teacher need the big labeled set?")
def label_budget():
    t_full = teacher()
    rows = []
    for n in [250, 500, 1000, 2000, 4000, 8000]:
        d = data(n_train=n)
        t_n = teacher(n_train=n)
        for s in SEEDS:
            arms = {
                "labels only (scratch)": lambda: scratch(STUDENT, s, d=d),
                "KD from big teacher, labeled inputs": lambda: distill(STUDENT, t_full, s, d=d, transfer="train", alpha=0.5),
                "KD from big teacher, + pool": lambda: distill(STUDENT, t_full, s, d=d, transfer="pool", alpha=0.5),
                "KD from teacher trained on same n, + pool": lambda: distill(STUDENT, t_n, s, d=d, transfer="pool", alpha=0.5),
            }
            for name, make in arms.items():
                rows.append(dict(n_labels=n, method=name, seed=s, **ev(make(), t_full)))
            rows.append(dict(n_labels=n, method="(teacher trained on same n)", seed=s, **ev(t_n)))
    return dict(rows=agg(rows, ["n_labels", "method"]), refs=refs(t_full), plots=[
        dict(x="n_labels", y="acc", group="method", xlog=True, hlines=["teacher"]),
    ])


@experiment("label_noise", G, "Distilling through label noise",
            "When the labels are noisy, is a teacher trained on those same noisy "
            "labels still a better target than the labels themselves?")
def label_noise():
    rows = []
    t_clean = teacher()
    for nz in [0.0, 0.2, 0.4, 0.6]:
        d = data(label_noise=nz)
        t_noisy = teacher(label_noise=nz)
        t_short = teacher(label_noise=nz, epochs=2 if QUICK else 10)
        rows.append(dict(noise=nz, method="(teacher, noisy labels)", seed=0, **ev(t_noisy)))
        rows.append(dict(noise=nz, method="(teacher, noisy labels, short training)", seed=0, **ev(t_short)))
        for s in SEEDS:
            arms = {
                "labels only (scratch)": lambda: scratch(STUDENT, s, d=d),
                "labels + smoothing 0.2": lambda: scratch(STUDENT, s, d=d, smoothing=0.2),
                "KD from noisy teacher + pool": lambda: distill(STUDENT, t_noisy, s, d=d),
                "KD from short-trained noisy teacher + pool": lambda: distill(STUDENT, t_short, s, d=d),
            }
            for name, make in arms.items():
                rows.append(dict(noise=nz, method=name, seed=s, **ev(make(), t_clean)))
    return dict(rows=agg(rows, ["noise", "method"]), refs=refs(t_clean), plots=[
        dict(x="noise", y="acc", group="method", hlines=["teacher", "bayes"]),
    ])
