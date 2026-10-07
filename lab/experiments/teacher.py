"""Group 2 -- teacher edge cases: is a better teacher always a better target?"""

from __future__ import annotations

import numpy as np

from ..common import (QUICK, SEEDS, STEPS, STUDENT, TINY, T_EPOCHS, agg, data, distill, ev,
                      experiment, net, refs, scratch, teacher, train, transfer_X)
from ..train import CE, KD, FixedSource, fit

G = "2 · Teacher edge cases"


@experiment("capacity_gap", G, "Capacity gap & teacher assistants",
            "Does a bigger teacher always make a better student? When the gap is huge, "
            "does an intermediate 'teacher assistant' help?")
def capacity_gap():
    rows = []
    teachers = {w: teacher(hidden=[w, w, w]) for w in [16, 32, 64, 128, 256, 512]}
    for w, t in teachers.items():
        tr = ev(t)
        for sw in [8, 32]:
            for s in SEEDS:
                r = ev(distill([sw, sw], t, s))
                rows.append(dict(teacher_width=w, student=f"student 2x{sw}", seed=s,
                                 teacher_acc=tr["acc"], **r))
    # teacher-assistant chains to a tiny student
    big = teachers[512]
    for s in SEEDS:
        rows.append(dict(teacher_width=512, student="chain", path="512 -> 8 (direct)", seed=s,
                         **ev(distill([8, 8], big, s))))
        for ta in [128, 32]:
            mid = distill([ta, ta], big, s)
            rows.append(dict(teacher_width=512, student="chain", path=f"512 -> {ta} -> 8", seed=s,
                             **ev(distill([8, 8], mid, s))))
    gap = [r for r in rows if r["student"] != "chain"]
    chain = [r for r in rows if r["student"] == "chain"]
    return dict(rows=agg(gap, ["teacher_width", "student"]) + agg(chain, ["path"]),
                refs=refs(), plots=[
        dict(x="teacher_width", y="acc", group="student", xlog=True,
             filter={"student": ["student 2x8", "student 2x32"]}),
        dict(x="teacher_width", y="teacher_acc", group="student", xlog=True,
             filter={"student": ["student 2x32"]}, title="teacher accuracy"),
        dict(x="path", y="acc", kind="bar", filter={"student": ["chain"]}),
    ])


@experiment("teacher_checkpoints", G, "Which teacher checkpoint to distill from",
            "Is the most accurate teacher the best teacher? Distill from snapshots "
            "along training, plus an over-trained, unregularized teacher.")
def teacher_checkpoints():
    d = data()
    snaps = {}
    keep = {1, 2, 4, 8, 16, 32, T_EPOCHS} if not QUICK else {1, 3, T_EPOCHS}

    def build():
        m = net([256, 256, 256], seed=0)
        fit(m, FixedSource(d.train.X, d.train.y, d.world.C), [CE()], epochs=T_EPOCHS, lr=2e-3,
            callback=lambda ep, n: snaps.__setitem__(ep + 1, n.copy()) if ep + 1 in keep else None)
        return snaps
    from ..common import cached
    snaps = cached(f"teacher_snaps_{T_EPOCHS}", build)
    teachers = {f"epoch {e}": t for e, t in sorted(snaps.items())}
    teachers["4x longer, no weight decay"] = teacher(epochs=4 * T_EPOCHS, wd=0.0)
    teachers["trained on 1k labels"] = teacher(n_train=1000)
    rows = []
    for name, t in teachers.items():
        tr = ev(t)
        for s in SEEDS:
            r = ev(distill(STUDENT, t, s))
            rows.append(dict(teacher=name, seed=s, teacher_acc=tr["acc"], teacher_nll=tr["nll"],
                             teacher_ece=tr["ece"], teacher_kl_bayes=tr["kl_bayes"], **r))
    return dict(rows=agg(rows, ["teacher"]), refs=refs(), plots=[
        dict(x="teacher_acc", y="acc", kind="scatter", label="teacher"),
        dict(x="teacher_kl_bayes", y="acc", kind="scatter", label="teacher", xlog=True),
    ])


@experiment("teacher_smoothing", G, "Label-smoothed teachers",
            "Label smoothing makes a better *model*; does it make a better *teacher*?")
def teacher_smoothing():
    rows = []
    for ls in [0.0, 0.1, 0.2, 0.4]:
        t = teacher(smoothing=ls)
        tr = ev(t)
        rows.append(dict(smoothing=ls, method="(teacher itself)", seed=0, **tr))
        for T in [1, 4]:
            for s in SEEDS:
                rows.append(dict(smoothing=ls, method=f"student, KD T={T}", seed=s,
                                 teacher_acc=tr["acc"], **ev(distill(TINY, t, s, T=T))))
    return dict(rows=agg(rows, ["smoothing", "method"]), refs=refs(), plots=[
        dict(x="smoothing", y="acc", group="method",
             filter={"method": ["student, KD T=1", "student, KD T=4"]}),
        dict(x="smoothing", y="acc", group="method", filter={"method": ["(teacher itself)"]},
             title="teacher accuracy"),
    ])


@experiment("ensemble_teacher", G, "Ensembles as teachers",
            "Is an ensemble of teachers a better target than one? Average logits or "
            "average probabilities?")
def ensemble_teacher():
    d = data()
    ts = [teacher(seed=i) for i in range(1 if QUICK else 8)]
    X, y = transfer_X(d)
    rows = []
    for k in ([1] if QUICK else [1, 2, 4, 8]):
        sub = ts[:k]
        logit_avg = np.mean([t.logits(X) for t in sub], 0)
        from ..train import softmax
        prob_avg = np.log(np.mean([softmax(t.logits(X)) for t in sub], 0) + 1e-8)
        test_probs = np.mean([softmax(t.logits(d.test.X)) for t in sub], 0)
        ens_acc = float((test_probs.argmax(1) == d.test.y).mean())
        for how, tl in [("avg logits", logit_avg), ("avg probs", prob_avg)]:
            for s in SEEDS:
                m = train(net(STUDENT, s), FixedSource(X, y, d.world.C, t_logits=tl),
                          [KD(T=4)], seed=s)
                rows.append(dict(n_teachers=k, method=f"student <- {how}", seed=s,
                                 ensemble_acc=ens_acc, **ev(m)))
        rows.append(dict(n_teachers=k, method="(ensemble itself)", seed=0, acc=ens_acc))
    return dict(rows=agg(rows, ["n_teachers", "method"]), refs=refs(ts[0]), plots=[
        dict(x="n_teachers", y="acc", group="method", xlog=True,
             filter={"method": ["student <- avg logits", "student <- avg probs"]}),
    ])


@experiment("oracle_teacher", G, "Distilling from the true posterior",
            "If the teacher's soft labels were *perfect* (the exact Bayes posterior), "
            "how much better would students get? This bounds what any teacher can give.")
def oracle_teacher():
    d = data()
    t = teacher()
    X, y = transfer_X(d)
    bayes = np.concatenate([d.train.bayes, d.pool.bayes])
    soft = np.log(bayes + 1e-8)
    rows = []
    for w in [8, 16, 32, 64, 128]:
        hid = [w, w]
        for s in SEEDS:
            arms = {
                "labels only (scratch)": lambda: scratch(hid, s),
                "KD from trained teacher": lambda: distill(hid, t, s),
                "KD from Bayes posterior": lambda: train(net(hid, s), FixedSource(
                    X, None, d.world.C, t_logits=soft), [KD(T=1)], seed=s),  # T=1: exact probs
                "Bayes-argmax labels (denoised)": lambda: train(net(hid, s), FixedSource(
                    X, None, d.world.C, t_logits=soft), [KD(kind="hard")], seed=s),
            }
            for name, make in arms.items():
                rows.append(dict(width=w, method=name, seed=s, **ev(make(), t)))
    return dict(rows=agg(rows, ["width", "method"]), refs=refs(t), plots=[
        dict(x="width", y="acc", group="method", xlog=True, hlines=["teacher", "bayes"]),
        dict(x="width", y="kl_bayes", group="method", xlog=True),
    ])


@experiment("born_again", G, "Born-again networks (self-distillation)",
            "Can a model teach a copy of *itself* to be better? Repeat for several "
            "generations, small and big.")
def born_again():
    d = data()
    rows = []
    for arch_name, hid in [("2x32 (student-size)", STUDENT), ("3x256 (teacher-size)", [256, 256, 256])]:
        for transfer in ["train", "pool"]:
            for s in SEEDS[:2]:
                prev = scratch(hid, s) if hid != [256, 256, 256] else teacher(seed=s)
                rows.append(dict(arch=arch_name, transfer=transfer, generation=0, seed=s, **ev(prev)))
                for g in range(1, 4):
                    cur = distill(hid, prev, seed=100 * g + s, transfer=transfer, T=2,
                                  alpha=0.5)
                    rows.append(dict(arch=arch_name, transfer=transfer, generation=g, seed=s,
                                     **ev(cur, prev)))
                    prev = cur
    rows = agg(rows, ["arch", "transfer", "generation"])
    for r in rows:
        r["series"] = f"{r['arch']}, {r['transfer']}"
    return dict(rows=rows, refs=refs(), plots=[
        dict(x="generation", y="acc", group="series", hlines=["bayes"]),
    ])


@experiment("fidelity_gap", G, "Students don't match their teachers",
            "Even a student big enough to copy the teacher exactly often doesn't. "
            "How does agreement with the teacher grow with capacity and training length?")
def fidelity_gap():
    t = teacher()
    rows = []
    for hid in [[16, 16], [64, 64], [256, 256, 256]]:
        for mult in [0.25, 1, 4]:
            steps = int(STEPS * mult)
            for s in SEEDS[:2]:
                m = distill(hid, t, s, steps=steps)
                rows.append(dict(student="x".join(map(str, hid)), steps=steps, seed=s, **ev(m, t)))
    return dict(rows=agg(rows, ["student", "steps"]), refs=refs(t), plots=[
        dict(x="steps", y="fidelity", group="student", xlog=True),
        dict(x="steps", y="acc", group="student", xlog=True, hlines=["teacher"]),
    ])


@experiment("biased_teacher", G, "A systematically wrong teacher",
            "The teacher learned a bias: half its 'bird' training labels said 'sibling'. "
            "KD copies the bias faithfully -- can mixing in clean labels undo it?")
def biased_teacher():
    base = data()
    rng = np.random.default_rng(5)
    yb = base.train.y.copy()
    flip = (yb == 0) & (rng.random(len(yb)) < 0.5)
    yb[flip] = 1

    def build():
        m = net([256, 256, 256])
        fit(m, FixedSource(base.train.X, yb, base.world.C), [CE()], epochs=T_EPOCHS, lr=2e-3)
        return m
    from ..common import cached
    tb = cached(f"teacher_biased_{T_EPOCHS}", build)
    rows = [dict(alpha="(biased teacher)", seed=0, **ev(tb))]
    for a in [1.0, 0.9, 0.5, 0.25, 0.0]:
        for s in SEEDS:
            rows.append(dict(alpha=str(a), seed=s, **ev(distill(STUDENT, tb, s, alpha=a, transfer="train"))))
    return dict(rows=agg(rows, ["alpha"]), refs=refs(teacher()), plots=[
        dict(x="alpha", y="target_ap", kind="bar", hlines=["teacher"]),
        dict(x="alpha", y="acc", kind="bar", hlines=["teacher"]),
    ])

