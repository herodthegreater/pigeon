"""The method registry: one entry per technique, shared by the website and the
test suite.

Every entry says which kind of model it belongs to (``family``), whether it
runs here (``status``), which functions implement it (``code``, shown on the
site), what knobs it has (``params``) and, if runnable, a ``run(params)`` that
trains/compresses a real model (most runs take seconds to a minute) and returns a comparison.
The tests iterate over this registry, so a method cannot appear on the site
without also being exercised by ``pytest``.

status:
    runnable   runs here, on the toy MLP world (numpy only)
    cockpit    real code for the YOLO detector; runs in the YOLOv8n cockpit
               (``uvicorn server:app``) with torch + ultralytics installed
    reference  needs a kind of model this repo doesn't have (e.g. a sequence
               model); explained, with code pointers, but not runnable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import layerwise as lw
from .common import (QUICK, STUDENT, cached, data, distill, ev, net, scratch, teacher, train,
                     transfer_X)
from .compress import (collapse_head, drop_layers, kbits, kmeans_share, low_rank,
                       magnitude_prune, neuron_importance, quantize, storage_bits,
                       structured_prune)
from .nn import MLP
from .train import CE, KD, SP, FixedSource, Hint, OnlineSource

STEPS = 100 if QUICK else 1000     # keeps most web runs under a minute
FT = STEPS // 2
FT_LR = 3e-3

FAMILIES = {
    "small": dict(title="Small networks", blurb=(
        "Every method runs here on a toy 20-class problem with a known best-possible answer. "
        "Start here: the ideas are the same at every scale, and most runs finish in under a minute.")),
    "llm": dict(title="Language-model methods", blurb=(
        "Algorithms invented for large language models. The layer-by-layer ones (GPTQ, AWQ, "
        "SmoothQuant, Wanda, SparseGPT, QLoRA, ShortGPT) are pure linear algebra on one weight "
        "matrix at a time, so they run here on the same toy network. Methods that need text, "
        "tokens or a KV cache are explained but marked as reference.")),
    "vision": dict(title="Vision / detector (YOLOv8n cockpit)", blurb=(
        "What changes when the model is a convolutional detector like YOLOv8: channels instead "
        "of neurons, dependency groups, batch-norm. The cockpit methods run in the YOLOv8n "
        "cockpit's web UI (cockpit/server.py) on a real model; the rest are explained for "
        "reference.")),
}

CATEGORIES = {
    "distill": "Distillation", "prune": "Pruning", "quantize": "Quantization",
    "factorize": "Factorization", "train": "Training small models", "specialize": "Specialization",
    "depth": "Depth reduction",
}


@dataclass
class P:
    name: str
    label: str
    kind: str                     # int | float | choice | bool
    default: object
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    choices: list | None = None
    help: str = ""

    def coerce(self, v):
        if v is None:
            return self.default
        if self.kind == "bool":
            return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
        if self.kind == "choice":
            if v not in self.choices:
                raise ValueError(f"{self.name}: {v!r} not in {self.choices}")
            return v
        v = int(v) if self.kind == "int" else float(v)
        return min(max(v, self.lo), self.hi)


@dataclass
class Method:
    id: str
    name: str
    family: str
    category: str
    status: str
    summary: str
    code: list[str] = field(default_factory=list)   # "path/to/file.py:Qual.name"
    params: list[P] = field(default_factory=list)
    run: Callable | None = None
    experiments: list[str] = field(default_factory=list)
    papers: list[str] = field(default_factory=list)

    def coerce(self, params: dict | None) -> dict:
        params = params or {}
        unknown = set(params) - {p.name for p in self.params}
        if unknown:
            raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
        return {p.name: p.coerce(params.get(p.name)) for p in self.params}

    def execute(self, params: dict | None = None) -> dict:
        if self.run is None:
            raise ValueError(f"{self.id} is not runnable here (status: {self.status})")
        return self.run(self.coerce(params))


REGISTRY: dict[str, Method] = {}


def method(**kw):
    def deco(fn):
        REGISTRY[kw["id"]] = Method(run=fn, status="runnable", **kw)
        return fn
    return deco


def reference(**kw):
    REGISTRY[kw["id"]] = Method(status=kw.pop("status", "reference"), **kw)


# ---------------------------------------------------------------------------
# result helpers
#
# A result is ready for the site to draw a bird per model ("bird-ready") when:
#   - every model row has ``kbits``, its storage in kilobits (compress.storage_bits:
#     weights at their format's bit width, fp32 biases, fp16 scales or fp32
#     codebooks, and for pruned nets only the surviving weights plus a 1-bit
#     occupancy bitmap);
#   - exactly one row has ``ref: True``: the uncompressed model the others are
#     measured against (species from its kbits / theirs, health from the
#     accuracy they keep);
#   - a row that isn't a model (the Bayes-optimal ceiling) has ``model: False``.
# Results whose models are all the same size have ``bird: False`` and say why
# in ``bird_reason``. lab/tests/test_birds.py checks all of this for every run.
# ---------------------------------------------------------------------------
def row(label: str, m: MLP, t: MLP | None = None, **extra) -> dict:
    """One model's line. ``kbits`` comes from the format the net declares
    (fake-quant bits, pruning masks; see ``compress.kbits``); a caller whose net
    was stored some other way passes ``kbits=`` itself."""
    r = ev(m, t)
    out = dict(label=label, acc=r["acc"], target_ap=r["target_ap"], params=r["params"])
    if "fidelity" in r:
        out["fidelity"] = r["fidelity"]
    if "kbits" not in extra:
        out["kbits"] = kbits(m)
    out.update(extra)
    return out


def teacher_row() -> dict:
    return row("teacher 3x256 (reference)", teacher(), ref=True)


def kd_student() -> MLP:
    """A 2x32 student distilled from the teacher; shared starting point."""
    return cached("student32_kdT1", lambda: distill(STUDENT, teacher(), 0))


def result(rows, note: str = "", chart: dict | None = None, metric: str = "acc",
           bird_reason: str = "") -> dict:
    out = dict(rows=rows, note=note, chart=chart, metric=metric, bird=not bird_reason)
    if bird_reason:
        out["bird_reason"] = bird_reason
    return out


W = lambda w: [w, w]  # noqa: E731  two hidden layers of width w


# ===========================================================================
# SMALL NETWORKS -- distillation
# ===========================================================================
@method(id="kd", name="Knowledge distillation (Hinton)", family="small", category="distill",
        summary="Train a small model to match a big model's softened output probabilities "
                "instead of (or as well as) the one-hot labels.",
        code=["lab/train.py:KD", "lab/train.py:softmax", "lab/common.py:distill"],
        params=[P("width", "student width", "int", 16, 2, 128, 2),
                P("T", "temperature", "float", 4.0, 1, 20, 0.5,
                  help="higher = softer teacher probabilities"),
                P("alpha", "weight on KD (vs. labels)", "float", 1.0, 0, 1, 0.05),
                P("transfer", "student sees", "choice", "pool",
                  choices=["train", "pool"], help="train = 8k labeled; pool = + 20k unlabeled")],
        experiments=["size_ladder", "temperature_alpha"],
        papers=["Hinton, Vinyals, Dean. Distilling the Knowledge in a Neural Network (2015)"])
def run_kd(p):
    t = teacher()
    hid = W(p["width"])
    return result([
        teacher_row(),
        row("student, labels only", scratch(hid, 0, steps=STEPS), t),
        row(f"student, KD T={p['T']:g} alpha={p['alpha']:g}",
            distill(hid, t, 0, T=p["T"], alpha=p["alpha"], transfer=p["transfer"], steps=STEPS), t),
    ])


@method(id="kd_losses", name="Distillation loss variants", family="small", category="distill",
        summary="Forward KL, reverse KL, logit MSE, hard pseudo-labels, and teachers that "
                "reveal only their top-k probabilities.",
        code=["lab/train.py:KD._teacher_probs", "lab/train.py:KD.__call__"],
        params=[P("kind", "loss", "choice", "rkl", choices=["kl", "rkl", "mse", "hard", "topk"]),
                P("k", "k (for topk)", "int", 3, 1, 19, 1),
                P("T", "temperature", "float", 4.0, 1, 20, 0.5),
                P("width", "student width", "int", 16, 2, 128, 2)],
        experiments=["loss_zoo"],
        papers=["Kim et al. Comparing Kullback-Leibler Divergence and Mean Squared Error Loss "
                "in Knowledge Distillation (2021)",
                "Gu et al. MiniLLM: Knowledge Distillation of Large Language Models (2023)"])
def run_kd_losses(p):
    t = teacher()
    hid = W(p["width"])
    return result([
        teacher_row(),
        row("labels only", scratch(hid, 0, steps=STEPS), t),
        row("forward KL (standard KD)", distill(hid, t, 0, T=p["T"], steps=STEPS), t),
        row(f"{p['kind']}" + (f" k={p['k']}" if p["kind"] == "topk" else ""),
            distill(hid, t, 0, T=p["T"], kind=p["kind"], k=p["k"], steps=STEPS), t),
    ])


@method(id="feature_kd", name="Feature & relational distillation", family="small",
        category="distill",
        summary="Match the teacher's hidden activations (FitNets) or the similarity "
                "structure between examples (similarity-preserving KD), not just its outputs.",
        code=["lab/train.py:Hint", "lab/train.py:SP"],
        params=[P("term", "extra loss", "choice", "sp", choices=["hint", "sp"]),
                P("weight", "loss weight", "float", 50.0, 0, 100, 0.05,
                  help="the two losses have different scales: sp works around 50, hint around 0.05"),
                P("width", "student width", "int", 16, 2, 128, 2)],
        experiments=["loss_zoo"],
        papers=["Romero et al. FitNets: Hints for Thin Deep Nets (2015)",
                "Tung & Mori. Similarity-Preserving Knowledge Distillation (2019)"])
def run_feature_kd(p):
    t = teacher()
    w = p["width"]
    hid = W(w)
    term = (Hint(1, 2, w, 256, weight=p["weight"]) if p["term"] == "hint"
            else SP(1, 2, weight=p["weight"]))
    return result([
        teacher_row(),
        row("labels only", scratch(hid, 0, steps=STEPS), t),
        row("KD", distill(hid, t, 0, steps=STEPS), t),
        row(f"KD + {p['term']}", distill(hid, t, 0, steps=STEPS, extra_terms=[term]), t),
    ])


@method(id="data_free_kd", name="Distilling without real data", family="small",
        category="distill",
        summary="Feed the teacher noise, or inputs synthesized by gradient descent on the "
                "teacher itself, and distill on those.",
        code=["lab/experiments/transfer.py:invert_teacher"],
        params=[P("source", "transfer inputs", "choice", "inversion",
                  choices=["real", "gaussian", "uniform", "inversion"]),
                P("n", "number of inputs", "int", 4000, 500, 28000, 500,
                  help="teacher inversion is cached per n; the first run at a new n is slow")],
        experiments=["transfer_set"],
        papers=["Lopes et al. Data-Free Knowledge Distillation for Deep Neural Networks (2017)",
                "Yin et al. Dreaming to Distill (DeepInversion) (2020)"])
def run_data_free(p):
    from .experiments.transfer import invert_teacher
    d, t = data(), teacher()
    rng = np.random.default_rng(0)
    n, dim = p["n"], d.world.spec.d
    X = {"real": lambda: np.concatenate([d.train.X, d.pool.X])[:n],
         "gaussian": lambda: rng.standard_normal((n, dim)).astype(np.float32),
         "uniform": lambda: rng.uniform(-3, 3, (n, dim)).astype(np.float32),
         "inversion": lambda: cached(f"inversion_{n}", lambda: invert_teacher(
             t, n, steps=30 if QUICK else 200))}[p["source"]]()
    m = train(net(STUDENT, 0), FixedSource(X, None, d.world.C, teacher=t), [KD(T=1)], steps=STEPS)
    return result([teacher_row(), row("labels only (8k)", scratch(STUDENT, 0, steps=STEPS), t),
                   row(f"KD on {p['source']} inputs ({n})", m, t)])


@method(id="function_matching", name="Function matching (patient, consistent teaching)",
        family="small", category="distill",
        summary="Query the teacher on freshly augmented inputs every step and train for "
                "longer: distillation as matching a function, not a dataset.",
        code=["lab/train.py:OnlineSource"],
        params=[P("aug", "augmentation", "choice", "mixup", choices=["mixup", "noise", "mixup+noise"]),
                P("length", "training length (x base)", "float", 2.0, 0.5, 24, 0.5,
                  help="online matching overtakes fixed targets only after ~8x; runs past 8x take several minutes")],
        experiments=["function_matching"],
        papers=["Beyer et al. Knowledge Distillation: A Good Teacher is Patient and Consistent (2022)"])
def run_function_matching(p):
    d, t = data(), teacher()
    steps = int(STEPS * p["length"])
    fixed = train(net(STUDENT, 0), FixedSource(d.train.X, None, d.world.C, teacher=t), [KD(T=1)],
                  steps=steps)
    online = train(net(STUDENT, 0), OnlineSource(d.train.X, d.train.y, d.world.C, t, aug=p["aug"]),
                   [KD(T=1)], steps=steps)
    return result([teacher_row(), row("fixed teacher targets on 8k inputs", fixed, t),
                   row(f"online {p['aug']}, teacher re-queried each step", online, t)])


@method(id="teacher_assistant", name="Teacher assistants", family="small", category="distill",
        summary="When the student is far smaller than the teacher, distill through a "
                "mid-sized model in between.",
        code=["lab/experiments/teacher.py:capacity_gap"],
        params=[P("student", "student width", "int", 8, 2, 32, 2),
                P("assistant", "assistant width", "int", 64, 8, 256, 8)],
        experiments=["capacity_gap"],
        papers=["Mirzadeh et al. Improved Knowledge Distillation via Teacher Assistant (2020)"])
def run_ta(p):
    t = teacher()
    mid = distill(W(p["assistant"]), t, 0, steps=STEPS)
    return result([teacher_row(),
                   row(f"assistant 2x{p['assistant']}", mid, t),
                   row(f"student 2x{p['student']}, direct from teacher",
                       distill(W(p["student"]), t, 0, steps=STEPS), t),
                   row(f"student 2x{p['student']}, via assistant",
                       distill(W(p["student"]), mid, 0, steps=STEPS), t)])


@method(id="ensemble_kd", name="Ensemble teachers", family="small", category="distill",
        summary="Average several teachers' predictions and distill the average into one "
                "small model.",
        code=["lab/experiments/teacher.py:ensemble_teacher"],
        params=[P("n", "teachers", "int", 3, 1, 4, 1),
                P("combine", "combine by averaging", "choice", "probs", choices=["logits", "probs"])],
        experiments=["ensemble_teacher"],
        papers=["Hinton et al. (2015), section 'Training ensembles of specialists'",
                "Allen-Zhu & Li. Towards Understanding Ensemble, Knowledge Distillation and "
                "Self-Distillation in Deep Learning (2020)"])
def run_ensemble(p):
    from .train import log_mean_softmax, softmax
    d = data()
    ts = [teacher(seed=i) for i in range(p["n"])]
    X, y = transfer_X(d)
    if p["combine"] == "logits":
        tl = np.mean([t.logits(X) for t in ts], 0)
    else:
        tl = log_mean_softmax([t.logits(X) for t in ts])
    probs = np.mean([softmax(t.logits(d.test.X)) for t in ts], 0)
    ens_acc = float((probs.argmax(1) == d.test.y).mean())
    m = train(net(STUDENT, 0), FixedSource(X, y, d.world.C, t_logits=tl), [KD(T=1)], steps=STEPS)
    # the reference is what gets distilled: all n teachers, stored side by side
    return result([dict(label=f"ensemble of {p['n']} teachers", acc=ens_acc,
                        params=sum(t.n_params() for t in ts),
                        kbits=sum(storage_bits(t) for t in ts) / 1000,
                        ref=True),
                   row("student from teacher #0 alone", distill(STUDENT, ts[0], 0, steps=STEPS)),
                   row(f"student from ensemble ({p['combine']} averaged)", m)])


@method(id="born_again", name="Self-distillation (born-again networks)", family="small",
        category="distill",
        summary="A model teaches an identical copy of itself, sometimes for several generations.",
        code=["lab/experiments/teacher.py:born_again"],
        params=[P("width", "width", "int", 32, 4, 128, 4),
                P("generations", "generations", "int", 2, 1, 4, 1)],
        experiments=["born_again", "small_tricks"],
        papers=["Furlanello et al. Born-Again Neural Networks (2018)"])
def run_born_again(p):
    hid = W(p["width"])
    prev = scratch(hid, 0, steps=STEPS)
    rows = [row("generation 0 (labels)", prev)]
    for g in range(1, p["generations"] + 1):
        prev = distill(hid, prev, 100 * g, transfer="train", T=2, alpha=0.5, steps=STEPS)
        rows.append(row(f"generation {g}", prev))
    return result(rows, chart=dict(x="generation", series={"accuracy": [
        [i, r["acc"]] for i, r in enumerate(rows)]}),
        bird_reason="Born-again networks retrain a copy of the same size: the aim is a "
                    "better model, not a smaller one.")


@method(id="bayes_teacher", name="The perfect teacher", family="small", category="distill",
        summary="Distill from the exact Bayes posterior. An upper bound on what any teacher's "
                "soft labels can give a student of a given size.",
        code=["lab/world.py:World._posterior"],
        params=[P("width", "student width", "int", 16, 2, 128, 2)],
        experiments=["oracle_teacher"],
        papers=["Menon et al. A Statistical Perspective on Distillation (2021)"])
def run_bayes(p):
    d, t = data(), teacher()
    X, _ = transfer_X(d)
    soft = np.log(np.concatenate([d.train.bayes, d.pool.bayes]) + 1e-8)
    hid = W(p["width"])
    oracle = train(net(hid, 0), FixedSource(X, None, d.world.C, t_logits=soft), [KD(T=1)],
                   steps=STEPS)
    return result([dict(label="Bayes-optimal (ceiling)", acc=float(
                       (d.test.bayes.argmax(1) == d.test.y).mean()), model=False),
                   teacher_row(),
                   row("labels only", scratch(hid, 0, steps=STEPS)),
                   row("KD from trained teacher", distill(hid, t, 0, steps=STEPS)),
                   row("KD from exact posterior", oracle)])


# ===========================================================================
# SMALL NETWORKS -- pruning
# ===========================================================================
@method(id="magnitude_pruning", name="Magnitude pruning", family="small", category="prune",
        summary="Set the smallest weights to zero. Cheap and surprisingly effective, as long "
                "as you let the network recover afterwards.",
        code=["lab/compress.py:magnitude_prune", "lab/nn.py:MLP.apply_masks"],
        params=[P("sparsity", "fraction removed", "float", 0.9, 0.1, 0.995, 0.005),
                P("recovery", "recovery", "choice", "kd", choices=["none", "labels", "kd"]),
                P("steps", "gradual steps", "int", 1, 1, 10, 1,
                  help="1 = one-shot; more = prune a bit, recover, repeat"),
                P("scope", "ranking", "choice", "global", choices=["global", "per-layer"])],
        experiments=["prune_unstructured"],
        papers=["Han et al. Learning both Weights and Connections for Efficient Neural Networks (2015)",
                "Zhu & Gupta. To prune, or not to prune (2017)"])
def run_magnitude(p):
    t = teacher()
    sp, k = p["sparsity"], p["steps"]
    m = t
    for i in range(1, k + 1):
        m = magnitude_prune(m, 1 - (1 - sp) ** (i / k), global_=p["scope"] == "global")
        if p["recovery"] == "labels":
            m = scratch(m, i, steps=FT // k, lr=FT_LR)
        elif p["recovery"] == "kd":
            m = distill(m, t, i, steps=FT // k, lr=FT_LR)
    from .compress import sparsity
    return result([teacher_row(),
                   row("one-shot, no recovery", magnitude_prune(t, sp, global_=p["scope"] == "global")),
                   row(f"{k}-step, recovery: {p['recovery']}", m, sparsity=sparsity(m))])


@method(id="structured_pruning", name="Structured (neuron) pruning", family="small",
        category="prune",
        summary="Delete whole neurons so the matrices actually get smaller and faster. "
                "The choice of which neurons to delete is the whole game.",
        code=["lab/compress.py:neuron_importance", "lab/compress.py:structured_prune",
              "lab/nn.py:MLP.keep_neurons"],
        params=[P("width", "keep width (of 256)", "int", 32, 4, 192, 4),
                P("criterion", "importance", "choice", "taylor",
                  choices=["l1", "act", "taylor", "random"]),
                P("recovery", "recovery", "choice", "kd", choices=["none", "labels", "kd"])],
        experiments=["prune_structured"],
        papers=["Li et al. Pruning Filters for Efficient ConvNets (2017)",
                "Molchanov et al. Importance Estimation for Neural Network Pruning (2019)"])
def run_structured(p):
    d, t = data(), teacher()
    probe = d.train.take(2000)
    sc = neuron_importance(t, probe.X, probe.y, p["criterion"], rng=np.random.default_rng(0))
    w = p["width"]
    cut = structured_prune(t, [w] * 3, sc)
    rows = [teacher_row(), row(f"pruned to 3x{w}, no recovery", cut)]
    if p["recovery"] == "labels":
        rows.append(row("recovered with labels", scratch(cut, 0, steps=FT, lr=FT_LR)))
    elif p["recovery"] == "kd":
        rows.append(row("recovered with KD", distill(cut, t, 0, steps=FT, lr=FT_LR)))
    rows.append(row(f"fresh 3x{w}, KD from scratch", distill([w] * 3, t, 0, steps=STEPS)))
    return result(rows)


@method(id="lottery_ticket", name="Lottery tickets", family="small", category="prune",
        summary="Train, prune, rewind the surviving weights to their early values, retrain. "
                "Tests whether the sparse network was trainable from the start.",
        code=["lab/experiments/shrink.py:lottery_ticket"],
        params=[P("rounds", "prune rounds (50% each)", "int", 3, 1, 6, 1),
                P("rewind", "rewind to", "choice", "early", choices=["init", "early", "random reinit"])],
        experiments=["lottery_ticket"],
        papers=["Frankle & Carbin. The Lottery Ticket Hypothesis (2019)",
                "Frankle et al. Linear Mode Connectivity and the Lottery Ticket Hypothesis (2020)"])
def run_lottery(p):
    hid = [64, 64]
    init = net(hid, seed=0)
    snap: dict = {}
    trained = scratch(init.copy(), 0, steps=STEPS,
                      callback=lambda ep, n: snap.setdefault("early", n.copy()) if ep == 1 else None)
    start = {"init": init, "early": snap.get("early", init), "random reinit": net(hid, seed=99)}[p["rewind"]]
    m = trained
    rows = [row("dense 2x64", trained, ref=True)]
    for r in range(1, p["rounds"] + 1):
        mask_src = magnitude_prune(m, 1 - 0.5 ** r)
        m = start.copy()
        m.masks = [None if x is None else x.copy() for x in mask_src.masks]
        m.apply_masks()
        m = scratch(m, 0, steps=STEPS)
        rows.append(row(f"round {r}: {100 * (1 - 0.5 ** r):.1f}% removed", m))
    return result(rows, chart=dict(x="sparsity", series={f"rewind to {p['rewind']}": [
        [1 - 0.5 ** i, r["acc"]] for i, r in enumerate(rows)]}))


@method(id="layer_dropping", name="Layer dropping", family="small", category="depth",
        summary="Delete entire hidden layers. Fatal in a plain network, often survivable in "
                "a residual one.",
        code=["lab/experiments/shrink.py:_greedy_drop", "lab/nn.py:MLP.drop_layer"],
        params=[P("arch", "architecture", "choice", "residual", choices=["plain", "residual"]),
                P("k", "layers to delete (of 8)", "int", 2, 1, 6, 1),
                P("recovery", "recovery", "choice", "kd", choices=["none", "kd"])],
        experiments=["layer_drop"],
        papers=["Veit et al. Residual Networks Behave Like Ensembles of Relatively Shallow "
                "Networks (2016)", "Fan et al. Reducing Transformer Depth on Demand with "
                "Structured Dropout (2019)"])
def run_layer_drop(p):
    from .experiments.shrink import _greedy_drop
    d = data()
    t = teacher(hidden=[128] * 8, res=p["arch"] == "residual")
    probe = d.train.take(2000)
    cut = drop_layers(t, _greedy_drop(t, p["k"], probe.X, probe.y))
    rows = [row(f"{p['arch']} 8x128", t, ref=True), row(f"{p['k']} layers deleted", cut)]
    if p["recovery"] == "kd":
        rows.append(row("+ KD recovery", distill(cut, t, 0, steps=FT, lr=FT_LR)))
    return result(rows)


# ===========================================================================
# SMALL NETWORKS -- quantization & factorization
# ===========================================================================
@method(id="ptq", name="Post-training quantization", family="small", category="quantize",
        summary="Round every weight to one of 2^b − 1 evenly spaced levels, symmetric around "
                "zero (two levels at 1 bit), after training. No data, no gradients; how fine the "
                "scale groups are decides how low b can go.",
        code=["lab/nn.py:fake_quantize", "lab/compress.py:quantize", "cockpit/harness/quant.py:fake_quantize"],
        params=[P("bits", "bits", "int", 4, 1, 8, 1),
                P("group", "scale shared by", "choice", "16",
                  choices=["tensor", "64", "16", "4"]),
                P("model", "model", "choice", "teacher", choices=["teacher", "student"])],
        experiments=["quantization"],
        papers=["Jacob et al. Quantization and Training of Neural Networks for Efficient "
                "Integer-Arithmetic-Only Inference (2018)"])
def run_ptq(p):
    m0 = teacher() if p["model"] == "teacher" else kd_student()
    g = None if p["group"] == "tensor" else int(p["group"])
    rows = [row("fp32", m0, ref=True)]
    for b in [8, 6, 4, 3, 2, 1]:
        q = quantize(m0, b, g)
        rows.append(row(f"{b}-bit", q, bits=b, kbits=kbits(q, b, g)))
    q = quantize(m0, p["bits"], g)
    chosen = row(f"chosen: {p['bits']}-bit, group {p['group']}", q, kbits=kbits(q, p["bits"], g))
    return result(rows + [chosen], chart=dict(x="bits", series={f"group {p['group']}": [
        [r["bits"], r["acc"]] for r in rows if "bits" in r]}))


@method(id="qat", name="Quantization-aware training", family="small", category="quantize",
        summary="Keep training with quantized weights in the forward pass and full-precision "
                "weights in the backward pass (the straight-through estimator).",
        code=["lab/nn.py:MLP.eff", "lab/nn.py:MLP.backward"],
        params=[P("bits", "bits", "int", 2, 1, 4, 1),
                P("recovery", "train with", "choice", "kd", choices=["labels", "kd"]),
                P("model", "model", "choice", "student", choices=["teacher", "student"])],
        experiments=["quantization"],
        papers=["Bengio et al. Estimating or Propagating Gradients Through Stochastic Neurons (2013)",
                "Esser et al. Learned Step Size Quantization (2020)"])
def run_qat(p):
    t = teacher()
    m0 = t if p["model"] == "teacher" else kd_student()
    m = m0.copy()
    m.qbits, m.qgroup = p["bits"], 16
    m = (scratch(m, 0, steps=FT, lr=FT_LR) if p["recovery"] == "labels"
         else distill(m, t, 0, steps=FT, lr=FT_LR))
    q = quantize(m0, p["bits"], 16)
    return result([row("fp32", m0, ref=True),
                   row(f"PTQ {p['bits']}-bit", q, kbits=kbits(q, p["bits"], 16)),
                   row(f"QAT {p['bits']}-bit with {p['recovery']}", m)])


@method(id="weight_sharing", name="Weight sharing (k-means codebooks)", family="small",
        category="quantize",
        summary="Each layer keeps only k distinct values, chosen by k-means. Store a "
                "log2(k)-bit index per weight plus a tiny codebook.",
        code=["lab/compress.py:kmeans_share"],
        params=[P("k", "values per layer", "int", 4, 2, 64, 1),
                P("finetune", "fine-tune + re-snap", "bool", True)],
        experiments=["weight_sharing"],
        papers=["Han et al. Deep Compression (2016)"])
def run_weight_sharing(p):
    t = teacher()
    k = p["k"]
    bits = int(np.ceil(np.log2(k)))
    km = kmeans_share(t, k)
    q = quantize(t, bits)
    rows = [teacher_row(), row(f"uniform {bits}-bit (per tensor)", q, kbits=kbits(q, bits)),
            row(f"k-means, k={k}", km, kbits=kbits(km, bits, codebook=k))]
    if p["finetune"]:
        m = km
        for r in range(2):
            m = kmeans_share(distill(m, t, r, steps=FT // 2, lr=FT_LR), k)
        rows.append(row("k-means + KD + re-snap", m, kbits=kbits(m, bits, codebook=k)))
    return result(rows)


@method(id="low_rank", name="Low-rank factorization (SVD)", family="small",
        category="factorize",
        summary="Replace a big matrix by the product of two thin ones, keeping its top "
                "singular directions.",
        code=["lab/compress.py:low_rank"],
        params=[P("rank", "rank", "int", 16, 1, 128, 1),
                P("recovery", "recovery", "choice", "kd", choices=["none", "kd"])],
        experiments=["low_rank"],
        papers=["Denton et al. Exploiting Linear Structure Within Convolutional Networks (2014)",
                "Hsu et al. Language Model Compression with Weighted Low-Rank Factorization (2022)"])
def run_low_rank(p):
    t = teacher()
    m = low_rank(t, p["rank"])
    rows = [teacher_row(), row(f"rank {p['rank']}, no recovery", m)]
    if p["recovery"] == "kd":
        rows.append(row("+ KD recovery", distill(m, t, 0, steps=FT, lr=FT_LR)))
    return result(rows)


# ===========================================================================
# SMALL NETWORKS -- making small models better
# ===========================================================================
@method(id="overparameterize", name="Over-parameterize, then collapse", family="small",
        category="train",
        summary="Train each weight matrix as a product of two wider ones, then multiply them "
                "back together. Same network at inference, different training path.",
        code=["lab/experiments/small.py:overparam", "lab/nn.py:MLP.collapse_linear"],
        params=[P("width", "width", "int", 16, 4, 64, 4),
                P("expand", "expansion factor", "int", 4, 1, 8, 1)],
        experiments=["small_tricks"],
        papers=["Arora et al. On the Optimization of Deep Networks: Implicit Acceleration by "
                "Overparameterization (2018)", "Guo et al. ExpandNets (2020)",
                "Ding et al. RepVGG (2021)"])
def run_overparam(p):
    from .experiments.small import overparam
    hid = W(p["width"])
    big = overparam(hid, 0, expand=p["expand"])
    trained = scratch(big, 0, steps=STEPS)
    small = trained.collapse_linear()
    return result([row("plain", scratch(hid, 0, steps=STEPS)),
                   row(f"trained expanded ({trained.n_params(False):,} params)", trained),
                   row(f"collapsed back ({small.n_params(False):,} params)", small)],
                  note="The collapsed network is exactly the expanded one, multiplied out.",
                  bird_reason="Over-parameterizing changes only how the network is trained: "
                              "multiplied back together, it is the same size as the plain one.")


@method(id="small_tricks", name="Regularizers & training tricks", family="small",
        category="train",
        summary="Label smoothing, mixup, weight averaging (EMA), longer training: what each "
                "does for a capacity-starved model.",
        code=["lab/train.py:CE", "lab/train.py:fit"],
        params=[P("trick", "trick", "choice", "ema",
                  choices=["label smoothing", "mixup", "ema", "4x longer"]),
                P("width", "width", "int", 16, 4, 64, 4)],
        experiments=["small_tricks"],
        papers=["Szegedy et al. Rethinking the Inception Architecture (label smoothing) (2016)",
                "Zhang et al. mixup (2018)", "Izmailov et al. Averaging Weights Leads to Wider "
                "Optima (2018)"])
def run_tricks(p):
    d = data()
    hid = W(p["width"])
    tr = p["trick"]
    if tr == "label smoothing":
        m = scratch(hid, 0, steps=STEPS, smoothing=0.1)
    elif tr == "mixup":
        m = train(net(hid, 0), OnlineSource(d.train.X, d.train.y, d.world.C, None),
                  [CE()], steps=STEPS)
    elif tr == "ema":
        # average over roughly the last 10% of training; a 0.999 decay over a
        # 1,000-step run would mostly average the untrained early weights
        m = scratch(hid, 0, steps=STEPS, ema=1 - 10 / STEPS)
    else:
        m = scratch(hid, 0, steps=4 * STEPS)
    return result([row("baseline", scratch(hid, 0, steps=STEPS)), row(tr, m)],
                  bird_reason="These tricks change how a small network is trained, not its "
                              "size: both rows are networks of the same shape.")


@method(id="train_big_then_prune", name="Train big, then prune to size", family="small",
        category="train",
        summary="To get a small network, train a wide one and cut it down instead of "
                "training the small one directly.",
        code=["lab/experiments/small.py:train_big_then_compress"],
        params=[P("target", "final width", "int", 16, 4, 64, 4),
                P("start", "trained width", "int", 128, 8, 256, 8)],
        experiments=["train_big_then_compress", "small_tricks"],
        papers=["Li et al. Train Large, Then Compress (2020)"])
def run_big_then_prune(p):
    d = data()
    probe = d.train.take(2000)
    big = scratch(W(p["start"]), 0, steps=STEPS)
    cut = structured_prune(big, W(p["target"]), neuron_importance(big, probe.X, probe.y, "taylor"))
    return result([row(f"2x{p['target']} trained directly",
                       scratch(W(p["target"]), 0, steps=STEPS + FT)),
                   row(f"2x{p['start']} (before pruning)", big, ref=True),
                   row(f"pruned to 2x{p['target']} + self-KD", distill(cut, big, 0, steps=FT, lr=FT_LR))],
                  bird_reason="" if p["start"] > p["target"] else (
                      "The trained width is not larger than the final width, so there was "
                      "nothing to prune."))


# ===========================================================================
# SMALL NETWORKS -- specialization (the bird-detector question)
# ===========================================================================
def _spec():
    from .experiments import specialist as S
    return S


@method(id="specialist_head", name="Specialist heads", family="small", category="specialize",
        summary="Keep only the outputs the task needs (bird vs. rest), and decide how much of "
                "the teacher's 'what else could it be' to keep.",
        code=["lab/experiments/specialist.py:groups", "lab/experiments/specialist.py:spec_distill"],
        params=[P("width", "width", "int", 4, 2, 32, 2),
                P("head", "head", "choice", "supers",
                  choices=["generalist", "binary labels", "binary", "siblings", "supers"])],
        experiments=["specialist_size"],
        papers=["Hinton et al. (2015), 'specialist models'"])
def run_specialist_head(p):
    S = _spec()
    d, t = data(), teacher()
    G = S.groups(d.world)
    w, h = p["width"], p["head"]
    if h == "generalist":
        r = row("20-way generalist", distill(W(w), t, 0, steps=STEPS))
    elif h == "binary labels":
        yb = (d.train.y != 0).astype(np.int64)
        m = train(S.spec_student(w, 2, 0), FixedSource(d.train.X, yb, 2), [CE()], steps=STEPS)
        r = dict(label="binary, labels only", **S.spec_eval(m), kbits=kbits(m))
    else:
        Gm = G[h]
        m = S.spec_distill(S.spec_student(w, Gm.shape[1], 0), t, Gm, 0, steps=STEPS)
        r = dict(label=f"{h} head ({Gm.shape[1]} outputs), KD", **S.spec_eval(m), kbits=kbits(m))
    tr = teacher_row()
    return result([tr, r], metric="target_ap")


@method(id="task_aware_pruning", name="Task-aware pruning", family="small", category="specialize",
        summary="Score neurons by how much they matter for the one task you keep, prune the "
                "rest, collapse the head, recover by distilling only that task.",
        code=["lab/compress.py:neuron_importance", "lab/compress.py:collapse_head"],
        params=[P("width", "keep width (of 256)", "int", 16, 4, 128, 4),
                P("criterion", "importance for", "choice", "bird", choices=["all classes", "bird"])],
        experiments=["specialist_prune"],
        papers=["Molchanov et al. (2019)", "Kim et al. Task-Specific Pruning (various)"])
def run_task_prune(p):
    S = _spec()
    d, t = data(), teacher()
    G = S.groups(d.world)
    probe = d.train.take(2000)
    sc = neuron_importance(t, probe.X, probe.y, "taylor",
                           group=G["binary"] if p["criterion"] == "bird" else None)
    w = p["width"]
    cut = collapse_head(structured_prune(t, [w] * 3, sc), G["supers"])
    rec = S.spec_distill(cut.copy(), t, G["supers"], 0, steps=FT, lr=FT_LR)
    return result([teacher_row(),
                   dict(label=f"3x{w}, {p['criterion']} importance, no recovery", **S.spec_eval(cut),
                        kbits=kbits(cut)),
                   dict(label="+ grouped-KD recovery", **S.spec_eval(rec), kbits=kbits(rec))],
                  metric="target_ap")


@method(id="specialist_data", name="Recovery data for a specialist", family="small",
        category="specialize",
        summary="What the recovery set must contain, and whether images of the target alone are enough.",
        code=["lab/experiments/specialist.py:specialist_data"],
        params=[P("data", "recovery inputs", "choice", "birds only",
                  choices=["everything", "birds only", "birds + random others", "birds + noise"])],
        experiments=["specialist_data"],
        papers=[])
def run_spec_data(p):
    S = _spec()
    d, t = data(), teacher()
    G = S.groups(d.world)
    X_all, _ = transfer_X(d)
    y_all = np.concatenate([d.train.y_clean, d.pool.y_clean])
    rng = np.random.default_rng(0)
    bird = X_all[y_all == 0]
    others = X_all[y_all != 0][rng.permutation((y_all != 0).sum())[:len(bird)]]
    X = {"everything": X_all, "birds only": bird,
         "birds + random others": np.concatenate([bird, others]),
         "birds + noise": np.concatenate([bird, rng.standard_normal(bird.shape).astype(np.float32)])
         }[p["data"]]
    def fit_on(Xs):
        m = S.spec_distill(S.spec_student(16, G["supers"].shape[1], 0), t, G["supers"], 0,
                           X=Xs, steps=STEPS)
        return dict(**S.spec_eval(m), kbits=kbits(m))
    rows = [teacher_row()]
    if p["data"] != "everything":
        rows.append(dict(label=f"2x16 specialist on everything ({len(X_all)})", **fit_on(X_all)))
    rows.append(dict(label=f"2x16 specialist on '{p['data']}' ({len(X)})", **fit_on(X)))
    return result(rows, metric="target_ap")


# ===========================================================================
# LANGUAGE-MODEL METHODS (layer-wise; runnable on the MLP)
# ===========================================================================
def _calib(n: int = 512) -> np.ndarray:
    return data().train.X[:n]


def _lw_kbits(m: MLP, bits: int, group: int | None) -> float:
    """Storage of a net quantized by ``apply_layerwise(..., skip_last=True)``:
    every layer but the output one at ``bits``, one fp16 scale per output row
    (per row x ``group`` input columns with groups). GPTQ and AWQ store exactly
    what RTN stores: AWQ's per-channel scales fold into the layer before, and
    GPTQ's error feedback only changes which integers are stored. Not counted:
    with act-order and groups, a real GPTQ checkpoint also keeps each column's
    group index."""
    return kbits(m, bits, group, per_row=True, fp32_last=True)


def _lw_quant_rows(fn_name: str, bits: int, group, **kw):
    t = teacher()
    X = _calib()
    g = None if group == "row" else int(group)
    rows = [teacher_row()]
    q = lw.apply_layerwise(t, X, lambda W_, h: lw.rtn(W_, h, bits, g), skip_last=True)
    rows.append(row(f"RTN {bits}-bit", q, kbits=_lw_kbits(q, bits, g)))
    if fn_name != "rtn":
        fn = getattr(lw, fn_name)
        q = lw.apply_layerwise(t, X, lambda W_, h: fn(W_, h, bits=bits, group=g, **kw),
                               skip_last=True)
        rows.append(row(f"{fn_name.upper()} {bits}-bit", q, kbits=_lw_kbits(q, bits, g)))
    return rows


def _sparse_row(label: str, m: MLP) -> dict:
    """Wanda and SparseGPT zero weights in place instead of masking them, so the
    net doesn't declare itself sparse: store it sparse explicitly (the fp32
    weights that survive, plus a 1-bit-per-position bitmap)."""
    return row(label, m, kbits=kbits(m, sparse=True))


LW_QPARAMS = [P("bits", "bits", "int", 3, 2, 8, 1),
              P("group", "scale per", "choice", "row", choices=["row", "64", "16"])]


@method(id="rtn", name="Round-to-nearest (RTN)", family="llm", category="quantize",
        summary="The baseline every LLM quantization paper beats: round each weight to the "
                "nearest level, one scale per output row.",
        code=["lab/layerwise.py:quant_rows", "lab/layerwise.py:rtn"],
        params=LW_QPARAMS, experiments=[],
        papers=["Dettmers et al. LLM.int8() (2022)"])
def run_rtn(p):
    return result(_lw_quant_rows("rtn", p["bits"], p["group"]))


@method(id="gptq", name="GPTQ", family="llm", category="quantize",
        summary="Quantize one column at a time and push each rounding error onto the columns "
                "not yet quantized, using the inverse Hessian from calibration data.",
        code=["lab/layerwise.py:gptq", "lab/layerwise.py:_hessian_inverse_chol",
              "lab/layerwise.py:apply_layerwise"],
        params=LW_QPARAMS + [P("act_order", "largest activations first", "bool", False)],
        papers=["Frantar et al. GPTQ: Accurate Post-Training Quantization for Generative "
                "Pre-trained Transformers (2022)", "Hassibi & Stork. Optimal Brain Surgeon (1993)"])
def run_gptq(p):
    return result(_lw_quant_rows("gptq", p["bits"], p["group"], act_order=p["act_order"]))


@method(id="awq", name="AWQ (activation-aware weight quantization)", family="llm",
        category="quantize",
        summary="Scale up the weight columns that see large activations before rounding, so "
                "the weights that matter most get the finest resolution.",
        code=["lab/layerwise.py:awq"],
        params=LW_QPARAMS,
        papers=["Lin et al. AWQ: Activation-aware Weight Quantization for LLM Compression and "
                "Acceleration (2023)"])
def run_awq(p):
    return result(_lw_quant_rows("awq", p["bits"], p["group"]))


@method(id="smoothquant", name="SmoothQuant (weights + activations)", family="llm",
        category="quantize",
        summary="Activations with a few huge channels can't be quantized to 8 bits; move that "
                "difficulty into the weights with an exact per-channel rescaling.",
        code=["lab/layerwise.py:smoothquant", "lab/layerwise.py:inject_outliers",
              "lab/layerwise.py:forward_wa", "lab/layerwise.py:quant_act"],
        params=[P("abits", "activation bits", "int", 6, 4, 16, 1),
                P("wbits", "weight bits", "int", 8, 4, 16, 1,
                  help="16 = leave weights in float; 8 = the W8A8 setting"),
                P("alpha", "migration strength", "float", 0.5, 0, 1, 0.05),
                P("outliers", "inject outlier channels", "bool", True),
                P("factor", "outlier size", "float", 30, 2, 100, 1)],
        papers=["Xiao et al. SmoothQuant (2022)", "Dettmers et al. LLM.int8() (2022)"])
def run_smoothquant(p):
    d, t = data(), teacher()
    X = _calib()
    m = lw.inject_outliers(t, 1, 3, p["factor"]) if p["outliers"] else t
    sm = lw.smoothquant(m, X, p["alpha"])
    m_float = m
    if p["wbits"] < 16:  # quantize weights (per output row) after smoothing, as in W8A8
        wq = lambda W_, h: lw.rtn(W_, h, p["wbits"])  # noqa: E731
        m = lw.apply_layerwise(m, X, wq, skip_last=True)
        sm = lw.apply_layerwise(sm, X, wq, skip_last=True)

    def acc(model, ab):
        return float((lw.forward_wa(model, d.test.X, ab).argmax(1) == d.test.y).mean())
    # activations are rounded on the fly and never stored, so only the weight
    # format sets a row's size; smoothing folds into the weights for free
    wl = f"W{p['wbits']}" if p["wbits"] < 16 else "float weights"
    kb = _lw_kbits(m, p["wbits"], None) if p["wbits"] < 16 else kbits(m)
    rows = [dict(label=f"{wl}, float activations", acc=acc(m, 16), kbits=kb),
            dict(label=f"{wl}, A{p['abits']}", acc=acc(m, p["abits"]), kbits=kb),
            dict(label=f"SmoothQuant (alpha {p['alpha']:g}), {wl}, A{p['abits']}",
                 acc=acc(sm, p["abits"]), kbits=kb)]
    if p["wbits"] < 16:  # the float model, before any rounding, is the reference
        rows.insert(0, dict(label="float weights, float activations", acc=acc(m_float, 16),
                            kbits=kbits(m_float), ref=True))
    chart = dict(x="activation bits", series={
        "plain": [[b, acc(m, b)] for b in (4, 5, 6, 8, 16)],
        "SmoothQuant": [[b, acc(sm, b)] for b in (4, 5, 6, 8, 16)]})
    return result(rows, chart=chart, note=(
        "Injected outliers don't change the function (full precision is identical); "
        "they only change how quantizable the activations are." if p["outliers"] else ""),
        bird_reason="" if p["wbits"] < 16 else (
            "With float weights nothing stored gets smaller: activations are rounded while "
            "the model runs, and every weight keeps its 32 bits."))


@method(id="wanda", name="Wanda", family="llm", category="prune",
        summary="Prune by |weight| times the norm of its input activation, per output row. "
                "No retraining, no weight updates.",
        code=["lab/layerwise.py:wanda"],
        params=[P("sparsity", "fraction removed", "float", 0.5, 0.1, 0.95, 0.05),
                P("outliers", "outlier channels injected", "int", 0, 0, 128, 8,
                  help="same function, some channels 30x louder: magnitude pruning is fooled, "
                       "Wanda's score doesn't change at all")],
        papers=["Sun et al. A Simple and Effective Pruning Approach for Large Language Models (2023)"])
def run_wanda(p):
    t, X = teacher(), _calib()
    if p["outliers"]:
        t = lw.inject_outliers(t, 1, p["outliers"], 30.0)
    s = p["sparsity"]
    return result([teacher_row(),
                   _sparse_row("magnitude, per row", lw.apply_layerwise(t, X, lambda W_, h: lw.wanda(W_, np.ones_like(h), s), skip_last=True)),
                   _sparse_row("Wanda", lw.apply_layerwise(t, X, lambda W_, h: lw.wanda(W_, h, s), skip_last=True))])


@method(id="sparsegpt", name="SparseGPT", family="llm", category="prune",
        summary="Prune with GPTQ's error-compensation trick: every removed weight's error is "
                "absorbed by the weights that remain.",
        code=["lab/layerwise.py:sparsegpt", "lab/layerwise.py:_hessian_inverse_chol"],
        params=[P("sparsity", "fraction removed", "float", 0.7, 0.1, 0.95, 0.05),
                P("block", "block size", "int", 32, 4, 256, 4)],
        papers=["Frantar & Alistarh. SparseGPT: Massive Language Models Can Be Accurately "
                "Pruned in One-Shot (2023)"])
def run_sparsegpt(p):
    t, X = teacher(), _calib()
    s = p["sparsity"]
    return result([teacher_row(),
                   _sparse_row("magnitude, per row", lw.apply_layerwise(t, X, lambda W_, h: lw.wanda(W_, np.ones_like(h), s), skip_last=True)),
                   _sparse_row("Wanda", lw.apply_layerwise(t, X, lambda W_, h: lw.wanda(W_, h, s), skip_last=True)),
                   _sparse_row("SparseGPT", lw.apply_layerwise(t, X, lambda W_, h: lw.sparsegpt(W_, h, s, p["block"]), skip_last=True))])


@method(id="qlora", name="QLoRA (low-rank adapters on a quantized model)", family="llm",
        category="quantize",
        summary="Freeze the quantized model and train tiny low-rank corrections on top. "
                "Fine-tuning that costs a fraction of the memory.",
        code=["lab/layerwise.py:qlora"],
        params=[P("bits", "base bits", "int", 2, 2, 8, 1),
                P("rank", "adapter rank", "int", 4, 1, 12, 1,
                  help="past rank ~16 the adapters would outnumber the 2,132-parameter model")],
        papers=["Hu et al. LoRA (2021)", "Dettmers et al. QLoRA (2023)"])
def run_qlora(p):
    t, s = teacher(), kd_student()
    m, n = lw.qlora(s, t, data().pool.X, bits=p["bits"], rank=p["rank"], steps=FT)
    base = quantize(s, p["bits"], 16)
    base_bits = storage_bits(base, p["bits"], 16)
    # ``m`` has the adapters merged back in, which makes it fp32 again; what
    # QLoRA keeps is the frozen quantized base plus the adapters (fp32, as trained)
    return result([row("student 2x32, fp32", s, ref=True),
                   row(f"quantized to {p['bits']} bits", base, kbits=base_bits / 1000),
                   row(f"+ rank-{p['rank']} adapters ({n} trainable params)", m,
                       kbits=(base_bits + n * 32) / 1000)])


@method(id="block_influence", name="ShortGPT (block influence)", family="llm", category="depth",
        summary="Measure how much each residual layer changes its input (1 - cosine "
                "similarity); delete the layers that change it least.",
        code=["lab/layerwise.py:block_influence"],
        params=[P("k", "layers to delete (of 7)", "int", 2, 1, 6, 1)],
        experiments=["layer_drop"],
        papers=["Men et al. ShortGPT: Layers in Large Language Models are More Redundant Than "
                "You Expect (2024)", "Gromov et al. The Unreasonable Ineffectiveness of the "
                "Deeper Layers (2024)"])
def run_block_influence(p):
    from .experiments.shrink import _greedy_drop
    d = data()
    t = teacher(hidden=[128] * 8, res=True)
    X = _calib()
    bi = lw.block_influence(t, X)
    k = p["k"]
    by_bi = sorted(bi, key=bi.get)[:k]
    rng = np.random.default_rng(0)
    rand = list(rng.choice(sorted(bi), k, replace=False))
    probe = d.train.take(2000)
    return result([row("residual 8x128", t, ref=True),
                   row(f"drop {k} lowest-influence layers {sorted(by_bi)}", drop_layers(t, by_bi)),
                   row(f"drop {k} random layers {sorted(int(x) for x in rand)}", drop_layers(t, rand)),
                   row(f"drop {k} chosen by trial and error", drop_layers(t, _greedy_drop(t, k, probe.X, probe.y)))],
                  chart=dict(x="layer", series={"block influence": [[i, v] for i, v in bi.items()]}))


# ---------------------------------------------------------------------------
# reference-only and cockpit methods
# ---------------------------------------------------------------------------
reference(id="sequence_kd", name="Sequence-level distillation", family="llm", category="distill",
          summary="Train the student on whole sequences the teacher generates (beam search "
                  "output), not on per-token probabilities.",
          code=["lab/train.py:KD"],
          papers=["Kim & Rush. Sequence-Level Knowledge Distillation (2016)",
                  "Agarwal et al. On-Policy Distillation of Language Models (GKD) (2024)"])
reference(id="kv_cache_quant", name="KV-cache quantization", family="llm", category="quantize",
          summary="Quantize the attention keys and values stored during generation, which "
                  "dominate memory at long context.",
          code=["lab/layerwise.py:quant_act"],
          papers=["Hooper et al. KVQuant (2024)", "Liu et al. KIVI (2024)"])
reference(id="speculative_decoding", name="Speculative decoding", family="llm",
          category="train",
          summary="A small draft model proposes several tokens; the big model verifies them in "
                  "one pass. A use for small models that loses no quality.",
          papers=["Leviathan et al. Fast Inference from Transformers via Speculative Decoding (2023)",
                  "Chen et al. Accelerating LLM Decoding with Speculative Sampling (2023)"])
reference(id="moe_expert_pruning", name="Expert pruning in mixture-of-experts", family="llm",
          category="prune",
          summary="Drop or merge the experts a router rarely picks for your task: structured "
                  "pruning where the 'neurons' are whole sub-networks.",
          code=["lab/compress.py:neuron_importance"],
          papers=["Lu et al. Not All Experts are Equal (2024)"])
reference(id="vocab_trimming", name="Vocabulary trimming", family="llm", category="specialize",
          summary="Delete embedding and output rows for tokens your task never uses. The "
                  "LLM version of collapsing a detector's 80-class head to one class.",
          code=["lab/compress.py:collapse_head"],
          papers=["Ushio et al. An Efficient Multilingual Language Model Compression through "
                  "Vocabulary Trimming (2023)"])

reference(id="channel_pruning", name="Channel pruning with dependency groups", family="vision",
          category="prune", status="cockpit",
          summary="In a CNN with skip connections and concatenations, a channel can't be "
                  "removed from one layer alone; every layer that shares it must lose it too.",
          code=["cockpit/harness/prune.py:structured_prune", "cockpit/harness/prune.py:structured_prune_per_stage",
                "cockpit/harness/prune.py:C2f_v2"],
          experiments=["prune_structured"],
          papers=["Fang et al. DepGraph: Towards Any Structural Pruning (2023)"])
reference(id="conv_quantization", name="Quantizing a detector", family="vision",
          category="quantize", status="cockpit",
          summary="The same fake-quantization dial applied to YOLOv8's convolutions, with "
                  "per-group scales and an optional spared detection head.",
          code=["cockpit/harness/quant.py:fake_quantize", "cockpit/harness/model.py:apply_uniform",
                "cockpit/harness/model.py:fuse"],
          experiments=["quantization"],
          papers=["Nagel et al. A White Paper on Neural Network Quantization (2021)"])
reference(id="prune_and_regrow", name="Prune, then fine-tune (cut and regrow)", family="vision",
          category="prune", status="cockpit",
          summary="The cockpit's core loop: remove channels, then fine-tune the survivor on bird "
                  "images until mAP climbs back.",
          code=["cockpit/harness/prune.py:removal_plan", "cockpit/harness/mask.py:apply_plan", "cockpit/recover_job.py:main"],
          experiments=["prune_structured", "specialist_data"],
          papers=["Han et al. (2015)", "Liu et al. Rethinking the Value of Network Pruning (2019)"])
reference(id="deepinversion", name="DeepInversion (data-free via batch-norm statistics)",
          family="vision", category="distill",
          summary="Synthesize images whose feature statistics match the teacher's stored "
                  "batch-norm means and variances, then distill on them.",
          code=["lab/experiments/transfer.py:invert_teacher"],
          experiments=["transfer_set"],
          papers=["Yin et al. Dreaming to Distill (2020)"])
reference(id="attention_transfer", name="Attention transfer", family="vision", category="distill",
          summary="Match where the teacher's feature maps are active (spatial attention maps), "
                  "a feature-distillation loss that needs spatial structure.",
          code=["lab/train.py:Hint"],
          experiments=["loss_zoo"],
          papers=["Zagoruyko & Komodakis. Paying More Attention to Attention (2017)"])
reference(id="detector_kd", name="Distilling a detector", family="vision", category="distill",
          summary="Detection outputs are boxes, not one distribution: distill the per-anchor "
                  "class scores, the box regressions and the neck features.",
          papers=["Chen et al. Learning Efficient Object Detection Models with Knowledge "
                  "Distillation (2017)", "Yang et al. Focal and Global Knowledge Distillation "
                  "for Detectors (2022)"])
reference(id="bn_folding", name="Batch-norm folding", family="vision", category="factorize",
          summary="Merge a batch-norm layer into the convolution before it: exact, free, and "
                  "always the first step before quantizing a CNN.",
          code=["lab/nn.py:MLP.collapse_linear"],
          papers=["Jacob et al. (2018)"])

