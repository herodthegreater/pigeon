"""Every run is bird-ready: the site draws one bird per model a run produces,
its species from how much smaller the model is than the run's reference
(``kbits``), its health from how much of the reference's accuracy it keeps.

So every runnable method's result must either say it has no bird (``bird``
False, with a reason a reader can understand), or have exactly one reference
row, a positive ``kbits`` on every model row, and no model bigger than the
reference. ``test_methods`` applies the same check to every run at both ends
of every parameter range; this file runs the defaults plus the settings that
change which rows a result has, and checks the storage model itself."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lab.common import net, teacher
from lab.compress import kbits, magnitude_prune, quantize, storage_bits
from lab.methods import REGISTRY

RUNNABLE = [m for m in REGISTRY.values() if m.status == "runnable"]

# Runs whose result is allowed a model bigger than the reference, and why.
GROWS = {
    # QLoRA keeps the adapters next to the quantized base, in fp32 as trained.
    # At high rank they outweigh what quantizing saved: at rank 12 the 2x32
    # student carries 1,920 adapter weights against its own 2,048, so the
    # base plus adapters (70.3 kbit at 2 bits) is bigger than the fp32
    # student (68.2 kbit). That is the method's real cost, not a bug.
    "qlora": lambda r: "adapters" in r["label"],
}

# Settings that change which rows a result has, or whether it has a bird.
VARIANTS = [
    ("smoothquant", {"wbits": 16}),          # float weights: nothing stored shrinks
    ("smoothquant", {"wbits": 4, "outliers": False}),
    ("train_big_then_prune", {"start": 8}),  # trained width below the final one
    ("qlora", {"rank": 12}),                 # adapters outgrow the savings (see GROWS)
    ("qlora", {"bits": 8, "rank": 1}),
    ("ptq", {"group": "tensor", "model": "student"}),
    ("weight_sharing", {"k": 2, "finetune": False}),
    ("rtn", {"group": "16", "bits": 2}),
    ("gptq", {"group": "64", "act_order": True}),
    ("low_rank", {"rank": 128}),             # too high a rank to factor anything: same size
    ("ensemble_kd", {"n": 1}),
    ("specialist_head", {"head": "generalist"}),
    ("specialist_head", {"head": "binary labels"}),
    ("specialist_data", {"data": "everything"}),
    ("magnitude_pruning", {"sparsity": 0.1, "recovery": "none"}),
    ("lottery_ticket", {"rounds": 1, "rewind": "random reinit"}),
    ("layer_dropping", {"arch": "plain", "recovery": "none"}),
    ("qat", {"model": "teacher", "recovery": "labels"}),
]


def check_birds(mid: str, res: dict) -> None:
    rows = res["rows"]
    assert isinstance(res.get("bird"), bool), f"{mid}: result has no bird flag"
    models = [r for r in rows if r.get("model", True)]
    assert models, f"{mid}: no model rows"
    for r in rows:
        assert r.get("model", True) in (True, False), (mid, r["label"])
        if not r.get("model", True):
            assert "kbits" not in r and not r.get("ref"), (mid, r["label"])
    for r in models:
        kb = r.get("kbits")
        assert isinstance(kb, float) and math.isfinite(kb) and kb > 0, (mid, r["label"], kb)
    if not res["bird"]:
        reason = res.get("bird_reason", "")
        assert isinstance(reason, str) and len(reason) > 20 and reason.endswith("."), (mid, reason)
        assert reason[0].isupper() and "\u2014" not in reason and " -- " not in reason, reason
        return
    assert "bird_reason" not in res, mid
    refs = [r for r in rows if r.get("ref")]
    assert len(refs) == 1, f"{mid}: {len(refs)} reference rows"
    ref = refs[0]
    assert ref.get("ref") is True, mid
    for r in models:
        assert isinstance(r.get(res["metric"]), float), (mid, r["label"], res["metric"])
    allowed = GROWS.get(mid, lambda r: False)
    bigger = [r["label"] for r in models if r["kbits"] > ref["kbits"] and not allowed(r)]
    assert not bigger, f"{mid}: bigger than the reference ({ref['kbits']} kbit): {bigger}"


@pytest.mark.parametrize("m", RUNNABLE, ids=lambda m: m.id)
def test_defaults_are_bird_ready(m):
    check_birds(m.id, m.execute({}))


@pytest.mark.parametrize("mid,params", VARIANTS, ids=lambda v: str(v))
def test_variants_are_bird_ready(mid, params):
    res = REGISTRY[mid].execute(params)
    check_birds(mid, res)
    if mid == "smoothquant":
        assert res["bird"] == (params["wbits"] < 16)
    if mid == "train_big_then_prune":
        assert not res["bird"]


def test_no_bird_for_same_size_methods():
    for mid in ("born_again", "overparameterize", "small_tricks"):
        res = REGISTRY[mid].execute({})
        assert res["bird"] is False and res["bird_reason"], mid


def test_bayes_ceiling_is_not_a_model():
    rows = REGISTRY["bayes_teacher"].execute({})["rows"]
    assert [r["label"] for r in rows if r.get("model") is False] == ["Bayes-optimal (ceiling)"]
    assert [r["label"] for r in rows if r.get("ref")] == ["teacher 3x256 (reference)"]


def test_ensemble_reference_is_every_teacher():
    res = REGISTRY["ensemble_kd"].execute({"n": 3})
    ref = res["rows"][0]
    assert ref["ref"] and ref["kbits"] == pytest.approx(3 * kbits(teacher()))
    assert ref["params"] == 3 * teacher().n_params()


# ---------------------------------------------------------------------------
# the storage model
# ---------------------------------------------------------------------------
def test_fp32_storage_is_32_bits_per_parameter():
    t = teacher()
    assert storage_bits(t) == 32 * t.n_params(False)
    assert kbits(t) == storage_bits(t) / 1000


def test_quantized_storage_ignores_weights_rounded_to_zero():
    t = teacher()
    q = quantize(t, 2, 16)
    assert q.n_params() < t.n_params()        # rounding made zeros...
    assert storage_bits(q, 2, 16) == storage_bits(t, 2, 16)   # ...that are still stored
    w = sum(x.size for x in t.W)
    b = sum(x.size for x in t.b)
    assert storage_bits(t, 2, 16) == 2 * w + 32 * b + 16 * w // 16
    assert storage_bits(t, 4) == 4 * w + 32 * b + 16 * t.n_layers   # one scale per tensor


def test_sparse_storage_counts_survivors_and_a_bitmap():
    t = teacher()
    p = magnitude_prune(t, 0.9)
    w = sum(x.size for x in t.W)
    nnz = sum(int((p.eff(i) != 0).sum()) for i in range(p.n_layers))
    assert storage_bits(p, sparse=True) == 32 * nnz + w + 32 * sum(x.size for x in t.b)
    assert kbits(p) == storage_bits(p, sparse=True) / 1000   # masks declare it sparse


def test_qat_net_declares_its_format():
    m = net([32, 32])
    m.qbits, m.qgroup = 3, 16
    assert kbits(m) == storage_bits(m, 3, 16) / 1000


def test_layerwise_and_codebook_storage():
    t = teacher()
    *hidden, out = range(t.n_layers)
    bias = 32 * sum(x.size for x in t.b)
    w_hidden = sum(t.W[i].size for i in hidden)
    rows = sum(t.W[i].shape[1] for i in hidden)
    # per output row, the output layer left in fp32
    assert storage_bits(t, 3, per_row=True, fp32_last=True) == (
        3 * w_hidden + 16 * rows + 32 * t.W[out].size + bias)
    # per row x 16 input columns: the 12-input first layer still needs one group per row
    groups = sum(t.W[i].shape[1] * -(-t.W[i].shape[0] // 16) for i in hidden)
    assert storage_bits(t, 3, 16, per_row=True, fp32_last=True) == (
        3 * w_hidden + 16 * groups + 32 * t.W[out].size + bias)
    # k-means: log2(k) bits per weight and a k-entry fp32 codebook per layer
    w = sum(x.size for x in t.W)
    assert storage_bits(t, 3, codebook=8) == 3 * w + 8 * 32 * t.n_layers + bias


def test_ptq_default_sizes():
    """A spot check with numbers: the 3x256 teacher is 140,052 fp32 parameters
    (4,481.664 kbit); at 4 bits with a scale per 16 weights it is 721.536 kbit,
    6.2 times smaller, because biases stay fp32 and scales cost 1 bit/weight."""
    res = REGISTRY["ptq"].execute({})
    ref = next(r for r in res["rows"] if r.get("ref"))
    four = next(r for r in res["rows"] if r["label"] == "4-bit")
    assert ref["label"] == "fp32" and ref["kbits"] == 4481.664
    assert four["kbits"] == 721.536
    assert np.isclose(ref["kbits"] / four["kbits"], 6.211, atol=1e-3)
