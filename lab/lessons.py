"""Build-it-yourself lessons: write each algorithm from scratch, then let the
checker run tests against your code.

Each lesson is self-contained numpy. Your code is saved as ``student.py`` in a
temporary directory and the lesson's tests run against it in a separate Python
process with a time limit (see ``check``). The site's "Check my code" button
and ``lab/tests/test_lessons.py`` both go through ``check``, and the tests
assert that every reference solution passes and every untouched skeleton fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Lesson:
    id: str
    title: str
    intro: str            # markdown (same subset as method pages)
    skeleton: str         # what the learner starts from
    solution: str         # reference implementation
    tests: str            # pytest-style test functions using `student`
    hints: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)   # related method pages


LESSONS: list[Lesson] = []


def lesson(**kw):
    for k in ("intro", "skeleton", "solution", "tests"):
        kw[k] = textwrap.dedent(kw[k]).strip() + "\n"
    LESSONS.append(Lesson(**kw))


def get(lesson_id: str) -> Lesson:
    for L in LESSONS:
        if L.id == lesson_id:
            return L
    raise KeyError(lesson_id)


def run_tests(tests_src: str, code: str) -> dict:
    """Import ``code`` as a fresh module named ``student`` and run every
    ``test_*`` function in ``tests_src`` against it. Returns the same shape as
    ``check``. Runs in the current process: ``check`` wraps it in a subprocess
    with a time limit; the static website calls it inside a Web Worker."""
    import traceback
    import types

    import numpy as np
    result: dict = {"tests": []}
    student = types.ModuleType("student")
    student.__file__ = "student.py"
    prev = sys.modules.get("student")
    sys.modules["student"] = student
    try:
        try:
            exec(compile(code, "student.py", "exec"), student.__dict__)
        except Exception:
            result["error"] = "Your code failed to import:\n" + traceback.format_exc(limit=3)
            return result
        ns = {"student": student, "np": np}
        exec(compile(tests_src, "lesson_tests", "exec"), ns)
        for name, fn in list(ns.items()):
            if not (name.startswith("test_") and callable(fn)):
                continue
            doc = (fn.__doc__ or "").strip()
            try:
                fn()
                result["tests"].append({"name": name, "doc": doc, "ok": True})
            except NotImplementedError:
                result["tests"].append({"name": name, "doc": doc, "ok": False,
                                        "message": "not implemented yet"})
            except AssertionError as e:
                result["tests"].append({"name": name, "doc": doc, "ok": False,
                                        "message": str(e) or "assertion failed"})
            except Exception as e:
                result["tests"].append({"name": name, "doc": doc, "ok": False,
                                        "message": type(e).__name__ + ": " + str(e),
                                        "trace": traceback.format_exc(limit=-2)})
        return result
    finally:
        if prev is None:
            sys.modules.pop("student", None)
        else:
            sys.modules["student"] = prev


def check_inprocess(lesson_id: str, code: str) -> dict:
    out = run_tests(get(lesson_id).tests, code)
    out["passed"] = "error" not in out and bool(out["tests"]) and all(t["ok"] for t in out["tests"])
    return out


_HARNESS = r'''
import json, sys
sys.path.insert(0, {root!r})
from lab.lessons import run_tests
print(json.dumps(run_tests({tests!r}, open("student.py").read())))
'''


def check(lesson_id: str, code: str, timeout: float = 60.0) -> dict:
    """Run the lesson's tests against ``code``. Returns
    ``{"passed": bool, "tests": [{name, doc, ok, message?}], "error"?: str}``.

    This executes the submitted code, so the web server only listens on
    localhost: it's your code running on your machine."""
    L = get(lesson_id)
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "student.py"), "w") as f:
            f.write(code)
        with open(os.path.join(d, "check.py"), "w") as f:
            f.write(_HARNESS.format(root=ROOT, tests=L.tests))
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        try:
            p = subprocess.run([sys.executable, "check.py"], cwd=d, capture_output=True,
                               text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return {"passed": False, "tests": [], "error": f"timed out after {timeout:.0f}s"}
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"passed": False, "tests": [], "error": (p.stderr or p.stdout)[-2000:]}
    out["passed"] = "error" not in out and bool(out["tests"]) and all(t["ok"] for t in out["tests"])
    return out


# ===========================================================================
# 1. softmax + cross-entropy
# ===========================================================================
lesson(
    id="softmax_ce", title="1. Softmax, cross-entropy and its gradient",
    methods=["kd"],
    intro="""
    Every classifier in this lab ends the same way: a vector of **logits** `z`
    (one number per class) becomes probabilities with **softmax**, and the
    loss is **cross-entropy** against the true class.

    `softmax(z)_i = exp(z_i) / sum_j exp(z_j)`

    Dividing the logits by a **temperature** `T` before the softmax flattens
    the distribution when `T > 1`. Distillation depends on that knob.

    Two things to get right:

    1. **Numerical stability.** `exp(1000)` overflows. Subtracting the row's
       maximum first changes nothing mathematically (it cancels in the ratio)
       and keeps every exponent at or below 0.
    2. **The gradient.** For `L = -log softmax(z)[y]`, the derivative with
       respect to the logits is beautifully simple: `p - onehot(y)`. Averaged
       over a batch of N rows, divide by N.

    Implement the three functions. All inputs are 2-D: one row per example.
    """,
    skeleton="""
    import numpy as np


    def softmax(z, T=1.0):
        \"\"\"Row-wise softmax of z / T. z has shape (N, C).\"\"\"
        raise NotImplementedError


    def cross_entropy(z, y):
        \"\"\"Mean of -log softmax(z)[i, y[i]] over the N rows.\"\"\"
        raise NotImplementedError


    def cross_entropy_grad(z, y):
        \"\"\"d cross_entropy / d z, shape (N, C).\"\"\"
        raise NotImplementedError
    """,
    solution="""
    import numpy as np


    def softmax(z, T=1.0):
        z = z / T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


    def cross_entropy(z, y):
        p = softmax(z)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


    def cross_entropy_grad(z, y):
        p = softmax(z)
        p[np.arange(len(y)), y] -= 1
        return p / len(y)
    """,
    tests="""
    def test_rows_sum_to_one():
        "softmax rows are probability distributions"
        z = np.random.default_rng(0).standard_normal((5, 7))
        p = student.softmax(z)
        assert p.shape == (5, 7), f"shape {p.shape}"
        assert np.allclose(p.sum(1), 1), "rows don't sum to 1"
        assert (p > 0).all()

    def test_stable_for_huge_logits():
        "no overflow when logits are ~1000"
        p = student.softmax(np.array([[1000.0, 1001.0, 999.0]]))
        assert np.isfinite(p).all(), "got inf/nan: subtract the row max first"
        assert np.allclose(p, student.softmax(np.array([[0.0, 1.0, -1.0]])))

    def test_temperature_flattens():
        "T > 1 makes the distribution flatter, T < 1 sharper"
        z = np.array([[3.0, 1.0, 0.0]])
        assert student.softmax(z, T=4).max() < student.softmax(z).max() < student.softmax(z, T=0.5).max()

    def test_cross_entropy_value():
        "matches -log p[y]"
        z = np.array([[2.0, 0.0], [0.0, 2.0]]); y = np.array([0, 0])
        want = (-np.log(np.exp(2) / (np.exp(2) + 1)) - np.log(1 / (np.exp(2) + 1))) / 2
        assert abs(student.cross_entropy(z, y) - want) < 1e-6

    def test_gradient_matches_finite_differences():
        "cross_entropy_grad agrees with a numerical derivative"
        rng = np.random.default_rng(1)
        z = rng.standard_normal((4, 5)); y = rng.integers(0, 5, 4)
        g = student.cross_entropy_grad(z.copy(), y)
        num = np.zeros_like(z)
        for i in range(4):
            for j in range(5):
                e = np.zeros_like(z); e[i, j] = 1e-6
                num[i, j] = (student.cross_entropy(z + e, y) - student.cross_entropy(z - e, y)) / 2e-6
        assert np.abs(g - num).max() < 1e-5, f"max error {np.abs(g - num).max():.2e}"
    """,
    hints=["`z.max(axis=1, keepdims=True)` keeps the shape (N, 1) so it broadcasts.",
           "For the gradient, start from `softmax(z)` and subtract 1 at each row's true class, "
           "then divide by N because the loss is a mean."],
)

# ===========================================================================
# 2. a dense layer, forward and backward
# ===========================================================================
lesson(
    id="dense_layer", title="2. A dense layer: forward and backward",
    methods=["kd", "magnitude_pruning"],
    intro="""
    A dense (fully connected) layer with a ReLU computes `a = relu(h @ W + b)`
    for a batch `h` of shape (N, in), weights `W` of shape (in, out) and bias
    `b` of shape (out,).

    Backpropagation runs the chain rule backwards. Given `g = dL/da` (same
    shape as `a`), you need three things:

    1. `gz = g * (z > 0)`, where `z = h @ W + b`. ReLU passes gradient only where it was active.
    2. `dW = h.T @ gz` and `db = gz.sum(axis=0)`: how the loss changes with each parameter.
    3. `dh = gz @ W.T`: the gradient to hand to the layer below.

    Every network in `lab/nn.py` is a stack of these. `MLP.backward` is this
    function in a loop.
    """,
    skeleton="""
    import numpy as np


    def dense_forward(h, W, b):
        \"\"\"Return (a, cache): a = relu(h @ W + b); cache = whatever backward needs.\"\"\"
        raise NotImplementedError


    def dense_backward(g, cache):
        \"\"\"Given g = dL/da, return (dh, dW, db).\"\"\"
        raise NotImplementedError
    """,
    solution="""
    import numpy as np


    def dense_forward(h, W, b):
        z = h @ W + b
        return np.maximum(z, 0), (h, W, z)


    def dense_backward(g, cache):
        h, W, z = cache
        gz = g * (z > 0)
        return gz @ W.T, h.T @ gz, gz.sum(axis=0)
    """,
    tests="""
    rng = np.random.default_rng(0)
    h = rng.standard_normal((6, 4)); W = rng.standard_normal((4, 3)); b = rng.standard_normal(3)
    R = rng.standard_normal((6, 3))   # a fixed random "upstream" so L = sum(a * R)

    def loss(h_, W_, b_):
        a, _ = student.dense_forward(h_, W_, b_)
        return float((a * R).sum())

    def numgrad(f, x):
        out = np.zeros_like(x)
        it = np.nditer(x, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = x[i]; x[i] = old + 1e-6; lp = f(); x[i] = old - 1e-6; lm = f(); x[i] = old
            out[i] = (lp - lm) / 2e-6
        return out

    def test_forward():
        "forward is relu(h @ W + b)"
        a, _ = student.dense_forward(h, W, b)
        assert np.allclose(a, np.maximum(h @ W + b, 0))

    def test_backward_weights():
        "dW and db match finite differences"
        a, cache = student.dense_forward(h, W, b)
        dh, dW, db = student.dense_backward(R, cache)
        assert np.abs(dW - numgrad(lambda: loss(h, W, b), W)).max() < 1e-5, "dW is wrong"
        assert np.abs(db - numgrad(lambda: loss(h, W, b), b)).max() < 1e-5, "db is wrong"

    def test_backward_input():
        "dh matches finite differences"
        a, cache = student.dense_forward(h, W, b)
        dh, _, _ = student.dense_backward(R, cache)
        assert dh.shape == h.shape, f"dh shape {dh.shape}"
        assert np.abs(dh - numgrad(lambda: loss(h, W, b), h)).max() < 1e-5, "dh is wrong"
    """,
    hints=["Keep `z` (before the ReLU) in the cache; the ReLU mask is `z > 0`.",
           "Check shapes: dW must be (in, out) like W, so it's `h.T @ gz`."],
)

# ===========================================================================
# 3. a full training loop
# ===========================================================================
lesson(
    id="train_mlp", title="3. Train a two-layer network from scratch",
    methods=["small_tricks"],
    intro="""
    Put lessons 1 and 2 together: a network `x -> relu(x W1 + b1) -> logits = h W2 + b2`
    trained with plain mini-batch gradient descent on a two-spirals dataset
    (provided). No libraries beyond numpy.

    The loop is the same one `lab/train.py:fit` runs, minus Adam and extras:

    1. Shuffle, take a batch.
    2. Forward: hidden activations, logits.
    3. `dz = (softmax(logits) - onehot(y)) / N`.
    4. Backward through the output layer, then the ReLU layer.
    5. `param -= lr * grad` for all four parameters.

    Implement `train(X, y, hidden, epochs, lr, seed)` returning a `predict(X)`
    function. The test requires 90% training accuracy on the spirals, which a
    linear model cannot get (about 50%), so your hidden layer has to work.
    """,
    skeleton="""
    import numpy as np


    def spirals(n=400, seed=0):
        rng = np.random.default_rng(seed)
        t = np.sqrt(rng.random(n)) * 3 * np.pi
        y = rng.integers(0, 2, n)
        r = t * (2 * y - 1)
        X = np.stack([r * np.cos(t), r * np.sin(t)], 1) / 10 + rng.normal(0, 0.05, (n, 2))
        return X, y


    def train(X, y, hidden=64, epochs=300, lr=0.5, seed=0):
        \"\"\"Train a 1-hidden-layer ReLU network with mini-batch SGD.
        Return a function predict(X) -> class indices.\"\"\"
        rng = np.random.default_rng(seed)
        C = int(y.max()) + 1
        W1 = rng.standard_normal((X.shape[1], hidden)) * np.sqrt(2 / X.shape[1])
        b1 = np.zeros(hidden)
        W2 = rng.standard_normal((hidden, C)) * np.sqrt(2 / hidden)
        b2 = np.zeros(C)
        raise NotImplementedError
    """,
    solution="""
    import numpy as np


    def spirals(n=400, seed=0):
        rng = np.random.default_rng(seed)
        t = np.sqrt(rng.random(n)) * 3 * np.pi
        y = rng.integers(0, 2, n)
        r = t * (2 * y - 1)
        X = np.stack([r * np.cos(t), r * np.sin(t)], 1) / 10 + rng.normal(0, 0.05, (n, 2))
        return X, y


    def train(X, y, hidden=64, epochs=300, lr=0.5, seed=0):
        rng = np.random.default_rng(seed)
        C = int(y.max()) + 1
        W1 = rng.standard_normal((X.shape[1], hidden)) * np.sqrt(2 / X.shape[1])
        b1 = np.zeros(hidden)
        W2 = rng.standard_normal((hidden, C)) * np.sqrt(2 / hidden)
        b2 = np.zeros(C)
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), 32):
                j = idx[s:s + 32]
                x, t = X[j], y[j]
                z1 = x @ W1 + b1
                h = np.maximum(z1, 0)
                z2 = h @ W2 + b2
                p = np.exp(z2 - z2.max(1, keepdims=True))
                p /= p.sum(1, keepdims=True)
                p[np.arange(len(t)), t] -= 1
                dz2 = p / len(t)
                dW2, db2 = h.T @ dz2, dz2.sum(0)
                dz1 = (dz2 @ W2.T) * (z1 > 0)
                dW1, db1 = x.T @ dz1, dz1.sum(0)
                W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2

        def predict(Xn):
            return (np.maximum(Xn @ W1 + b1, 0) @ W2 + b2).argmax(1)
        return predict
    """,
    tests="""
    def test_learns_spirals():
        "reaches 90% training accuracy on two spirals"
        X, y = student.spirals()
        predict = student.train(X, y)
        acc = (predict(X) == y).mean()
        assert acc >= 0.9, f"training accuracy {acc:.2f} < 0.90"

    def test_predict_shape():
        "predict returns one class index per row"
        X, y = student.spirals(n=50)
        predict = student.train(X, y, epochs=2)
        out = predict(X)
        assert out.shape == (50,), f"shape {out.shape}"
    """,
    hints=["Reuse your softmax and dense-layer code from lessons 1 and 2.",
           "If accuracy is stuck near 0.5, check the ReLU mask uses z1 (pre-activation), "
           "and that you update W1/b1, not only W2/b2."],
)

# ===========================================================================
# 4. Adam
# ===========================================================================
lesson(
    id="adam", title="4. The Adam optimizer",
    methods=["kd"],
    intro="""
    Plain gradient descent uses one learning rate for every parameter. **Adam**
    keeps two running averages per parameter, the gradient (`m`, momentum) and
    the squared gradient (`v`), and steps by `m / sqrt(v)`: large where the
    gradient is consistent, small where it is noisy or huge.

    For step `t = 1, 2, ...`:

    1. `m = b1 * m + (1 - b1) * g`
    2. `v = b2 * v + (1 - b2) * g * g`
    3. `m_hat = m / (1 - b1**t)`, `v_hat = v / (1 - b2**t)` (bias correction: both start at 0)
    4. `p -= lr * m_hat / (sqrt(v_hat) + eps)`

    Implement it as a class that updates arrays **in place**, like
    `lab/nn.py:Adam` (which also adds decoupled weight decay).
    """,
    skeleton="""
    import numpy as np


    class Adam:
        def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
            \"\"\"params: list of numpy arrays to update in place.\"\"\"
            raise NotImplementedError

        def step(self, grads):
            \"\"\"grads: list of arrays, same shapes as params.\"\"\"
            raise NotImplementedError
    """,
    solution="""
    import numpy as np


    class Adam:
        def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
            self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
            self.m = [np.zeros_like(x) for x in params]
            self.v = [np.zeros_like(x) for x in params]
            self.t = 0

        def step(self, grads):
            self.t += 1
            for p, g, m, v in zip(self.p, grads, self.m, self.v):
                m *= self.b1; m += (1 - self.b1) * g
                v *= self.b2; v += (1 - self.b2) * g * g
                mh = m / (1 - self.b1 ** self.t)
                vh = v / (1 - self.b2 ** self.t)
                p -= self.lr * mh / (np.sqrt(vh) + self.eps)
    """,
    tests="""
    def test_first_step_is_lr_times_sign():
        "after bias correction, the first step has size lr in every coordinate"
        p = np.array([1.0, -2.0, 3.0]); g = np.array([0.5, -10.0, 1e-3])
        opt = student.Adam([p], lr=0.1)
        opt.step([g])
        assert np.allclose(p, [0.9, -1.9, 2.9], atol=1e-6), f"got {p}"

    def test_updates_in_place():
        "the arrays passed in are modified, not replaced"
        p = np.ones(4); ref = p
        student.Adam([p], lr=0.01).step([np.ones(4)])
        assert ref is p and (p < 1).all()

    def test_minimizes_ill_conditioned_quadratic():
        "finds the minimum of 100 x^2 + 0.01 y^2 where plain SGD struggles"
        p = np.array([1.0, 1.0])
        opt = student.Adam([p], lr=0.05)
        for _ in range(1500):
            opt.step([np.array([200 * p[0], 0.02 * p[1]])])
        assert np.abs(p).max() < 0.05, f"ended at {p}"
    """,
    hints=["Store one `m` and one `v` array per parameter, initialized with `np.zeros_like`.",
           "Use in-place operations (`p -= ...`) so the caller's arrays change."],
)

# ===========================================================================
# 5. the distillation loss
# ===========================================================================
lesson(
    id="kd_loss", title="5. The distillation loss and its gradient",
    methods=["kd", "kd_losses"],
    intro="""
    Hinton's distillation loss compares the student's and teacher's softened
    distributions:

    `L = T^2 * mean_i KL(p_t || p_s)`, with `p_t = softmax(z_t / T)`, `p_s = softmax(z_s / T)`

    and `KL(p || q) = sum_c p_c (log p_c - log q_c)`.

    Why `T^2`? The gradient of the softened loss shrinks like `1/T^2` as `T`
    grows; multiplying back keeps its size comparable to the cross-entropy
    term when the two are mixed.

    The gradient with respect to the student's logits works out to

    `dL/dz_s = T * (p_s - p_t) / N`

    Derive it yourself first: it's the cross-entropy gradient from lesson 1
    with a soft target, with the chain rule through `z / T`.
    """,
    skeleton="""
    import numpy as np


    def kd_loss(zs, zt, T=4.0):
        \"\"\"Return (loss, dloss/dzs) for student logits zs and teacher logits zt, both (N, C).\"\"\"
        raise NotImplementedError
    """,
    solution="""
    import numpy as np


    def _log_softmax(z, T):
        z = z / T
        z = z - z.max(1, keepdims=True)
        return z - np.log(np.exp(z).sum(1, keepdims=True))


    def kd_loss(zs, zt, T=4.0):
        ls, lt = _log_softmax(zs, T), _log_softmax(zt, T)
        ps, pt = np.exp(ls), np.exp(lt)
        N = len(zs)
        loss = T * T * (pt * (lt - ls)).sum() / N
        return float(loss), T * (ps - pt) / N
    """,
    tests="""
    rng = np.random.default_rng(0)
    zs, zt = rng.standard_normal((6, 5)) * 3, rng.standard_normal((6, 5)) * 3

    def test_zero_when_equal():
        "loss is 0 and gradient is 0 when student == teacher"
        L, g = student.kd_loss(zt.copy(), zt.copy())
        assert abs(L) < 1e-9 and np.abs(g).max() < 1e-12

    def test_matches_lab_implementation():
        "same value and gradient as lab.train.KD"
        from lab.train import KD, Batch
        for T in (1.0, 4.0, 10.0):
            L, g = student.kd_loss(zs, zt, T)
            L2, g2, _ = KD(T=T)(zs, None, Batch(zs, None, None, zt))
            # lab.train.KD adds 1e-8 inside a log, hence the small tolerance
            assert abs(L - L2) < 1e-4 * max(1.0, abs(L2)), f"T={T}: loss {L} vs {L2}"
            assert np.abs(g - g2).max() < 1e-6, f"T={T}: gradient differs"

    def test_gradient_finite_differences():
        "gradient agrees with a numerical derivative"
        L, g = student.kd_loss(zs.copy(), zt, 3.0)
        num = np.zeros_like(zs)
        for i in range(6):
            for j in range(5):
                e = np.zeros_like(zs); e[i, j] = 1e-6
                num[i, j] = (student.kd_loss(zs + e, zt, 3.0)[0] - student.kd_loss(zs - e, zt, 3.0)[0]) / 2e-6
        assert np.abs(g - num).max() < 1e-5
    """,
    hints=["Compute log-probabilities with a stable log-softmax; `log(softmax(...))` can hit log(0).",
           "The `T^2` in the loss and the `1/T` from the chain rule leave a single `T` in the gradient."],
)

# ===========================================================================
# 6. magnitude pruning
# ===========================================================================
lesson(
    id="magnitude_mask", title="6. Global magnitude pruning",
    methods=["magnitude_pruning", "lottery_ticket"],
    intro="""
    Magnitude pruning keeps the largest weights and zeros the rest. **Global**
    pruning ranks the weights of all layers together, so a layer full of tiny
    weights can lose most of them while a sensitive layer keeps more.

    Implement `global_masks(weights, sparsity)`: given a list of weight
    matrices and the fraction to remove, return one 0/1 mask per matrix
    (same shapes) that removes exactly `round(sparsity * total)` weights, the
    smallest in absolute value across all matrices.

    Then the training loop multiplies weights by their mask after every step
    so pruned weights stay dead (`lab/nn.py:MLP.apply_masks`).
    """,
    skeleton="""
    import numpy as np


    def global_masks(weights, sparsity):
        \"\"\"Return a list of 0/1 float arrays, one per weight matrix.\"\"\"
        raise NotImplementedError
    """,
    solution="""
    import numpy as np


    def global_masks(weights, sparsity):
        flat = np.concatenate([np.abs(w).ravel() for w in weights])
        k = int(round(sparsity * flat.size))
        drop = np.zeros(flat.size, bool)
        drop[np.argsort(flat, kind="stable")[:k]] = True
        masks, i = [], 0
        for w in weights:
            masks.append((~drop[i:i + w.size]).reshape(w.shape).astype(float))
            i += w.size
        return masks
    """,
    tests="""
    rng = np.random.default_rng(0)
    Ws = [rng.standard_normal((20, 30)), rng.standard_normal((30, 30)) * 0.1, rng.standard_normal((30, 5))]

    def test_exact_count():
        "removes exactly round(sparsity * total) weights"
        for s in (0.0, 0.3, 0.9, 1.0):
            ms = student.global_masks(Ws, s)
            removed = sum((m == 0).sum() for m in ms)
            want = round(s * sum(w.size for w in Ws))
            assert removed == want, f"sparsity {s}: removed {removed}, want {want}"

    def test_keeps_largest():
        "every kept weight is at least as large as every removed one"
        ms = student.global_masks(Ws, 0.6)
        kept = np.concatenate([np.abs(w)[m == 1] for w, m in zip(Ws, ms)])
        gone = np.concatenate([np.abs(w)[m == 0] for w, m in zip(Ws, ms)])
        assert kept.min() >= gone.max()

    def test_global_not_per_layer():
        "the small-weight layer loses more than its share"
        ms = student.global_masks(Ws, 0.5)
        assert (ms[1] == 0).mean() > 0.75, "rank all layers together, not each layer separately"
    """,
    hints=["Concatenate `abs(w).ravel()` for every matrix, rank once, then split back by sizes.",
           "`np.argsort(...)[:k]` gives the indices of the k smallest."],
)

# ===========================================================================
# 7. fake quantization
# ===========================================================================
lesson(
    id="fake_quant", title="7. Fake quantization with group scales",
    methods=["ptq", "qat", "conv_quantization"],
    intro="""
    To store a weight in `b` bits with a symmetric scheme, pick a scale `s`
    and keep the integer `q = round(w / s)`, clipped to `[-qmax, qmax]` where
    `qmax = 2^(b-1) - 1`. The stored model holds `q` and `s`; using it means
    computing `q * s`.

    **Fake quantization** returns `q * s` in float directly, so you can measure
    the accuracy of the quantized model without any integer kernels. That's
    what `harness/quant.py` does to YOLO and `lab/nn.py:fake_quantize` does here.

    The scale is shared by a **group** of consecutive weights:
    `s = max|w in group| / qmax`. Smaller groups, more scales, less error.

    Implement `fake_quantize(w, bits, group)` for `bits` in 2..8, where
    `group=None` means one scale for the whole tensor. Group along the
    flattened tensor; the last group may be shorter.
    """,
    skeleton="""
    import numpy as np


    def fake_quantize(w, bits, group=None):
        \"\"\"Return an array like w, rounded to the levels a `bits`-bit symmetric code can store.\"\"\"
        raise NotImplementedError
    """,
    solution="""
    import numpy as np


    def fake_quantize(w, bits, group=None):
        qmax = 2 ** (bits - 1) - 1
        flat = w.ravel().astype(float)
        g = group or flat.size
        out = np.empty_like(flat)
        for s in range(0, flat.size, g):
            blk = flat[s:s + g]
            scale = max(np.abs(blk).max(), 1e-12) / qmax
            out[s:s + g] = np.clip(np.round(blk / scale), -qmax, qmax) * scale
        return out.reshape(w.shape)
    """,
    tests="""
    from lab.nn import fake_quantize as ref
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 12)).astype(np.float32)

    def test_level_count():
        "a 3-bit tensor-wide quantizer uses at most 7 distinct values"
        q = student.fake_quantize(w, 3)
        assert len(np.unique(np.round(q, 6))) <= 7

    def test_matches_lab_reference():
        "same result as lab.nn.fake_quantize for several bit widths and groups"
        for bits in (2, 3, 4, 8):
            for g in (None, 16, 7):
                a = student.fake_quantize(w, bits, g); b = ref(w, bits, g)
                assert np.allclose(a, b, atol=1e-5), f"bits={bits} group={g}: max diff {np.abs(a - b).max():.2e}"

    def test_smaller_groups_less_error():
        "error shrinks as groups get smaller"
        errs = [((student.fake_quantize(w, 3, g) - w) ** 2).mean() for g in (None, 32, 8)]
        assert errs[0] > errs[1] > errs[2], f"errors {errs}"
    """,
    hints=["Flatten, loop over chunks of `group`, quantize each with its own scale, reshape back.",
           "`group=None` is the same as one group the size of the whole tensor."],
)

# ===========================================================================
# 8. neuron surgery
# ===========================================================================
lesson(
    id="neuron_surgery", title="8. Structured pruning: removing neurons for real",
    methods=["structured_pruning", "channel_pruning"],
    intro="""
    Zeroing weights doesn't make a network faster on ordinary hardware; the
    matrix multiply is the same size. **Structured** pruning deletes whole
    neurons, so the matrices shrink.

    Deleting hidden neuron `j` between two layers means three edits that must
    agree:

    1. remove column `j` of `W1` (its incoming weights),
    2. remove entry `j` of `b1`,
    3. remove row `j` of `W2` (its outgoing weights).

    Miss one and the shapes no longer match. In a CNN with skip connections
    the same idea needs a *dependency group* of layers, which is what
    torch-pruning solves for the cockpit's YOLOv8n.

    Implement `keep_neurons(W1, b1, W2, keep)` returning the smaller
    `(W1, b1, W2)` that keeps only the hidden neurons listed in `keep`, and
    `l1_scores(W1)` returning one importance score per hidden neuron (the L1
    norm of its incoming weights).
    """,
    skeleton="""
    import numpy as np


    def l1_scores(W1):
        raise NotImplementedError


    def keep_neurons(W1, b1, W2, keep):
        raise NotImplementedError
    """,
    solution="""
    import numpy as np


    def l1_scores(W1):
        return np.abs(W1).sum(axis=0)


    def keep_neurons(W1, b1, W2, keep):
        keep = np.sort(np.asarray(keep))
        return W1[:, keep], b1[keep], W2[keep, :]
    """,
    tests="""
    rng = np.random.default_rng(0)
    W1, b1, W2 = rng.standard_normal((5, 8)), rng.standard_normal(8), rng.standard_normal((8, 3))
    X = rng.standard_normal((10, 5))
    f = lambda W1_, b1_, W2_: np.maximum(X @ W1_ + b1_, 0) @ W2_

    def test_shapes():
        "keeping 3 of 8 neurons gives (5,3), (3,), (3,3)"
        a, b, c = student.keep_neurons(W1, b1, W2, [6, 1, 4])
        assert (a.shape, b.shape, c.shape) == ((5, 3), (3,), (3, 3))

    def test_removing_dead_neurons_is_exact():
        "deleting neurons whose outgoing weights are zero leaves the function unchanged"
        W2d = W2.copy(); W2d[[0, 3, 5]] = 0
        keep = [1, 2, 4, 6, 7]
        a, b, c = student.keep_neurons(W1, b1, W2d, keep)
        assert np.allclose(f(W1, b1, W2d), f(a, b, c))

    def test_l1_scores():
        "one score per hidden neuron = column L1 norm of W1"
        s = student.l1_scores(W1)
        assert s.shape == (8,) and np.allclose(s, np.abs(W1).sum(0))
    """,
    hints=["Column j of W1 is `W1[:, j]`; row j of W2 is `W2[j, :]`.",
           "Index with the whole `keep` array at once: `W1[:, keep]`."],
)

# ===========================================================================
# 9. low rank
# ===========================================================================
lesson(
    id="low_rank_svd", title="9. Low-rank factorization with the SVD",
    methods=["low_rank", "qlora"],
    intro="""
    Any matrix `W` (m x n) can be written as `U diag(S) V^T` with singular
    values `S` sorted from large to small. Keeping only the first `r` gives the
    **best** rank-`r` approximation in squared error (the Eckart-Young
    theorem), and the error is exactly the energy you dropped:

    `||W - W_r||_F = sqrt(S[r]^2 + S[r+1]^2 + ...)`

    Store it as two thin matrices `A = U_r * sqrt(S_r)` (m x r) and
    `B = sqrt(S_r) * V_r^T` (r x n): `m*r + r*n` numbers instead of `m*n`.
    It only saves space when `r < m*n / (m + n)`.

    Implement `factorize(W, r)` returning `(A, B)` with `A @ B` the best
    rank-r approximation.
    """,
    skeleton="""
    import numpy as np


    def factorize(W, r):
        raise NotImplementedError
    """,
    solution="""
    import numpy as np


    def factorize(W, r):
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        root = np.sqrt(S[:r])
        return U[:, :r] * root, root[:, None] * Vt[:r]
    """,
    tests="""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((40, 30)) @ np.diag(np.linspace(3, 0.01, 30)) @ rng.standard_normal((30, 30))

    def test_shapes():
        "A is (m, r), B is (r, n)"
        A, B = student.factorize(W, 5)
        assert A.shape == (40, 5) and B.shape == (5, 30)

    def test_eckart_young_error():
        "the error equals the norm of the dropped singular values"
        S = np.linalg.svd(W, compute_uv=False)
        for r in (1, 5, 20):
            A, B = student.factorize(W, r)
            err = np.linalg.norm(W - A @ B)
            assert abs(err - np.sqrt((S[r:] ** 2).sum())) < 1e-6 * np.linalg.norm(W), f"rank {r}"

    def test_full_rank_is_exact():
        "rank = min(m, n) reproduces W"
        A, B = student.factorize(W, 30)
        assert np.allclose(A @ B, W)
    """,
    hints=["`np.linalg.svd(W, full_matrices=False)` returns U, S (a vector), Vt.",
           "Splitting sqrt(S) between the two factors keeps them balanced in size."],
)

# ===========================================================================
# 10. GPTQ
# ===========================================================================
lesson(
    id="gptq_loop", title="10. GPTQ's error-compensation loop",
    methods=["gptq", "sparsegpt"],
    intro="""
    Round-to-nearest quantizes each weight independently. GPTQ notices that
    what we care about is the layer's **output** on real inputs `X`, and that
    the error from rounding one column can be partly cancelled by nudging the
    columns that haven't been quantized yet.

    With `H = 2 X^T X / N` and `U` the upper Cholesky factor of `H^-1`
    (so `H^-1 = U^T U`), process columns `j = 0, 1, ...` of `W` (shape out x in):

    1. `q = round_to_grid(W[:, j])`
    2. `err = (W[:, j] - q) / U[j, j]`
    3. `W[:, j+1:] -= outer(err, U[j, j+1:])`
    4. store `q` as column `j` of the result.

    Step 3 is the Optimal Brain Surgeon update: the change to the remaining
    weights that best compensates the error, given the input statistics.

    You get `quantize_column(col, scale, qmax)` and the Cholesky factor.
    Implement `gptq(W, X, bits)` with one scale per output row computed from
    the original `W`. The test checks that your output error on `X` is well
    below plain rounding's.
    """,
    skeleton="""
    import numpy as np


    def hessian_inverse_chol(X, damp=0.01):
        H = 2 * X.T @ X / len(X)
        H += damp * np.mean(np.diag(H)) * np.eye(len(H))
        return np.linalg.cholesky(np.linalg.inv(H)).T


    def quantize_column(col, scale, qmax):
        return np.clip(np.round(col / scale), -qmax, qmax) * scale


    def gptq(W, X, bits=3):
        \"\"\"W: (out, in) weights. X: (N, in) calibration inputs. Return quantized W.\"\"\"
        qmax = 2 ** (bits - 1) - 1
        U = hessian_inverse_chol(X)
        scale = np.maximum(np.abs(W).max(axis=1), 1e-12) / qmax
        raise NotImplementedError
    """,
    solution="""
    import numpy as np


    def hessian_inverse_chol(X, damp=0.01):
        H = 2 * X.T @ X / len(X)
        H += damp * np.mean(np.diag(H)) * np.eye(len(H))
        return np.linalg.cholesky(np.linalg.inv(H)).T


    def quantize_column(col, scale, qmax):
        return np.clip(np.round(col / scale), -qmax, qmax) * scale


    def gptq(W, X, bits=3):
        qmax = 2 ** (bits - 1) - 1
        U = hessian_inverse_chol(X)
        scale = np.maximum(np.abs(W).max(axis=1), 1e-12) / qmax
        W = W.copy()
        Q = np.zeros_like(W)
        for j in range(W.shape[1]):
            q = quantize_column(W[:, j], scale, qmax)
            Q[:, j] = q
            err = (W[:, j] - q) / U[j, j]
            W[:, j + 1:] -= np.outer(err, U[j, j + 1:])
        return Q
    """,
    tests="""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((512, 32)) @ rng.standard_normal((32, 32))   # correlated inputs
    W = rng.standard_normal((16, 32))

    def rtn(W, bits):
        qmax = 2 ** (bits - 1) - 1
        s = np.abs(W).max(1, keepdims=True) / qmax
        return np.clip(np.round(W / s), -qmax, qmax) * s

    def test_does_not_modify_input():
        "the caller's W is unchanged"
        W0 = W.copy(); student.gptq(W, X, 3)
        assert np.array_equal(W, W0), "work on a copy"

    def test_beats_round_to_nearest():
        "output error on X is at most 60% of RTN's"
        for bits in (2, 3):
            e_g = ((X @ (student.gptq(W, X, bits) - W).T) ** 2).mean()
            e_r = ((X @ (rtn(W, bits) - W).T) ** 2).mean()
            assert e_g < 0.6 * e_r, f"{bits}-bit: gptq {e_g:.3f} vs rtn {e_r:.3f}"

    def test_on_the_grid():
        "every output value is an integer multiple of its row's scale"
        Q = student.gptq(W, X, 3)
        s = np.abs(W).max(1, keepdims=True) / 3
        assert np.allclose(Q / s, np.round(Q / s), atol=1e-6)

    def test_matches_lab():
        "same result as lab.layerwise.gptq"
        from lab.layerwise import gptq as ref
        assert np.allclose(student.gptq(W, X, 3), ref(W, X, 3), atol=1e-6)
    """,
    hints=["Work on `W = W.copy()` because step 3 rewrites the columns you haven't reached.",
           "`U[j, j + 1:]` is a row vector; `np.outer(err, U[j, j + 1:])` has the shape of `W[:, j + 1:]`."],
)
