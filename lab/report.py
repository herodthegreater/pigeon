"""Turn lab/results/*.json into figures + lab/results/RESULTS.md.

    python -m lab.report
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .common import QUICK, RESULTS  # noqa: E402

FIGS = os.path.join(RESULTS, "figures")

# Validated categorical order (dataviz reference palette, light mode). Order is
# the CVD-safety mechanism -- never cycle; markers/dashes double-encode identity.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DASHES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

LABELS = {"acc": "test accuracy (20-way)", "target_ap": "target-class AP ('bird')",
          "fidelity": "agreement with teacher", "nll": "test NLL", "kl_bayes": "KL(Bayes || model)",
          "params": "parameters (nonzero)", "macs": "MACs / example (nonzero)",
          "kbits": "storage (kbit)", "teacher_acc": "teacher accuracy",
          "teacher_kl_bayes": "teacher KL(Bayes || teacher)", "width": "hidden width",
          "sparsity": "fraction of weights removed", "bits": "bits / weight",
          "n_labels": "labeled examples", "steps": "training steps", "T": "temperature T",
          "noise": "label-noise rate", "teacher_width": "teacher width", "dropped": "layers removed (of 8)",
          "start_width": "width trained before pruning", "generation": "generation",
          "n_teachers": "teachers in ensemble", "smoothing": "teacher label smoothing",
          "rank_or_width": "rank / width", "overlap": "overlap"}
METRICS = ["acc", "target_ap", "fidelity", "nll", "ece", "kl_bayes", "teacher_acc", "params",
           "macs", "kbits", "overlap"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _filter(rows, spec):
    out = [r for r in rows if spec["x"] in r and spec["y"] in r]
    for k, allowed in (spec.get("filter") or {}).items():
        out = [r for r in out if k in r and (allowed is None or r[k] in allowed)]
    return out


def _series(rows, spec):
    g = spec.get("group")
    order: list = []
    by: dict = {}
    for r in rows:
        key = r.get(g, "") if g else ""
        if key not in by:
            by[key] = []
            order.append(key)
        by[key].append(r)
    return [(k, by[k]) for k in order]


def _hlines(ax, spec, refs, horizontal_bars=False):
    metric = spec.get("ref_metric", spec["y"])
    for name in spec.get("hlines", []):
        v = (refs.get(name) or {}).get(metric)
        if v is None:
            continue
        if horizontal_bars:
            ax.axvline(v, color=INK2, lw=1, ls=(0, (4, 3)))
            ax.text(v, 1.0, f" {name} {v:.3f}", color=INK2, fontsize=7, va="bottom",
                    transform=ax.get_xaxis_transform())
        else:
            ax.axhline(v, color=INK2, lw=1, ls=(0, (4, 3)))
            ax.text(1.0, v, f" {name} {v:.3f}", color=INK2, fontsize=7, va="center",
                    transform=ax.get_yaxis_transform())


def _plot(res, spec, path):
    rows = _filter(res["rows"], spec)
    if not rows:
        return False
    kind = spec.get("kind", "line")
    x, y = spec["x"], spec["y"]
    series = _series(rows, spec)
    if len(series) > len(PALETTE):
        raise ValueError(f"{res['name']}: {len(series)} series > {len(PALETTE)} slots; filter it")
    if kind == "bar":
        cats = list(dict.fromkeys(r[x] for r in rows))
        n = len(series)
        h = 0.8 / n
        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(cats) * max(1, n * 0.7) + 1.2))
        _style(ax)
        for si, (name, rs) in enumerate(series):
            vals = {r[x]: r for r in rs}
            for ci, c in enumerate(cats):
                if c not in vals:
                    continue
                v = vals[c][y]
                err = vals[c].get(y + "_std")
                pos = ci + (si - (n - 1) / 2) * h
                ax.barh(pos, v, height=h * 0.9, color=PALETTE[si], xerr=err,
                        error_kw=dict(ecolor=INK2, lw=0.8, capsize=2),
                        label=name if ci == 0 or c == next(iter(vals)) else None)
                ax.text(v, pos, f" {v:.3f}", va="center", fontsize=7, color=INK)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels([str(c) for c in cats], fontsize=8, color=INK)
        ax.invert_yaxis()
        ax.set_xlabel(LABELS.get(y, y), color=INK2, fontsize=9)
        lo = min(r[y] for r in rows)
        ax.set_xlim(max(0, lo - 0.1 * (1 - lo) - 0.05), None)
        _hlines(ax, spec, res.get("refs", {}), horizontal_bars=True)
        if n > 1:
            ax.legend(fontsize=7, frameon=False, loc="lower right")
    else:
        fig, ax = plt.subplots(figsize=(7.5, 4.3))
        _style(ax)
        for si, (name, rs) in enumerate(series):
            rs = sorted(rs, key=lambda r: r[x])
            xs, ys = [r[x] for r in rs], [r[y] for r in rs]
            col = PALETTE[si]
            if kind == "scatter" and not spec.get("connect"):
                ax.scatter(xs, ys, s=40, color=col, edgecolor=SURFACE, lw=1.5, zorder=3,
                           marker=MARKERS[si], label=name or None)
            else:
                ax.plot(xs, ys, color=col, lw=2, ls=DASHES[si], marker=MARKERS[si], ms=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, label=name or None, zorder=3)
                sd = [r.get(y + "_std") for r in rs]
                if all(v is not None for v in sd):
                    ax.fill_between(xs, [a - b for a, b in zip(ys, sd)],
                                    [a + b for a, b in zip(ys, sd)], color=col, alpha=0.12, lw=0)
            if spec.get("label"):
                for r in rs:
                    ax.annotate(str(r[spec["label"]]), (r[x], r[y]), fontsize=6.5, color=INK2,
                                xytext=(4, 3), textcoords="offset points")
        if spec.get("xlog"):
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ticks = sorted({r[x] for r in rows})
            if len(ticks) <= 10:
                ax.set_xticks(ticks)
                ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_xlabel(LABELS.get(x, x), color=INK2, fontsize=9)
        ax.set_ylabel(LABELS.get(y, y), color=INK2, fontsize=9)
        _hlines(ax, spec, res.get("refs", {}))
        if len(series) > 1 or series[0][0]:
            ax.legend(fontsize=7, frameon=False, loc="best")
    ax.set_title(spec.get("title") or res["title"], fontsize=10, color=INK, loc="left")
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return True


def _fmt(r, k):
    v = r.get(k)
    if v is None:
        return ""
    if isinstance(v, float):
        if k in ("params", "macs"):
            s = f"{v:,.0f}"
        elif k == "kbits":
            s = f"{v:,.1f}"
        else:
            s = f"{v:.4f}" if abs(v) < 10 else f"{v:,.1f}"
        sd = r.get(k + "_std")
        return s + (f" ± {sd:.3f}" if sd is not None and k not in ("params", "macs", "kbits") else "")
    return str(v)


def _table(res):
    rows = res["rows"]
    keys = []
    for spec in res.get("plots", []):
        for k in (spec.get("group"), spec["x"], spec.get("label")):
            if k and k not in keys and k not in METRICS and k != "series":
                keys.append(k)
    for r in rows:
        for k, v in r.items():
            if (isinstance(v, str) and k not in keys and k not in ("series", "arm")):
                keys.append(k)
    mets = [m for m in METRICS if any(m in r for r in rows)]
    cols = keys + mets
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r, c) for c in cols) + " |")
    return "\n".join(lines)


def build(quick: bool | None = None):
    quick = QUICK if quick is None else quick
    os.makedirs(FIGS, exist_ok=True)
    pat = "quick_*.json" if quick else "[!q]*.json"
    results = []
    for p in sorted(glob.glob(os.path.join(RESULTS, pat))):
        with open(p) as f:
            results.append(json.load(f))
    results.sort(key=lambda r: (r["group"], r["name"]))
    out = ["# Simulation results (auto-generated)",
           "",
           "Regenerate with `python -m lab.run` (or `python -m lab.report` to re-render). "
           "Interpretation lives in [`../REPORT.md`](../REPORT.md). Mean ± std over seeds "
           "where more than one seed ran.",
           ""]
    if results:
        b = results[0]["refs"]["bayes"]
        out += [f"Reference ceiling (Bayes-optimal): acc {b['acc']:.4f}, target AP {b['target_ap']:.4f}.", ""]
    group = None
    for res in results:
        if res["group"] != group:
            group = res["group"]
            out += [f"## {group}", ""]
        out += [f"### {res['title']}  (`{res['name']}`)", "", f"*{res['question']}*", ""]
        for i, spec in enumerate(res.get("plots", [])):
            fn = f"{'quick_' if quick else ''}{res['name']}_{i}.png"
            if _plot(res, spec, os.path.join(FIGS, fn)):
                out += [f"![{res['name']} {i}](figures/{fn})", ""]
        if res.get("notes"):
            out += [f"> {res['notes']}", ""]
        out += ["<details><summary>table</summary>", "", _table(res), "", "</details>", "",
                f"<sub>{res['seconds']:.0f}s</sub>", ""]
    name = "RESULTS_quick.md" if quick else "RESULTS.md"
    with open(os.path.join(RESULTS, name), "w") as f:
        f.write("\n".join(out))
    print(f"wrote {os.path.join(RESULTS, name)} ({len(results)} experiments)")


if __name__ == "__main__":
    build()
