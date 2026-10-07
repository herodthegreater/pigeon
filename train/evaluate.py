"""Baseline scoreboard on the fixed eval set: the general model vs specialized Pigeon.

Run:  python -m train.evaluate      -> prints a table, writes runs/baseline.json
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .dataset import load_sheets, to_input
from .metrics import predict, scoreboard
from .model import Pigeon, general_model, imagenet_bird_classes, param_count
from .train import device

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


@torch.no_grad()
def general_zero_shot(split, dev, bs=128) -> torch.Tensor:
    """The untouched ImageNet model says "bird" when its top-1 class is one of the 59 bird classes."""
    from torchvision.transforms.functional import normalize

    m = general_model().to(dev).eval()
    birds = torch.tensor(imagenet_bird_classes(), device=dev)
    preds = []
    for i in range(0, len(split), bs):
        x = normalize(to_input(split.images[i : i + bs].to(dev)), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        top1 = m(x).argmax(1)
        preds.append(torch.isin(top1, birds).float().cpu())
    return torch.cat(preds)


def main() -> None:
    dev = device()
    ev = load_sheets()  # the exact pixels the site evaluates

    g_pred = general_zero_shot(ev, dev)
    g_acc = (g_pred == ev.labels).float().mean().item()
    g_hard = (g_pred[ev.hard] == 0).float().mean().item()
    g_params = param_count(general_model())

    model = Pigeon(pretrained=False).to(dev)
    model.load_state_dict(torch.load(RUNS / "pigeon_fp32.pt", map_location=dev))
    logits, cams = predict(model, ev, dev)
    p = scoreboard(logits, cams, ev, ref_pred=(logits > 0).float())
    p_params = param_count(model)

    out = {
        "eval_images": len(ev),
        "eval_birds": int(ev.labels.sum()),
        "eval_hard_negatives": int(ev.hard.sum()),
        "general": {"params": g_params, "bytes_fp32": g_params * 4, "accuracy": g_acc, "hard_neg_acc": g_hard},
        "pigeon": {"params": p_params, "bytes_fp32": p_params * 4, **p},
    }
    (RUNS / "baseline.json").write_text(json.dumps(out, indent=1))
    print(f"eval set: {len(ev)} images, {int(ev.labels.sum())} birds, {int(ev.hard.sum())} hard negatives\n")
    print(f"{'model':28s} {'params':>10s} {'size':>9s} {'acc':>7s} {'hard-neg':>9s} {'pointing':>9s}")
    print(f"{'general (top-1 is a bird)':28s} {g_params:>10,d} {g_params*4/1e6:>7.2f}MB {g_acc:>7.3f} {g_hard:>9.3f} {'—':>9s}")
    print(
        f"{'Pigeon (specialized)':28s} {p_params:>10,d} {p_params*4/1e6:>7.2f}MB {p['accuracy']:>7.3f} "
        f"{p['hard_neg_acc']:>9.3f} {p['pointing']:>9.3f}"
    )


if __name__ == "__main__":
    main()
