"""Pack the fixed eval set for the site.

Outputs (site/public/testset/):
  sheet-0.webp … sheet-4.webp   10 x 10 grids of 224 px images (100 per sheet)
  testset.json                  per image: sheet, cell, label, hard, boxes, attribution,
                                and the uncompressed Pigeon's logit (browser parity check)

Images are interleaved so every sheet is balanced (half birds, same share of hard
negatives): sheet 0 alone is a fair 100-image quick test for live slider moves.

Run:  python -m train.export_testset       (after train.train)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from .dataset import INPUT, load_split
from .metrics import predict
from .model import Pigeon
from .train import device

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site/public/testset"
GRID = 10
PER_SHEET = GRID * GRID


def balanced_order(labels: list[int], hard: list[int]) -> list[int]:
    birds = [i for i, y in enumerate(labels) if y == 1]
    hardn = [i for i, (y, h) in enumerate(zip(labels, hard)) if y == 0 and h]
    easy = [i for i, (y, h) in enumerate(zip(labels, hard)) if y == 0 and not h]
    sheets = [[] for _ in range((len(labels) + PER_SHEET - 1) // PER_SHEET)]
    for group in (birds, hardn, easy):
        for k, i in enumerate(group):
            sheets[k % len(sheets)].append(i)
    order = []
    for s in sheets:  # interleave within the sheet too, so any prefix is roughly balanced
        b = [i for i in s if labels[i] == 1]
        n = [i for i in s if labels[i] == 0]
        for k in range(max(len(b), len(n))):
            order += b[k : k + 1] + n[k : k + 1]
    return order


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = load_split("eval")
    df = pd.read_csv(ROOT / "data/pigeon/manifest.csv")
    df = df[df.split == "eval"].set_index("image_id").loc[ev.ids]

    dev = device()
    model = Pigeon(pretrained=False).to(dev)
    model.load_state_dict(torch.load(ROOT / "runs/pigeon_fp32.pt", map_location=dev))
    logits, _ = predict(model, ev, dev)

    order = balanced_order(ev.labels.int().tolist(), ev.hard.int().tolist())
    records, sheets = [], {}
    for pos, i in enumerate(order):
        s, cell = divmod(pos, PER_SHEET)
        sheet = sheets.setdefault(s, Image.new("RGB", (GRID * INPUT, GRID * INPUT)))
        img = Image.fromarray(ev.images[i].permute(1, 2, 0).numpy()).resize((INPUT, INPUT), Image.BICUBIC)
        sheet.paste(img, ((cell % GRID) * INPUT, (cell // GRID) * INPUT))
        row = df.loc[ev.ids[i]]
        records.append(
            {
                "id": ev.ids[i],
                "sheet": s,
                "cell": cell,
                "label": int(ev.labels[i]),
                "hard": bool(ev.hard[i]),
                "boxes": ev.boxes[i],
                "ref_logit": round(float(logits[i]), 5),
                "credit": {
                    "author": None if pd.isna(row.Author) else row.Author,
                    "title": None if pd.isna(row.Title) else row.Title,
                    "url": row.OriginalLandingURL,
                    "license": row.License,
                },
            }
        )
    total = 0
    for s, im in sheets.items():
        p = OUT / f"sheet-{s}.webp"
        im.save(p, quality=82, method=6)
        total += p.stat().st_size
    (OUT / "testset.json").write_text(json.dumps({"grid": GRID, "size": INPUT, "images": records}))
    print(f"wrote {len(records)} images in {len(sheets)} sheets ({total/1e6:.1f} MB) to {OUT}")


if __name__ == "__main__":
    main()
