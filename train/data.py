"""Build Pigeon's bird / no-bird dataset from Open Images V7.

Splits (Open Images split -> Pigeon split):
  test        -> train   (~3.2k birds + equal negatives, ~40% of them hard)
  validation  -> val     (500: model selection, calibration)
  validation  -> eval    (500: the fixed test set the site ships and scores on)

Positives: at least one real (non-depiction) bird box covering >= MIN_AREA of
the image, so a 224 px classifier has something to see.
Negatives: no bird box and no positive bird image-level label.
Hard negatives: negatives containing things a bird-spotter could confuse —
flying objects, winged insects, small furry animals.

Images are squashed to SIZE x SIZE (boxes stay valid in normalized coords)
and saved as JPEG. A manifest CSV holds labels, boxes and attribution.

Run:  python -m train.data            (idempotent; skips files already present)
"""

from __future__ import annotations

import io
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data/openimages/meta"
OUT = ROOT / "data/pigeon"
META_URL = "https://storage.googleapis.com/openimages/"
META_FILES = [
    "v5/class-descriptions-boxable.csv",
    "2018_04/bbox_labels_600_hierarchy.json",
    "v5/validation-annotations-bbox.csv",
    "v5/validation-annotations-human-imagelabels-boxable.csv",
    "v5/test-annotations-bbox.csv",
    "v5/test-annotations-human-imagelabels-boxable.csv",
    "2018_04/validation/validation-images-with-rotation.csv",
    "2018_04/test/test-images-with-rotation.csv",
]
IMG_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"

SIZE = 256
BIRD_ROOT = "/m/015p6"
HARD_CLASSES = [
    # things in the sky
    "Kite", "Airplane", "Helicopter", "Balloon", "Rocket", "Flying disc",
    # winged, not birds
    "Bat (Animal)", "Butterfly", "Moths and butterflies", "Insect", "Dragonfly", "Bee",
    # small animals
    "Squirrel", "Rabbit", "Hamster", "Mouse", "Cat", "Fish",
]
SEED = 0

# (pigeon split, open images split, positives, negatives, hard negatives, min bird area)
PLAN = [
    ("train", "test", None, None, 0.40, 0.03),  # None = use every eligible positive, match negatives
    ("val", "validation", 250, 250, 0.40, 0.05),
    ("eval", "validation", 250, 250, 0.40, 0.05),
]


def fetch_meta() -> None:
    META.mkdir(parents=True, exist_ok=True)
    for f in META_FILES:
        dst = META / Path(f).name
        if not dst.exists():
            print("downloading", f)
            urllib.request.urlretrieve(META_URL + f, dst)


def bird_classes() -> set[str]:
    h = json.loads((META / "bbox_labels_600_hierarchy.json").read_text())

    def find(n):
        if n.get("LabelName") == BIRD_ROOT:
            return n
        for c in n.get("Subcategory", []):
            if r := find(c):
                return r

    def desc(n):
        out = {n["LabelName"]}
        for c in n.get("Subcategory", []):
            out |= desc(c)
        return out

    return desc(find(h))


def candidates(oi_split: str, min_area: float):
    """Return (positives df with boxes, negatives list, hard-negative set, image metadata)."""
    names = pd.read_csv(META / "class-descriptions-boxable.csv", header=None, names=["id", "name"])
    to_id = dict(zip(names.name, names.id))
    birds = bird_classes()

    bx = pd.read_csv(META / f"{oi_split}-annotations-bbox.csv")
    lb = pd.read_csv(META / f"{oi_split}-annotations-human-imagelabels-boxable.csv")
    info = pd.read_csv(META / f"{oi_split}-images-with-rotation.csv").set_index("ImageID")
    upright = set(info.index[info.Rotation.fillna(0) == 0])

    bb = bx[bx.LabelName.isin(birds)].copy()
    bb["area"] = (bb.XMax - bb.XMin) * (bb.YMax - bb.YMin)
    real = bb[bb.IsDepiction != 1]
    biggest = real.groupby("ImageID").area.max()
    pos_ids = set(biggest.index[biggest >= min_area]) & upright
    # ambiguous: has bird boxes but only depictions / tiny ones -> used by neither side
    any_bird = set(bb.ImageID) | set(lb[lb.LabelName.isin(birds) & (lb.Confidence == 1)].ImageID)
    neg_ids = (set(bx.ImageID) - any_bird) & upright
    hard_ids = set(bx[bx.LabelName.isin([to_id[n] for n in HARD_CLASSES])].ImageID) & neg_ids

    boxes = (
        real[real.ImageID.isin(pos_ids)]
        .groupby("ImageID")
        .apply(lambda g: [[round(v, 4) for v in r] for r in g[["XMin", "YMin", "XMax", "YMax"]].values.tolist()],
               include_groups=False)
    )
    return boxes, sorted(neg_ids), hard_ids, info


def choose(oi_split, n_pos, n_neg, hard_frac, min_area, taken, rng):
    boxes, negs, hard, info = candidates(oi_split, min_area)
    pos = sorted(set(boxes.index) - taken)
    rng.shuffle(pos)
    pos = pos if n_pos is None else pos[:n_pos]
    n_neg = len(pos) if n_neg is None else n_neg
    n_hard = int(round(n_neg * hard_frac))
    hard_pool = sorted(hard - taken)
    easy_pool = sorted(set(negs) - hard - taken)
    rng.shuffle(hard_pool)
    rng.shuffle(easy_pool)
    rows = []
    for i in pos:
        rows.append(dict(image_id=i, label=1, hard=0, boxes=json.dumps(boxes[i])))
    for i in hard_pool[:n_hard]:
        rows.append(dict(image_id=i, label=0, hard=1, boxes="[]"))
    for i in easy_pool[: n_neg - n_hard]:
        rows.append(dict(image_id=i, label=0, hard=0, boxes="[]"))
    df = pd.DataFrame(rows)
    meta = info.loc[df.image_id, ["OriginalLandingURL", "License", "Author", "AuthorProfileURL", "Title"]]
    df = pd.concat([df.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
    df["oi_split"] = oi_split
    return df


def download(row, split_dir: Path) -> bool:
    dst = split_dir / f"{row.image_id}.jpg"
    if dst.exists():
        return True
    url = IMG_URL.format(split=row.oi_split, id=row.image_id)
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                im = Image.open(io.BytesIO(r.read())).convert("RGB")
            im.resize((SIZE, SIZE), Image.BICUBIC).save(dst, quality=90)
            return True
        except Exception:
            continue
    return False


def main() -> None:
    fetch_meta()
    rng = random.Random(SEED)
    taken: set[str] = set()
    frames = []
    for split, oi_split, n_pos, n_neg, hard_frac, min_area in PLAN:
        df = choose(oi_split, n_pos, n_neg, hard_frac, min_area, taken, rng)
        taken |= set(df.image_id)
        df.insert(0, "split", split)
        frames.append(df)
        print(f"{split:5s} {len(df):5d} images  birds={df.label.sum()}  hard={df.hard.sum()}")
    manifest = pd.concat(frames, ignore_index=True)

    ok = []
    for split, g in manifest.groupby("split", sort=False):
        d = OUT / split
        d.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(32) as ex:
            res = list(tqdm(ex.map(lambda r: download(r, d), g.itertuples()), total=len(g), desc=split))
        ok += res
    manifest["ok"] = ok
    dropped = (~manifest.ok).sum()
    manifest = manifest[manifest.ok].drop(columns="ok")
    manifest.to_csv(OUT / "manifest.csv", index=False)
    print(f"wrote {len(manifest)} rows to {OUT/'manifest.csv'} ({dropped} failed downloads dropped)")


if __name__ == "__main__":
    main()
