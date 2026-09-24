"""Build a bird fine-tuning dataset from COCO, in the model's native 80-class
space (bird = index 14), so the pruned 80-class head trains without reshaping
and the existing evaluator (which predicts class 14) needs no change.

Train split is used by Ultralytics during fine-tuning; final mAP is still
reported by BirdmAPEvaluator on eval_set/ (class-agnostic boxes), unchanged.
"""

from __future__ import annotations

import os

import fiftyone as fo
import fiftyone.zoo as foz
import yaml
from ultralytics import YOLO

OUT = "/Users/jarodferkin/Projects/pigeon/finetune_data"
NAMES = [YOLO("yolov8n.pt").names[i] for i in range(80)]  # COCO order; bird=14


def pull(split, n, name):
    return foz.load_zoo_dataset(
        "coco-2017", split=split, label_types=["detections"],
        classes=["bird"], max_samples=n, only_matching=True,
        dataset_name=name, drop_existing_dataset=True,
    )


def main() -> None:
    for ds, split in [(pull("train", 400, "bird_ft_train"), "train"),
                      (pull("validation", 125, "bird_ft_val"), "val")]:
        ds.export(export_dir=OUT, dataset_type=fo.types.YOLOv5Dataset,
                  label_field="ground_truth", classes=NAMES, split=split)

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        yaml.safe_dump(
            {"path": OUT, "train": "images/train", "val": "images/val",
             "names": {i: n for i, n in enumerate(NAMES)}},
            f, sort_keys=False,
        )
    n_tr = len(os.listdir(os.path.join(OUT, "images", "train")))
    n_va = len(os.listdir(os.path.join(OUT, "images", "val")))
    print(f"\nfinetune_data ready: {n_tr} train + {n_va} val images (bird=class 14)")


if __name__ == "__main__":
    main()
