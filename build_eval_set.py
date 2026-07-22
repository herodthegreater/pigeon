"""Fetch a fixed bird eval set from COCO val2017 (the model's native label
space) and export it in YOLO format under eval_set/. Reusable: change N.

    python build_eval_set.py
"""

import fiftyone as fo
import fiftyone.zoo as foz

N = 150
OUT = "/Users/jarodferkin/Projects/pigeon/eval_set"

ds = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=["bird"],
    max_samples=N,
    only_matching=True,          # keep only bird boxes
    dataset_name="birds_eval",
    drop_existing_dataset=True,
)

ds.export(
    export_dir=OUT,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    classes=["bird"],
    split="val",
)

n_boxes = sum(len(s.ground_truth.detections) for s in ds if s.ground_truth)
print(f"\nexported {len(ds)} images, {n_boxes} bird boxes -> {OUT}")
