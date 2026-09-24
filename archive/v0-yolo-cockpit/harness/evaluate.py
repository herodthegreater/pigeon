"""Evaluators behind one seam, so Phase 2 (LLM + lm-eval-harness) is a plug-in,
not a rewrite. Convention: score() returns a float where **higher = better**.

`BirdmAPEvaluator` computes single-class AP@0.50 (VOC all-points interpolation)
by treating every ground-truth box as 'bird' and asking the detector for COCO
class 14. That sidesteps the label-space mismatch (our YOLO export uses class 0,
the model predicts class 14) and gives a fast, dependency-light metric that's
consistent across prune levels -- which is what matters for a relative curve.
"""

from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image

BIRD = 14  # COCO index the model predicts


class Evaluator:
    name = "metric"
    unit = ""

    def score(self, yolo) -> float:  # higher = better
        raise NotImplementedError


def _iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(P,4)x(G,4) xyxy -> (P,G) IoU."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    p, g = pred[:, None, :], gt[None, :, :]
    x1 = np.maximum(p[..., 0], g[..., 0])
    y1 = np.maximum(p[..., 1], g[..., 1])
    x2 = np.minimum(p[..., 2], g[..., 2])
    y2 = np.minimum(p[..., 3], g[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


class BirdmAPEvaluator(Evaluator):
    name = "bird mAP@50"
    unit = ""

    def __init__(self, root: str, imgsz: int = 640, conf: float = 0.001,
                 iou_thr: float = 0.5, device: str | None = None,
                 max_images: int | None = None):
        self.images = sorted(glob.glob(os.path.join(root, "images", "val", "*")))
        if max_images:
            self.images = self.images[:max_images]  # fast subset for live eval
        labels_dir = os.path.join(root, "labels", "val")
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou_thr, device
        self.gt: dict[str, np.ndarray] = {}
        for img in self.images:
            w, h = Image.open(img).size
            lp = os.path.join(labels_dir, os.path.splitext(os.path.basename(img))[0] + ".txt")
            boxes = []
            if os.path.exists(lp):
                for line in open(lp):
                    p = line.split()
                    if len(p) < 5:
                        continue
                    cx, cy, bw, bh = map(float, p[1:5])
                    boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                  (cx + bw / 2) * w, (cy + bh / 2) * h])
            self.gt[img] = np.array(boxes, dtype=float) if boxes else np.zeros((0, 4))
        self.n_gt = int(sum(len(v) for v in self.gt.values()))

    def score(self, yolo) -> float:
        results = yolo.predict(source=self.images, classes=[BIRD], conf=self.conf,
                               imgsz=self.imgsz, device=self.device,
                               verbose=False, stream=True)
        scores, is_tp = [], []
        for img, r in zip(self.images, results):
            gt = self.gt[img]
            matched = np.zeros(len(gt), dtype=bool)
            if r.boxes is None or len(r.boxes) == 0:
                continue
            pb = r.boxes.xyxy.cpu().numpy()
            ps = r.boxes.conf.cpu().numpy()
            order = np.argsort(-ps)
            pb, ps = pb[order], ps[order]
            ious = _iou_matrix(pb, gt)
            for i in range(len(pb)):
                scores.append(ps[i])
                if gt.shape[0]:
                    j = int(np.argmax(ious[i]))
                    if ious[i, j] >= self.iou and not matched[j]:
                        matched[j] = True
                        is_tp.append(1)
                        continue
                is_tp.append(0)

        if self.n_gt == 0 or not scores:
            return 0.0
        order = np.argsort(-np.array(scores))
        tp = np.array(is_tp)[order]
        tpc, fpc = np.cumsum(tp), np.cumsum(1 - tp)
        recall = tpc / self.n_gt
        precision = tpc / np.clip(tpc + fpc, 1e-9, None)
        # VOC all-points interpolation
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
