## The idea

A classifier's teacher gives one probability vector per image. A detector like YOLOv8 gives one prediction per grid cell at three scales: 8400 cells for a 640x640 image (80x80 + 40x40 + 20x20). Each cell has a class-score vector and a box. Most cells are background, so plain per-cell distillation mostly teaches "nothing here". Detector distillation transfers three signals: class scores, box regressions and neck features, each weighted toward cells where the teacher sees an object.

## How it works

1. Run teacher and student on the same batch at the same input size so their grids line up.
2. Class term: YOLOv8 scores each class with an independent sigmoid, not a softmax, so use binary cross-entropy against the teacher's sigmoid scores, weighted per cell by the teacher's top score `w`.
3. Box term: YOLOv8 predicts each box edge as a distribution over 16 bins (its DFL head). On cells where `w` is high, match the student's bin distribution to the teacher's with KL divergence at temperature `T`.
4. Feature term: at the three neck outputs, a 1x1 conv adapter maps student features to the teacher's width, then mean-squared error, ideally masked to foreground.
5. `L = det_loss(labels) + l_cls + l_box + l_feat`, each with its own weight.

```python
with torch.no_grad():
    ft, ct, bt = teacher_parts(img)   # neck feats, class logits [B,A,80], box logits [B,A,4,16]
fs, cs, bs = student_parts(img)
w = ct.sigmoid().amax(-1)                                   # [B,A] teacher confidence
bce = F.binary_cross_entropy_with_logits(cs, ct.sigmoid(), reduction="none").sum(-1)
l_cls = (bce * w).sum() / w.sum()
fg = w > 0.25
l_box = T * T * F.kl_div(F.log_softmax(bs[fg] / T, -1), F.softmax(bt[fg] / T, -1),
                         reduction="batchmean")
l_feat = sum(F.mse_loss(adapt(s), t) for adapt, s, t in zip(adapters, fs, ft))
loss = det_loss(student_out, targets) + a * l_cls + b * l_box + c * l_feat
```

## The code, line by line

This method has no code entry; the closest is `KD` in lab/train.py. `_teacher_probs` computes `softmax(tl, T)`. With a `group` matrix, `p = p @ self.G` sums teacher probabilities into a coarser label space, such as 20 classes into {bird, other}. That is how a bird-only detector would fold the teacher's 80 COCO scores. For `kind="kl"`, the loss is `T * T * (pt * (lt - ls)).sum() / N` and the logit gradient is `T * (ps - pt) / N`. A detector applies the same idea per cell, with sigmoid targets, plus the box and feature terms.

## When it breaks

- If strides or input sizes differ, cell i in the student is not cell i in the teacher.
- Without the confidence weight `w`, thousands of background cells dominate the loss.
- Softmax KD on YOLOv8's sigmoid scores mismatches the model's own loss.
- The teacher's missed birds become the student's missed birds.
- A pruned student has fewer channels, so feature matching needs the adapters.

## Going further

Missing here: a trainer whose loss adds these terms. Ultralytics' `DetectionTrainer`, which `recover_job.py` drives by setting `trainer.model` to the pruned network, computes only the label loss, so you would subclass the model's loss, keep a frozen teacher on the same device, and hook the neck outputs. In pigeon, the natural teacher is the unpruned, full-precision `yolov8n.pt`. The student is the model `recover_job.py` fine-tunes: pruned per the recipe and, when the recipe quantizes, with its convs swapped for `QATConv2d` so the forward pass sees rounded weights. The teacher also fills a gap in the labels: `build_finetune_set.py` keeps only the bird boxes (`only_matching=True`), so other objects in those 400 photos count as background, while the teacher's 80 class scores still describe them. The closest analogue is `lab/train.py:KD`.
