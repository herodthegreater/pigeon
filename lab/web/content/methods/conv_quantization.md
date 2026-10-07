## The idea

Quantization stores each weight in fewer bits, for example as an 8-bit integer instead of a 16-bit float. pigeon measures the cost with *fake quantization*: round every weight to the nearest value an N-bit format could hold, but keep it as a float. The normal forward pass then gives the accuracy a real N-bit export with the same scheme would have. What keeps low bit-widths alive is the shared *scale*: each group of weights gets one float multiplier, and smaller groups fit their weights more tightly.

## How it works

1. Flatten the weight tensor and cut it into groups of `group_size` consecutive values, or use one group for the whole tensor (*per-tensor*).
2. For each group, `scale = max(|w|) / qmax`, where `qmax = 2^(bits-1) - 1` (127 at 8 bits, 7 at 4 bits).
3. `q = clamp(round(w / scale), -qmax, qmax)` is the integer a real file would store.
4. `w_hat = q * scale` is the float the model uses.
5. At 1 bit there is no zero level: `w_hat = sign(w) * mean(|w|)`.

Example at 4 bits, one group `[0.50, -0.20, 0.03]`: `scale = 0.5 / 7 ≈ 0.0714`, `q = [7, -3, 0]`, `w_hat = [0.50, -0.214, 0.0]`. The 0.03 vanished: the largest weight sets the step for the whole group.

## The code, line by line

`_pos_levels(bits)` returns `(1 << (bits - 1)) - 1`, so 127 for 8 bits. Dropping the -128 level keeps zero exactly in the middle.

`fake_quantize(w, bits, group_size)`:

- `bits >= 16` returns `w.clone()`: the native end of the dial.
- `flat = w.reshape(-1).float()` flattens in row-major order. For a conv weight of shape (out, in, kh, kw), a group walks through one filter, then into the next.
- If `group_size` is `None`, not positive, or at least `n`, `groups = flat.unsqueeze(0)`: one row, one scale.
- Otherwise `pad = (-n) % group_size` zero-pads to a whole number of groups and `groups = flat.reshape(-1, group_size)`.
- Multi-bit branch: `scale = groups.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax`. The clamp stops an all-zero group from dividing by zero. Then `q = torch.clamp(torch.round(groups / scale), -qmax, qmax)` and `deq = q * scale`.
- 1-bit branch: `scale` is the mean absolute value, `q = torch.sign(groups)`, and `torch.where(q == 0, ...)` maps exact zeros to +1.
- `out[:-pad]` drops the padding, then shape and dtype are restored.

With `group_size == 1` every weight comes back unchanged, a deliberate no-op per the docstring.

`apply_uniform(model, bits, group_size, skip)` in harness/model.py covers the whole network. `iter_weight_modules` yields every module in `QUANTIZABLE = (nn.Conv2d, nn.Linear, nn.Embedding)` that has a weight; biases and batch-norm stay native. For each module whose name contains no `skip` substring, `mod.weight.data.copy_(fake_quantize(mod.weight.data, bits, group_size))` overwrites the weight in place under `@torch.no_grad()`. It returns how many modules it touched.

Because this is destructive, `snapshot` clones every quantizable weight once and `restore` copies them back. `/api/quantize` in server.py calls `restore(yolo.model, SNAP)` before each `apply_uniform`, so every bit-width starts from native. server.py also runs one `predict` before `snapshot`, which (per its comment) fuses each conv with its batch norm, so the cockpit quantizes the folded weights a deployed model would store (see bn_folding).

## What the simulations show

The toy-MLP analogue, not YOLO, is `quantization` (lab/results/quantization.json), using a NumPy twin of `fake_quantize` with groups of 16. The table gives 20-class accuracy after rounding, no retraining. In float, the teacher (3x256) scores 0.9139 and the student (2x32) 0.8009.

| bits | teacher, per-tensor | teacher, group-16 | student, per-tensor | student, group-16 |
|---|---|---|---|---|
| 8 | 0.914 | 0.9138 | 0.801 | 0.8014 |
| 6 | 0.9139 | 0.9133 | 0.7966 | 0.7965 |
| 4 | 0.8998 | 0.9128 | 0.7186 | 0.767 |
| 3 | 0.7766 | 0.897 | 0.4461 | 0.6283 |
| 2 | 0.0707 | 0.6098 | 0.0764 | 0.2173 |

The shape matches the detector's: 8 bits are free, per-group scales extend the range, then a cliff. The small student breaks earlier, and YOLOv8n is a small model. Fine-tuning with the rounding in the loop (quantization-aware training, with KD) brought the 2-bit teacher back to 0.91165. The toy's cliff sits at 2 to 3 bits, not the detector's 4, so don't carry the exact bit-width over.

## When it breaks

- The real result (STATUS.md, from `carve.py`): 8-bit is about free, 6-bit survives only with `group_size=128`, 4-bit collapses. `carve.py` loads `yolo11n.pt`, so these were measured on YOLO11n, not the cockpit's YOLOv8n.
- Groups ignore filter boundaries. A 3x3 conv with 16 inputs has 144 values per filter, so a 128-value group can straddle two filters. Most deployment formats use one scale per output channel instead.
- `skip` is a substring match: `"model.2"` also matches `model.20`, `model.21` and `model.22`. The docstring's `("model.23",)` spares YOLO11n's head; YOLOv8n's head is `model.22`. The cockpit's `/api/quantize` passes no `skip`, so the head is always quantized.
- The head's DFL layer (it decodes each box edge from 16 bins) is a `Conv2d` with fixed weights 0 to 15. It is quantized too, so at low bits box decoding is distorted regardless of the learned weights.
- Activations stay in float; a real int8 deployment also quantizes them.
- File size and speed don't change until you export to a real integer format. The cockpit's size estimate `NATIVE_PARAMS * bits / 8` ignores the stored scales.

## Going further

In the cockpit (`uvicorn server:app --port 8000`), the Precision module has "Apply 8-bit", a ladder of bit-widths from 16b to 1b, and a scale-sharing toggle, "per-group (128)" or "per-tensor (crude)". Each click posts to `/api/quantize` and scores the real detector on the 40-image live subset. Headless, `python carve.py --group-size 128` sweeps the same ladder on the 125-image set. For a real small file, export to a runtime with integer kernels and match its scheme so these numbers transfer. The toy version is `lab/compress.py:quantize`.
