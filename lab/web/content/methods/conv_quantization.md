## The idea

Quantization stores each weight in fewer bits, for example as an 8-bit integer instead of a 16-bit float. The YOLOv8n cockpit measures the cost with *fake quantization*: round every weight to the nearest value an $N$-bit format could hold, but keep it as a float. The normal forward pass then gives the accuracy a real $N$-bit export with the same scheme would have. What keeps low bit-widths alive is the shared *scale*: each group of weights gets one float multiplier, and smaller groups fit their weights more tightly. For a CNN, round the weights inference actually uses, which means folding batch norm into each conv first.

## How it works

1. Fuse: fold each batch-norm (BN) layer into the conv before it, so each output channel $o$ is scaled, $W'_o = W_o \, \gamma_o / \sqrt{\sigma_o^2 + \epsilon}$, and the BN shift becomes the conv's bias (see bn_folding).
2. Flatten the weight tensor and cut it into groups of `group_size` consecutive values, or use one group for the whole tensor (*per-tensor*).
3. For each group, the scale is $s = \max_i \lvert w_i \rvert / q_{\max}$, where $q_{\max} = 2^{b-1} - 1$ for $b$ bits (127 at 8 bits, 7 at 4 bits).
4. $q = \clip(\round(w / s), -q_{\max}, q_{\max})$ is the integer a real file would store, and $\hat{w} = q \cdot s$ is the float the model uses.
5. At 1 bit there is no zero level: $\hat{w}_i = \sign(w_i) \cdot \frac{1}{n} \sum_j \lvert w_j \rvert$ over the group's $n$ weights.

In one line, for a group at $b \ge 2$ bits:

$$
\hat{w} = s \cdot \clip\!\left(\round\!\left(\frac{w}{s}\right), -q_{\max}, q_{\max}\right), \qquad s = \frac{\max_i \lvert w_i \rvert}{q_{\max}}, \qquad q_{\max} = 2^{b-1} - 1
$$

Example at 4 bits, one group $w = (0.50, -0.20, 0.03)$: $s = 0.5 / 7 \approx 0.0714$, $q = (7, -3, 0)$, $\hat{w} = (0.50, -0.214, 0.0)$. The 0.03 vanished: the largest weight sets the step for the whole group.

## The code, line by line

`fuse(yolo)` in harness/model.py calls Ultralytics' `yolo.model.fuse(verbose=False)` unless `yolo.model.is_fused()`, folding every BN into its conv in place; calling it twice is harmless. It is explicit because newer Ultralytics `predict()` fuses a private copy and leaves `yolo.model` unfused. STATUS.md records that before this fix the cockpit rounded pre-BN weights.

`_pos_levels(bits)` returns `(1 << (bits - 1)) - 1`, so 127 for 8 bits. Dropping the $-128$ level keeps zero exactly in the middle.

`fake_quantize(w, bits, group_size)`:

- `bits >= 16` returns `w.clone()`: the native end of the dial.
- `flat = w.reshape(-1).float()` flattens in row-major order. For a conv weight of shape (out, in, kh, kw), a group walks through one filter, then into the next.
- If `group_size` is `None`, not positive, or at least `n`, `groups = flat.unsqueeze(0)`: one row, one scale.
- Otherwise `pad = (-n) % group_size` zero-pads to a whole number of groups and `groups = flat.reshape(-1, group_size)`.
- Multi-bit branch: `scale = groups.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax`. The clamp stops an all-zero group from dividing by zero. Then `q = torch.clamp(torch.round(groups / scale), -qmax, qmax)` and `deq = q * scale`.
- 1-bit branch: `scale` is the mean absolute value, `q = torch.sign(groups)`, and `torch.where(q == 0, ...)` maps exact zeros to $+1$.
- `out[:-pad]` drops the padding, then shape and dtype are restored. With `group_size == 1` every weight comes back unchanged, a deliberate no-op.

`apply_uniform(model, bits, group_size, skip)` covers the whole network. `iter_weight_modules` yields every module in `QUANTIZABLE = (nn.Conv2d, nn.Linear, nn.Embedding)` that has a weight; biases stay native. For each module whose name contains no `skip` substring, `mod.weight.data.copy_(fake_quantize(mod.weight.data, bits, group_size))` overwrites the weight in place under `@torch.no_grad()`. It returns how many modules it touched. `carve.py` calls `fuse`, then `snapshot` once, then for each bit-width `restore` (weights and biases) followed by `apply_uniform` with `skip=(".dfl.",)`.

The cockpit no longer calls `apply_uniform`. Its Precision module uses `apply_precision` from harness/precision.py, a richer dial: scales per tensor, per output channel, or per group of 32 or 16 weights *inside* one output channel; evenly spaced INT levels or an FP grid (at 4 bits, E2M1, dense near zero); a max-fit or MSE-clipped range; per-stage bit-widths; and storage counted with a 16-bit cost per scale. It always spares the DFL conv.

## What the simulations show

The toy-MLP analogue, not YOLO, is `quantization` (lab/results/quantization.json), using a NumPy twin of `fake_quantize` with groups of 16. The table gives 20-class accuracy after rounding, no retraining. In float, the teacher (3x256) scores 0.9139 and the KD-trained student (2x32) 0.8281.

| bits | teacher, per-tensor | teacher, group-16 | student, per-tensor | student, group-16 |
|---|---|---|---|---|
| 8 | 0.914 | 0.9138 | 0.8276 | 0.8282 |
| 6 | 0.9139 | 0.9133 | 0.8238 | 0.828 |
| 4 | 0.8998 | 0.9128 | 0.7229 | 0.7944 |
| 3 | 0.7766 | 0.897 | 0.4851 | 0.6397 |
| 2 | 0.0707 | 0.6098 | 0.0686 | 0.1683 |

8 bits are free, per-group scales extend the range, then a cliff, and the small student breaks first. Training with the rounding in the loop (QAT) plus KD brought the 2-bit teacher back to 0.91705 but the 2-bit student only to 0.488. The toy's cliff sits lower than the detector's, so don't carry the exact bit-width over.

## When it breaks

- The real curve (README.md, `carve.py`, 125 images, DFL spared): YOLOv8n per-tensor scores 0.4088 at 8 bits, 0.3031 at 6 and 0.0000 at 4, from 0.4284 native; with `--group-size 128`, 0.4263, 0.3905 and 0.0379. YOLO11n collapses at 6-bit per-tensor (0.0136), where the original "6-bit needs groups" finding came from.
- Groups ignore filter boundaries. A $3 \times 3$ conv with 16 inputs has 144 values per filter, so a 128-value group can straddle two filters. precision.py's in-channel groups avoid this.
- `skip` is a substring match: `"model.2"` also matches `model.20`, `model.21` and `model.22`, YOLOv8n's head.
- The DFL conv (it decodes each box edge from 16 bins) holds fixed weights 0 to 15. Sparing it did not rescue 4-bit: group-128 went from 0.0191 to 0.0379, per-tensor stayed 0.0000 (STATUS.md).
- Uniform INT rounding is the weak scheme. In the precision lab, 4-bit with one scale per output channel kept 16% of native mAP; FP4 levels kept 79%; FP4 with stages 0-2 at 8 bits kept 92% at 4.04 bits per weight; MSE clipping cut FP4 from 79% to 18% (README.md, measured on a VOC-2007 bird stand-in set, so read them as relative).
- Activations stay in float; a real int8 deployment also quantizes them.
- File size doesn't change until you pack the weights. The cockpit's size (weight bits plus 16 bits per scale) is an estimate.

## Going further

In the cockpit (`uvicorn server:app --port 8000 --app-dir .`), the Precision module has "Apply 8-bit", a bits ladder from 16 to 1, the scheme knobs above, a weight microscope, a mixed-precision grid with a fragility scan and Auto-mix, and confession curves. Each change posts the recipe to `/api/measure`, which scores the real detector on 40 images. Below about 3.5 bits rounding alone can't save it (README.md); Regrow then fine-tunes with the quantizer in the loop (see qat). Headless, `python carve.py --group-size 128` sweeps the ladder with `fake_quantize`. For a real small file, STATUS.md's next step is packing weights and scales, for example as ONNX QDQ. The toy version is `lab/compress.py:quantize`.
