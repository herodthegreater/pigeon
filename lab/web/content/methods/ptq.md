## The idea

A trained network stores every weight as a 32-bit float, but it rarely needs that much precision. *Post-training quantization* (PTQ) rounds each weight to one of a small set of evenly spaced levels, so it can be stored as a `b`-bit integer plus a shared *scale* (the float that turns integers back into weights). It needs no data and no training. The knob that decides how low `b` can go is how many weights share one scale. One scale for a whole matrix is set by its largest weight, which leaves too few levels for the typical one. A separate scale for every *group* of 16 weights follows the local size of the weights much more closely.

## How it works

1. Flatten the weight matrix and cut it into groups of `g` consecutive values (or keep one group for the whole tensor).
2. For `b >= 2` bits, symmetric quantization uses the integers `-qmax..qmax` with `qmax = 2^(b-1) - 1`. For 4 bits that is -7..7: 15 levels.
3. Per group: `scale = max|w| / qmax`, then `q = clip(round(w / scale), -qmax, qmax)`.
4. The stored model is `q` (b bits each) plus one scale per group. The weight used at inference is `q * scale`.
5. For `b = 1` there are just two levels: `w_hat = sign(w) * mean|w|` per group.

```python
qmax = 2 ** (bits - 1) - 1
scale = abs(group).max() / qmax
w_hat = clip(round(group / scale), -qmax, qmax) * scale
```

## The code, line by line

`fake_quantize` in `lab/nn.py` is a line-for-line numpy twin of `harness/quant.py:fake_quantize`, the function pigeon runs on YOLOv8n. Both return floats ("fake" quantization): the values are exactly the ones a `b`-bit format could store, but kept in float, so measured accuracy matches a real quantize-then-dequantize export.

`if bits >= 16: return w` is the native setting. `flat = w.reshape(-1)` flattens row by row. `lab/nn.py` stores `W` as (inputs, outputs), so a group of 16 is 16 weights leaving one input towards 16 consecutive neurons. If `group_size` is `None`, or at least the tensor size, `g = flat[None, :]` makes one group. Otherwise `pad = (-n) % group_size` zeros are appended so the tensor splits evenly, and `reshape(-1, group_size)` makes one row per group.

For 1 bit, `scale = np.maximum(np.abs(g).mean(1, keepdims=True), 1e-12)` and `q = np.where(g >= 0, 1.0, -1.0)`, so even an exact zero becomes `+scale`. The torch version maps zeros to +1 explicitly, with the same result. For more bits, `qmax = (1 << (bits - 1)) - 1`, `scale = max|g| / qmax`, and `q = np.clip(np.round(g / scale), -qmax, qmax)`. The `1e-12` floor stops an all-zero group from dividing by zero. `out = (q * scale)` is reshaped back, the padding is cut off, and `astype(w.dtype)` keeps float32.

`quantize` in `lab/compress.py` copies the net, sets `net.qbits, net.qgroup`, and bakes the result in with `net.W = [net.eff(i) for i in range(net.n_layers)]`, since `MLP.eff` applies `fake_quantize` when `qbits` is set. It then clears `qbits`, so the returned net is an ordinary float MLP whose weights happen to lie on the grid.

One consequence of the symmetric grid is easy to miss. At 2 bits, `qmax = 1`, so the levels are only `{-scale, 0, +scale}`, and every weight smaller than half the group maximum becomes zero. The `params` column counts nonzero weights and shows this directly.

## What the simulations show

From `quantization`, accuracy on 20 classes. The "teacher" is the 3x256 net (140,052 params). The "student" is a 2x32 net (2,132 params) distilled from it.

| bits | teacher, per-tensor | teacher, group-16 | student, per-tensor | student, group-16 |
|---|---|---|---|---|
| fp32 | 0.9139 | 0.9139 | 0.8009 | 0.8009 |
| 8 | 0.9140 | 0.9138 | 0.8010 | 0.8014 |
| 6 | 0.9139 | 0.9133 | 0.7966 | 0.7965 |
| 4 | 0.8998 | 0.9128 | 0.7186 | 0.7670 |
| 3 | 0.7766 | 0.8970 | 0.4461 | 0.6283 |
| 2 | 0.0707 | 0.6098 | 0.0764 | 0.2173 |
| 1 | 0.4013 | 0.4356 | 0.1662 | 0.1767 |

8 and 6 bits are free for both models. Below that, group scales are worth a lot (teacher at 3 bits: 0.7766 → 0.8970), and the small student degrades much earlier than the big teacher. Per-tensor 2-bit is worse than 1-bit. At 2 bits only 3,616 of the teacher's 140,052 weights stay nonzero, while 1-bit keeps every weight as ±mean.

Scales are not free. The experiment's storage count adds an fp16 scale per group, so the teacher at 4 bits takes 582.3 kbit per-tensor and 721.5 kbit with group-16, against 4,481.7 kbit in fp32. A group of 16 adds one bit per weight, and a group of 4 adds four.

Page runs: the default (teacher, 4-bit, group 16) gives 0.9128. Group 4 at 2 bits gives 0.8057, but with four extra bits of scale per weight.

pigeon's `carve.py` on YOLOv8n found the same shape with a lower floor: 8-bit about free, 6-bit alive only with group scales (group size 128), 4-bit collapsed. The toy teacher has more slack than YOLOv8n. The small student, with little spare capacity, is the closer analogue.

## When it breaks

- **Outliers.** One large weight sets `scale` for its whole group and pushes everything else towards zero. This is why per-tensor scales fail first.
- **2-bit symmetric is ternary.** Most weights become zero, which is effectively heavy pruning.
- **Small models have no slack.** Each weight carries more of the function, so rounding errors are not averaged away.
- **Group overhead.** Very small groups cost more in scales than they save in bits.
- **Activations are not quantized here.** Real integer inference also rounds activations, which has its own outlier problem (see SmoothQuant).

## Try this

- Set **scale shared by** to tensor and read the bits chart. Then switch to 16. Find the lowest **bits** that stays within 0.01 of fp32 in each case.
- Set **bits** to 2 and compare the `params` column for tensor and 4. The nonzero count shows how many weights rounded to zero.
- Switch **model** to student and repeat the first experiment. The cliff moves up by about two bits.

## Going further

Real deployments store integers (int8, int4) with per-channel or per-group fp16 scales, and quantize activations too, using a calibration set to pick their ranges. For 4 bits and below on large models, layer-wise methods that correct rounding error with calibration data (GPTQ, AWQ pages) beat round-to-nearest, and quantization-aware training (QAT page) beats both when you can afford training. In pigeon, the cockpit's Precision module applies `harness/quant.py:fake_quantize` to every `Conv2d` (via `harness/model.py`), so accuracy measured in the cockpit is what an export with the same scheme would give.
