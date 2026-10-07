## The idea

Quantizing a weight matrix means storing each weight as a small integer times a shared floating-point *scale*. Round-to-nearest (RTN) picks the scale from the largest weight in a row and rounds every weight to the nearest integer level. It looks at nothing but the weights themselves: no data, no feedback. Every LLM quantization paper reports it as the baseline: the gap between RTN and a smarter method is the argument of those papers.

## How it works

For a linear layer `y = W x` with `W` shaped (out, in), and a bit width `b`:

1. Compute the largest representable integer: `qmax = 2^(b-1) - 1`. At 4 bits that is 7, at 3 bits 3, at 2 bits 1.
2. For each output row `r` (or each run of `g` consecutive input columns within the row, if you use groups), set `scale = max|W[r, :]| / qmax`.
3. Replace each weight with `clip(round(w / scale), -qmax, qmax) * scale`.
4. Store the integers and one scale per row (or per group).

This is *symmetric* quantization: levels centered on zero, no zero-point offset. At 2 bits, `qmax = 1` leaves three levels, `{-scale, 0, +scale}`, so one of the four codes is unused.

In a transformer, the same rule is applied to every `nn.Linear` inside every block: the query, key, value and output projections of attention, and the up, gate and down projections of the MLP. The token embedding and the output layer (`lm_head`) usually stay at 16 bits. RTN needs no calibration data.

## The code, line by line

`quant_rows(Wt, bits, group)` does all the work. `Wt` is the weight in the papers' (out, in) orientation; `apply_layerwise` transposes the MLP's stored (in, out) matrix before calling it.

- `qmax = qmax_for(bits)` computes `(1 << (bits - 1)) - 1`, the formula in step 1.
- `g = group or din`: without a group size, one group spans the whole row, so there is one scale per output row.
- `for c0 in range(0, din, g)` walks over column groups. `blk = Wt[:, c0:c0 + g]` is every row's slice of that group.
- `scale = np.maximum(np.abs(blk).max(1, keepdims=True), 1e-12) / qmax` gives one scale per row of the block. The `1e-12` floor keeps an all-zero row from dividing by zero.
- `np.clip(np.round(blk / scale), -qmax, qmax) * scale` rounds, clips and maps back to real values. The function returns *dequantized* weights ("fake quantization"), so the network still runs in float and the accuracy loss is measured directly.

`rtn(Wt, X, ...)` is a wrapper with the same signature as `gptq` and `awq`, so all three can be passed to `apply_layerwise`. `del X` says it plainly: RTN never looks at the data.

## What the simulations show

The model is the 3x256 teacher (91.39% test accuracy). All four of its linear layers are quantized, including the first (12 inputs) and the 20-way output layer. The `params` column counts *nonzero* weights, so it shows how many weights rounded to zero.

| setting | accuracy | nonzero params |
|---|---|---|
| teacher, fp32 | 0.9139 | 140,052 |
| RTN 8-bit, per row | 0.9138 | 138,657 |
| RTN 4-bit, per row | 0.9078 | 115,504 |
| RTN 3-bit, per row | 0.8669 | 85,152 |
| RTN 3-bit, scale per 16 | 0.8947 | 102,263 |
| RTN 2-bit, per row | 0.2483 | 19,371 |

8 and 4 bits are nearly free. At 3 bits RTN loses almost 5 points; at 2 bits the network collapses and roughly one weight in seven stays nonzero. Smaller groups help because one large weight no longer sets the step size for 256 others. The toy world can't show the main reason RTN fails in real LLMs: a few input channels with activations 100 times larger than the rest (see AWQ and SmoothQuant).

## When it breaks

- One outlier weight per row stretches the scale, so every other weight in that row gets coarser levels. Groups limit the damage, but each group needs its own stored scale: one 16-bit scale per 16 weights adds 1 bit per weight.
- Rounding errors in a row add up in the output. RTN never checks whether they cancel or pile up on the inputs that matter.
- Below 4 bits, symmetric rounding with a zero level sends most small weights to exactly zero. That is accidental pruning.

## Try this

- Set **bits** to 2, 3 and 4 with **scale per** = row. Look for the cliff between 3 and 2, and watch the nonzero-params column fall with it.
- Keep **bits** at 3 and switch **scale per** from row to 64 to 16. Look for how much of the 3-bit loss comes from the scale alone.
- Open the GPTQ page with the same settings. Its table includes this RTN row, so you can read off how much error compensation buys back.

## Going further

LLM.int8() (Dettmers et al.) showed that per-row 8-bit RTN ("vector-wise" quantization) is accurate up to about 6.7B parameters. Above that, outlier feature dimensions appear and break it. Their fix keeps those few dimensions in 16 bits. Production 4-bit formats (GPTQ and AWQ checkpoints, llama.cpp's Q4 variants) all start from RTN with groups of 32 to 128, then improve the rounding or the scale. In this repo the same rounding lives in `lab/nn.py:fake_quantize`, which uses one scale per whole matrix, or per run of consecutive entries in the flattened (in, out) matrix, rather than one per output row. So the `quantization` experiment's numbers are not directly comparable to these.
