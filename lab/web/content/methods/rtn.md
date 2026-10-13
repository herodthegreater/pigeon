## The idea

Quantizing a weight matrix means storing each weight as a small integer times a shared floating-point *scale*. Round-to-nearest (RTN) picks the scale from the largest weight in a row and rounds every weight to the nearest integer level. It looks at nothing but the weights themselves: no data, no feedback. Every LLM quantization paper reports it as the baseline: the gap between RTN and a smarter method is the argument of those papers.

## How it works

For a linear layer $y = Wx$ with $W$ shaped (out, in), and a bit width $b$:

1. Compute the largest representable integer: $q_{\max} = 2^{b-1} - 1$. At 4 bits that is 7, at 3 bits 3, at 2 bits 1.
2. For each output row $r$ (or each run of $g$ consecutive input columns within the row, if you use groups), set the scale $s_r = \max_c \lvert W_{rc} \rvert / q_{\max}$.
3. Replace each weight $w$ with $s_r \cdot \clip(\round(w / s_r), -q_{\max}, q_{\max})$.
4. Store the integers and one scale per row (or per group).

Written out for row $r$ and input column $c$, steps 2 and 3 are

$$
\begin{aligned}
s_r &= \frac{\max_c \lvert W_{rc} \rvert}{q_{\max}}, \\
\hat W_{rc} &= s_r \cdot \clip\!\left( \round\!\left( \frac{W_{rc}}{s_r} \right),\, -q_{\max},\, q_{\max} \right).
\end{aligned}
$$

This is *symmetric* quantization: levels centered on zero, no zero-point offset. At 2 bits, $q_{\max} = 1$ leaves three levels, $\{-s_r, 0, +s_r\}$: "2-bit" here is really ternary, by design and the same as `harness/quant.py`. One of the four codes is unused; real 2-bit formats add a zero point to use all four.

In a transformer, the same rule is applied to every `nn.Linear` inside every block: the query, key, value and output projections of attention, and the up, gate and down projections of the MLP. The token embedding and the output layer (`lm_head`) usually stay at 16 bits. RTN needs no calibration data.

## The code, line by line

`quant_rows(Wt, bits, group)` does all the work. `Wt` is the weight in the papers' (out, in) orientation; `apply_layerwise` transposes the MLP's stored (in, out) matrix before calling it.

- `qmax = qmax_for(bits)` computes `(1 << (bits - 1)) - 1`, which is $2^{b-1} - 1$, the formula in step 1.
- `g = group or din`: without a group size, one group spans the whole row, so there is one scale per output row.
- `for c0 in range(0, din, g)` walks over column groups. `blk = Wt[:, c0:c0 + g]` is every row's slice of that group.
- `scale = np.maximum(np.abs(blk).max(1, keepdims=True), 1e-12) / qmax` gives one scale per row of the block, $s_r = \max_c \lvert W_{rc} \rvert / q_{\max}$ over the group's columns. The `1e-12` floor keeps an all-zero row from dividing by zero.
- `np.clip(np.round(blk / scale), -qmax, qmax) * scale` rounds, clips and maps back to real values. The function returns *dequantized* weights ("fake quantization"), so the network still runs in float and the accuracy loss is measured directly.

`rtn(Wt, X, ...)` is a wrapper with the same signature as `gptq` and `awq`, so all three can be passed to `apply_layerwise`. `del X` says it plainly: RTN never looks at the data.

## What the simulations show

The model is the 3x256 teacher (0.9150 test accuracy). As in the papers, the output layer (the toy's `lm_head`) is left at full precision: `apply_layerwise(..., skip_last=True)`. The first layer, which reads the 12 raw inputs, *is* quantized, whereas papers usually keep the embedding at 16 bits. The `params` column counts *nonzero* parameters, so it shows how many weights rounded to zero.

| setting | accuracy | nonzero params |
|---|---|---|
| teacher, fp32 | 0.9150 | 140,052 |
| RTN 8-bit, per row | 0.9146 | 138,738 |
| RTN 4-bit, per row | 0.9087 | 116,376 |
| RTN 3-bit, per row | 0.8971 | 87,183 |
| RTN 3-bit, scale per 16 | 0.9007 | 103,706 |
| RTN 2-bit, per row | 0.4796 | 23,815 |

8 and 4 bits are nearly free. At 3 bits RTN loses about 2 points; at 2 bits (ternary) accuracy halves and only 23,815 of 140,052 parameters stay nonzero. Smaller groups help because one large weight no longer sets the step size for 256 others. The toy world can't show the main reason RTN fails in real LLMs: a few input channels with activations 100 times larger than the rest (see AWQ and SmoothQuant).

## When it breaks

- One outlier weight per row stretches the scale, so every other weight in that row gets coarser levels. Groups limit the damage, but each group needs its own stored scale: one 16-bit scale per 16 weights adds 1 bit per weight.
- Rounding errors in a row add up in the output. RTN never checks whether they cancel or pile up on the inputs that matter.
- Below 4 bits, symmetric rounding with a zero level sends most small weights to exactly zero. That is accidental pruning, and at 2 bits it is the dominant effect.

## Try this

- Set **bits** to 2, 3 and 4 with **scale per** = row. Look for the cliff between 3 and 2, and watch the nonzero-params column fall with it.
- Keep **bits** at 3 and switch **scale per** from row to 64 to 16. Look for how much of the 3-bit loss comes from the scale alone.
- Open the GPTQ page with the same settings. Its table includes this RTN row, so you can read off how much error compensation buys back.

## Going further

LLM.int8() (Dettmers et al.) showed that per-row 8-bit RTN ("vector-wise" quantization) is accurate up to about 6.7B parameters. Above that, outlier feature dimensions appear and break it. Their fix keeps those few dimensions in 16 bits. Production 4-bit formats (GPTQ and AWQ checkpoints, llama.cpp's Q4 variants) all start from RTN with groups of 32 to 128, then improve the rounding or the scale. In this repo the same rounding lives in `lab/nn.py:fake_quantize`, which uses one scale per whole matrix, or per run of consecutive entries in the flattened (in, out) matrix, rather than one per output row. So the `quantization` experiment's numbers are not directly comparable to these.
