## The idea

When a transformer generates text, every attention layer stores the *key* and *value* vectors of every past token, so that each new token only computes its own. This store is the *KV cache*, and it grows linearly with context length and batch size. For a 7B model with 32 layers and a 4096-wide hidden state, it takes

$$
\underbrace{2}_{K,\,V} \cdot \underbrace{32}_{\text{layers}} \cdot \underbrace{4096}_{\text{width}} \cdot \underbrace{2\ \text{bytes}}_{\text{fp16}} = 512\ \text{KiB}
$$

per token in fp16, or 2 GiB for one 4096-token sequence. Quantizing it to 2-4 bits fits longer contexts and bigger batches. The key insight: keys and values have different outlier structure, so they need different scale layouts.

## How it works

The recipe follows KIVI; KVQuant's refinements come after.

1. **Keys per channel.** Key vectors have a few fixed channels with large magnitude, the same kind of outlier SmoothQuant deals with. Quantize keys with one scale and zero point *per channel*, computed over a group of $G$ tokens (for example 32).
2. **Values per token.** Values have no fixed outlier channels, and attention mixes them per token. Quantize each token's value vector with its own scale and zero point.
3. **Asymmetric, low-bit.** Quantize each group (a key channel or a value token) with its own minimum and step, at $b$ = 2 to 4 bits, and dequantize to $\hat x$:

$$
\begin{aligned}
\Delta &= \frac{x_{\max} - x_{\min}}{2^b - 1} \\
q &= \round\!\left( \frac{x - x_{\min}}{\Delta} \right) \\
\hat x &= x_{\min} + q\, \Delta
\end{aligned}
$$

4. **Full-precision residual window.** Per-channel key scales need a full group of tokens. Keep the most recent $R$ tokens in fp16, and quantize them once a group fills.
5. **Attention.** Compute attention with $K$ and $V$ dequantized group by group:

$$
\operatorname{Attention}(q, \hat K, \hat V) = \softmax\!\left( \frac{q \hat K^\top}{\sqrt{d}} \right) \hat V
$$

Fused kernels do this on the fly, without materializing fp16 copies.

KVQuant adds:

- quantizing keys *before* the rotary position embedding (RoPE), which otherwise mixes channel pairs and smears the per-channel structure;
- non-uniform levels fitted to sensitivity-weighted data;
- keeping about 1% of outlier values in a sparse fp16 side table;
- keeping the first token (the "attention sink") in full precision.

## The code, line by line

`lab/layerwise.py:quant_act(h, bits)` is the closest code here. It is symmetric, dynamic *per-token* quantization, one scale per row:

- `qmax = qmax_for(bits)`, then `scale = np.maximum(np.abs(h).max(1, keepdims=True), 1e-12) / qmax`.
- `np.clip(np.round(h / scale), -qmax, qmax) * scale` rounds and maps back.

That is already the value-cache layout: `.max(1)` gives one scale per token. For keys, take the max over the token axis instead, `np.abs(K).max(0)` within each group of $G$ tokens, to get one scale per channel. Add a zero point (min and max instead of absmax) for asymmetric ranges. The SmoothQuant page shows why per-token scales fail when a few channels are huge, as in keys.

## When it breaks

- Per-token key quantization: the outlier channels set every token's scale and the small channels round to zero. Attention scores then go wrong.
- 2 bits without a residual window: the most recent tokens, which attention often weights most, lose precision where it matters most.
- Average perplexity can hide failures on long-context retrieval ("needle in a haystack").

## Going further

You need a transformer decoder with a real generation loop and long inputs, for example a 7B model in PyTorch with `transformers` (which ships quantized-cache options) or vLLM (FP8 KV cache). This repo has no attention, no tokens and no cache. `quant_act` and `forward_wa` in `lab/layerwise.py` are the closest analogue: they quantize activations per token on the fly, the same as a value cache, and switching the scale axis gives the key-cache layout.
