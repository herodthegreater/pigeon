## The idea

Fine-tuning every weight of a large model needs memory for the weights, their gradients and two Adam moment buffers, several times the model's size. LoRA (low-rank adaptation) freezes the weights `W0` and learns only a low-rank correction `ΔW = A B`, with `A` and `B` thin matrices. QLoRA also stores the frozen `W0` in 4 bits. Here the same trick is used for *repair*: the adapters are trained to undo the damage from heavy quantization.

## How it works

For each linear layer, with weight stored as (in, out) as in this repo:

1. Quantize the original weights once: `Wq = Q(W0)`. Freeze them.
2. Add two trainable matrices: `A` (in x r), random, and `B` (r x out), zeros. The layer's weight becomes `W = Wq + A B`. Since `B = 0`, training starts exactly at the quantized model.
3. Forward and backward as usual, giving `dL/dW` for each layer.
4. Since `W = Wq + A B`, the chain rule gives `dL/dA = dL/dW · Bᵀ` and `dL/dB = Aᵀ · dL/dW`. Update only `A` and `B`.
5. Afterwards, either keep `A`, `B` separate (base stays 4-bit, plus one small extra matmul) or merge `Wq + A B` into a dense weight.

Adapter size per layer is `r · (in + out)` instead of `in · out`. For a 4096 x 4096 projection at `r = 16`, that is 131,072 instead of 16.8 million.

Note that `dL/dA = dL/dW · Bᵀ` is zero at the start, because `B = 0`. The first steps move only `B`; once `B` is nonzero, `A` starts learning too.

**What QLoRA adds over LoRA** (Dettmers et al. 2023):

- **NF4 (4-bit NormalFloat).** The 16 levels are quantiles of a standard normal, matching how pretrained weights are distributed. Each block of 64 weights has its own absmax scale.
- **Double quantization.** Those per-block scales are themselves quantized to 8 bits, saving about 0.37 bits per weight.
- **Paged optimizers.** Optimizer states spill to CPU memory during memory spikes.
- **Adapters on every linear layer.** The paper found this necessary to match full 16-bit fine-tuning.

The frozen weights are dequantized to bf16 on the fly for each matmul. Gradients flow *through* them to the adapters but are never applied to them.

## The code, line by line

`qlora(net, teacher, X, bits, group, rank, steps, lr, bs, seed)`:

- `base = quantize(net, bits, group)` bakes the quantized weights in. This uses `lab/compress.py:quantize`: symmetric uniform integers, one scale per 16 consecutive entries of the flattened (in, out) matrix.
- `A = [rng.standard_normal((w.shape[0], rank)) / np.sqrt(w.shape[0]) ...]` and `B = [np.zeros((rank, w.shape[1])) ...]` are step 2, one pair per layer. Paper notation writes `ΔW = B A` with `W` as (out, in); this is the transpose of the same thing.
- `opt = Adam(A + B, lr=lr)` hands the optimizer only the adapter matrices, so nothing else can change.
- Each step rebuilds `work.W[i] = base.W[i] + A[i] @ B[i]`, runs `work.forward`, and gets `dz` from `KD(T=2.0)`, distillation from the 3x256 teacher's outputs. Then `work.backward` returns `dW`, the gradient for every full weight matrix.
- `[g @ Bi.T ...] + [Ai.T @ g ...]` is step 4: `dA = dW Bᵀ`, `dB = Aᵀ dW`.
- The function returns the merged model and the adapter parameter count.

**Simplifications.** No NF4 (uniform INT with group 16 instead of block 64), no double quantization, no paged optimizer. No LoRA scaling factor `α/r`, which only rescales the learning rate. Biases are frozen at full precision. The training signal is distillation on the 20,000 unlabeled pool inputs for 500 steps, not supervised instruction tuning.

## What the simulations show

The starting model is the distilled 2x32 student (2,132 parameters, 0.8009 accuracy).

| base bits | quantized | + rank-4 adapters (640 params) |
|---|---|---|
| 2 | 0.2173 | 0.6229 |
| 3 | 0.6283 | 0.7788 |
| 4 | 0.7670 | 0.7979 |

At 2 bits, rank 1 (160 parameters) reaches 0.3284 and rank 16 (2,560 parameters) reaches 0.7571. Rank 16 has more adapter parameters than the model has weights. In a toy this small, "low-rank" stops being cheap quickly; in a 7B model even rank 64 on every layer is a small fraction of the weights. At 4 bits the adapters recover almost everything. At 2 bits they recover most of the drop but not all.

The merged model is no longer on the quantization grid: `Wq + A B` is a dense float matrix (the `params` column shows 2,132). To keep the memory savings at inference you keep the adapters separate, or re-quantize the merged weights and accept some loss.

## When it breaks

- If the base is quantized too coarsely, a low-rank correction can't undo errors that are full-rank. Rounding noise is spread over every direction.
- Training data that doesn't match the target task gives adapters that fix the wrong thing.
- Merging and re-quantizing can throw away much of what the adapters learned, because their corrections are often smaller than one quantization step.
- Learning rate and rank interact: without `α/r` scaling, changing the rank also changes the effective step size.

## Try this

- **base bits** 2, then **adapter rank** 1, 4, 16: look for diminishing returns, and compare the adapter count with the model's 2,132 parameters.
- **base bits** 4, **adapter rank** 4: almost full recovery.
- **base bits** 3 with **adapter rank** 1: see whether a single direction per layer is enough when the damage is milder.

## Going further

QLoRA fine-tuned a 65B model on a single 48 GB GPU. Its Guanaco models were close to ChatGPT on the Vicuna benchmark, as judged by GPT-4. For running it yourself, the `bitsandbytes` library provides NF4 layers and `peft` provides the adapters. LoftQ and similar methods start `A` and `B` from an SVD of `W0 - Q(W0)` instead of zero, so training begins closer to the original model. In this repo, `lab/compress.py:low_rank` is the same low-rank factorization used for compression instead of correction.
