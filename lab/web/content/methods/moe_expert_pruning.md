## The idea

In a mixture-of-experts (MoE) transformer, each MLP is replaced by `E` parallel MLPs, the *experts*, plus a *router* that sends each token to its top `k` experts (Mixtral: 8 experts, top 2). All experts must sit in memory even though each token uses only two. For a narrow task, some experts are rarely chosen or barely matter. Removing them is structured pruning where each "neuron" is a whole sub-network; each removed expert frees `1/E` of the layer's expert memory.

## How it works

An MoE layer computes `y = Σ_{i in topk(g)} g_i(x) · E_i(x)`, where `g = softmax(W_r x)` gives the router weights.

Lu et al. (2024) prune *layer by layer, by reconstruction*:

1. Run calibration data (generic text, or your task's data for a task-specific model) and record each MoE layer's input `X` and output `Y`.
2. For each layer, enumerate every subset `S` of `r` experts to keep. For 8 choose 6 that is 28 subsets.
3. For each subset, recompute the layer with the router restricted to `S`: renormalize the top-k over `S` only. Measure `|Y_S - Y|²`.
4. Keep the subset with the smallest error. Delete the other experts' weights and their rows in `W_r`.
5. Move to the next layer.

The same paper also *skips* experts dynamically at inference: if the second expert's router weight is small relative to the first (`g_2 < β · g_1`), only the first runs.

Cheaper scores when there are too many experts to enumerate:

- **frequency**: how often the router picks each expert on task data;
- **mean gate weight**: the average `g_i` when the expert is picked;
- **output norm**: `mean |g_i · E_i(x)|`, the expert's actual contribution.

**Merging** is the softer alternative: average the weights of experts with similar outputs, and route their tokens to the merged expert.

## The code, line by line

`lab/compress.py:neuron_importance(net, X, y, method)` scores the hidden neurons of the toy MLP. It is the one-neuron version of expert scoring:

- `method="act"`: `np.abs(h).mean(0)` is the mean absolute activation per neuron, the analogue of expert output norm.
- `method="taylor"`: `np.abs(h * g).sum(0)`, activation times gradient. It is a first-order estimate of the loss change from removing the neuron. With `group`, the loss is restricted to the task's label space, which is the analogue of using task-specific calibration data.
- `method="l1"`: the weight norm, a data-free score.

`structured_prune` then keeps the top-scoring neurons with `keep_neurons`, which deletes the neuron's incoming and outgoing weights, just as removing an expert deletes its MLP and its router row.

## When it breaks

- Load-balancing losses spread tokens evenly across experts, so frequency alone says little.
- The remaining experts receive tokens they were never trained on. Without fine-tuning, quality falls off as more experts go.
- Task-specific pruning trades away general ability. The pruned model can fail outside the calibration domain.
- Shared experts (DeepSeek-MoE style), used by every token, are not candidates.

## Going further

You need an MoE model (Mixtral 8x7B, Qwen-MoE, DeepSeek-MoE), access to its router logits, and calibration text from your task. Hooks on each MoE block record inputs and outputs; pruning edits the expert list and the router's output rows. This repo has no MoE layer. The closest analogues are `neuron_importance` and `structured_prune` in `lab/compress.py`, and the `specialist_prune` experiment (`lab/experiments/specialist.py`), which carves a task-specific network out of a general one.
