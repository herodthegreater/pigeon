## The idea

A language model's vocabulary sets the size of two matrices: the input embedding (one row per token) and the output layer, or *LM head* (one row per token, producing one logit each). Multilingual models have 100k-250k tokens, so these matrices can be a large share of a small model. A model used for one language or domain only sees and emits a fraction of them; vocabulary trimming deletes the other rows. For inputs made of kept tokens, the hidden states are unchanged, and the output differs only by renormalization.

## How it works

1. **Count.** Tokenize a target-language or target-domain corpus with the model's tokenizer and count each token id.
2. **Choose the kept set $K$.** All special tokens (padding, end-of-sequence, unknown), plus either every token seen or the top $N$ by frequency.
3. **Slice.** $E' = E_K$ for the embedding (rows), $W' = W_{\text{out},K}$ for the LM head, and $b' = b_K$ if there is a bias. With tied embeddings, slice the shared matrix once.
4. **Remap ids.** Build the map $\text{old id} \mapsto \text{new id}$ for $K$ and wrap the tokenizer so its output ids pass through the map. Tokens outside $K$ map to the unknown token. Generation maps back with the inverse.
5. **Optionally fine-tune.** Ushio et al. compare trimming before fine-tuning with trimming after it.

What stays exact: every logit for a kept token is identical to the original. The softmax is now taken over $K$ only, so kept-token probabilities rise by the factor $1 / \sum_{t \in K} p(t)$, but their ranking is unchanged. With logits $z$ and original $p(t) = e^{z_t} / \sum_{t'} e^{z_{t'}}$, for a kept token $t \in K$:

$$
p'(t) = \frac{e^{z_t}}{\sum_{t' \in K} e^{z_{t'}}} = \frac{p(t)}{\sum_{t' \in K} p(t')} .
$$

Greedy decoding is identical whenever the original argmax was in $K$.

## The code, line by line

`lab/compress.py:collapse_head(net, G)` is the same surgery on the toy MLP's output layer. `G` is a ($C \times K$) 0/1 matrix saying which old classes go into each new output.

- `norm = G / G.sum(0, keepdims=True)` turns each column into an average over its member classes.
- `net.W[-1] = net.W[-1] @ norm` and `net.b[-1] = net.b[-1] @ norm` build each new output's weights and bias as the mean of its members'.

For vocabulary trimming, `G` is a *selection*: column $k$ has a single 1, at the kept token's old index, and removed tokens have no 1 anywhere. Then `@ norm` is exact slicing, as the docstring says: "Exact for singleton groups". The toy model has no input embedding; a real one is sliced the same way.

## When it breaks

- A token not seen in the counting corpus becomes unknown at inference: rare names, code, or text in a second language.
- The model can no longer *generate* removed tokens, even when they would have been the right answer.
- Keep the byte-fallback tokens, or unseen words become unknown tokens instead of byte sequences.
- Savings are large only when the vocabulary is a big part of the model; for a 70B model they are small.

## Going further

You need a pretrained language model with its tokenizer, and a corpus representative of the deployment language or domain. With `transformers`, slice `model.get_input_embeddings().weight` and `model.lm_head.weight` by $K$ and wrap the tokenizer. This repo has no tokens. Its closest analogue is `collapse_head`, used by the `specialist_prune` and `specialist_data` experiments (`lab/experiments/specialist.py`) to shrink a 20-class head to the target task: the same "keep only the outputs this task needs" decision, for classes instead of tokens.
