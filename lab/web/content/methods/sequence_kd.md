## The idea

A language model produces a sequence one token at a time. Ordinary (word-level) distillation trains the student to match the teacher's next-token distribution at every position of a human-written sequence. Kim and Rush argued that what should be copied is the teacher's distribution over *whole sequences*, and used its mode as a stand-in: run the teacher's beam search on the training inputs and fine-tune the student on those outputs with plain cross-entropy. On-policy distillation (GKD) goes one step further: the student generates the sequences and the teacher grades them.

## How it works

**Sequence-level KD** (Kim & Rush 2016):

1. For each training source $x$ (say, a sentence to translate), decode the teacher with beam search (beam size about 5) and keep the top hypothesis $\hat{y}$.
2. Replace the reference target $y$ with $\hat{y}$.
3. Train the student by ordinary cross-entropy on $(x, \hat{y})$, summed over the tokens of $\hat{y}$.
4. Variants: combine it with word-level KD, a KL term at every position, or train on the beam hypothesis closest to $y$ instead of the top one (*sequence-level interpolation*).

With $p_t$ the teacher, $p_s$ the student, and $i$ indexing the $L$ tokens of a sequence, beam search approximates the teacher's most probable sequence $\hat{y}$, and the two losses are:

$$
\begin{aligned}
\hat{y} &\approx \argmax_{y} \, p_t(y \mid x) \\
\mathcal{L}_{\text{seq}} &= -\sum_{i=1}^{L} \log p_s\big( \hat{y}_i \mid \hat{y}_{<i}, x \big) \\
\mathcal{L}_{\text{word}} &= \sum_{i=1}^{L} \KL\big( p_t(\cdot \mid y_{<i}, x) \\
&\qquad\qquad \,\|\, p_s(\cdot \mid y_{<i}, x) \big)
\end{aligned}
$$

**On-policy distillation** (GKD, Agarwal et al. 2024):

1. With probability $\lambda$, sample $\hat{y} \sim p_s(\cdot \mid x)$ from the current student; otherwise take a fixed sequence (from the dataset or the teacher).
2. Run the teacher on $(x, \hat{y})$ to get its next-token distribution at every position, all in one forward pass.
3. Minimize a token-level divergence between teacher and student along $\hat{y}$: forward KL, reverse KL, or generalized Jensen-Shannon. No gradient flows through the sampling.
4. Repeat. The student is corrected on the states it actually reaches at inference time, which removes the train/test mismatch known as *exposure bias*.

For on-policy samples the objective is

$$
\begin{aligned}
\mathcal{L}_{\text{GKD}} &= \E_{\hat{y} \sim p_s(\cdot \mid x)} \Big[ \sum_{i=1}^{L} D_i \Big] \\
D_i &= D\big( p_t(\cdot \mid \hat{y}_{<i}, x) \\
&\qquad \,\|\, p_s(\cdot \mid \hat{y}_{<i}, x) \big)
\end{aligned}
$$

where $D$ is forward KL, reverse KL, or the generalized Jensen-Shannon divergence

$$
\begin{aligned}
\operatorname{JSD}_\beta(p \,\|\, q) &= \beta \, \KL(p \,\|\, r) + (1 - \beta) \, \KL(q \,\|\, r) \\
r &= \beta p + (1 - \beta) q
\end{aligned}
$$

## The code, line by line

There is no sequence model here, so `lab/train.py:KD` is the per-example analogue:

- `kind="kl"` is word-level KD at a single position: $T^2 \cdot \KL(p_t \,\|\, p_s)$ on probabilities softened by $T$, with gradient $T (p_s - p_t) / N$ over a batch of $N$.
- `kind="hard"` turns the teacher's distribution into a one-hot of its argmax, `onehot(p.argmax(1), ...)`, and trains with cross-entropy. That is sequence-level KD with sequences of length one: the teacher's mode, used as a label.
- `kind="rkl"` is reverse KL, $\KL(p_s \,\|\, p_t)$, a mode-seeking divergence that GKD found useful on some generation tasks.

## When it breaks

- Beam outputs are lower-entropy than human text. The student inherits that blandness and repetitiveness, and loses calibration.
- Teacher errors become hard labels, with no signal about how confident the teacher was.
- On-policy sampling is expensive: every step needs student generation and a teacher forward pass.
- A weak student's early samples are garbage, so the teacher's feedback on them teaches little. Mixing in fixed sequences ($\lambda < 1$) helps.

## Going further

To run this you need an autoregressive teacher and student that share a tokenizer, a decoding loop (beam search or sampling), and a corpus of source texts. A small pair such as a 1B and a 7B open model, with the `transformers` library, is enough. This repo has none of these: its models classify a single 12-D input. The closest analogue is the `loss_zoo` experiment (`lab/experiments/distill.py`), which compares hard pseudo-labels, soft KL and reverse KL on the toy world. The per-position `KD` loss can be applied unchanged at each token of a teacher- or student-generated sequence.
