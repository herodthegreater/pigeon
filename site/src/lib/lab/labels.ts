/**
 * Words for the lab's result keys. LABELS mirrors `LABELS` in lab/report.py,
 * the axis labels drawn on the figures, so a table column and the figure axis
 * that plots it read the same. results.test.ts fails if the two drift apart.
 */

export const LABELS: Record<string, string> = {
  acc: "test accuracy (20-way)",
  target_ap: "target-class AP ('bird')",
  fidelity: "agreement with teacher",
  nll: "test NLL",
  kl_bayes: "KL(Bayes || model)",
  params: "parameters (nonzero)",
  macs: "MACs / example (nonzero)",
  kbits: "storage (kbit)",
  teacher_acc: "teacher accuracy",
  teacher_kl_bayes: "teacher KL(Bayes || teacher)",
  width: "hidden width",
  sparsity: "fraction of weights removed",
  bits: "bits / weight",
  n_labels: "labeled examples",
  steps: "training steps",
  T: "temperature T",
  noise: "label-noise rate",
  teacher_width: "teacher width",
  dropped: "layers removed (of 8)",
  start_width: "width trained before pruning",
  generation: "generation",
  n_teachers: "teachers in ensemble",
  smoothing: "teacher label smoothing",
  rank_or_width: "rank / width",
  overlap: "overlap",
};

/** Keys the lab measures but never plots, worded the same way. */
const EXTRA: Record<string, string> = {
  ensemble_acc: "ensemble accuracy",
  teacher_nll: "teacher NLL",
  teacher_ece: "teacher ECE",
  dense_width: "dense width",
};

/** The lab's metrics, in the order lab/report.py puts them in its tables. */
export const METRICS = ["acc", "target_ap", "fidelity", "nll", "ece", "kl_bayes", "teacher_acc", "params", "macs", "kbits", "overlap"];

export const label = (k: string) => LABELS[k] ?? EXTRA[k] ?? k.replace(/_/g, " ");
export const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
