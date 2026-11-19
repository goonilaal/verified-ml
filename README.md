# verified-ml

Classical machine learning classifiers, defined in Lean 4 — towards
verified machine learning.

This is the "definitions first" scaffold for a proof-of-work project in
the spirit of [tiny-bayes](https://github.com/leochlon/tiny-bayes)
(Pythia-14M defined exactly in Lean 4). The idea is the same: take a
trained model, freeze its weights, define its prediction function
exactly, and then prove things about that fixed function. No training
procedure is formalised here — only the trained models.

## What is defined

All points live in `ℝⁿ`, represented as `Fin n → ℝ`. Labels are an
arbitrary type with decidable equality.

- **SVM** (`VerifiedML/SVM.lean`) — a linear SVM is a weight vector
  plus a bias. Decision function `w ⬝ x + b`, prediction is its sign,
  geometric margin `|w ⬝ x + b| / ‖w‖`, and the RBF kernel
  `exp(-γ ‖x - y‖²)` via `Real.exp`.
- **Random forest** (`VerifiedML/RandomForest.lean`) — a decision tree
  as an inductive type (leaf with a label; node with a feature index,
  a threshold, and two subtrees). A forest is a list of trees;
  prediction is majority vote over the trees.
- **k-NN** (`VerifiedML/KNN.lean`) — Euclidean distance, the `k`
  nearest neighbours of a query point (sorted by distance, first `k`
  taken), prediction by majority vote.
- **Worked example** (`VerifiedML/Examples.lean`) — a toy 2-D SVM with
  weights `(1, 0)` and bias `-1`, chosen by hand for four separable
  points (the line `x₁ = 1` separates the classes), with its decision
  values, predictions, margin, and a stability radius proved.
- **nanoGPT** (`VerifiedML/NanoGPT.lean`) — Karpathy's GPT, defined
  exactly over ℝ: token + learned position embeddings, pre-LayerNorm
  blocks with sequential residuals, causal multi-head attention
  (scores `q ⬝ k / √head_dim`, softmax over the prefix), MLP with
  exact GELU (`x * Φ x`), final LayerNorm, `lm_head`. Inference mode
  (dropout is the identity); `lm_head` is a separate matrix, so the
  definition covers both the weight-tied and untied cases.
- **Tiny nanoGPT** (same file) — a miniature config (vocab 8, block
  size 4, 1 layer, 2 heads, width 8) with hand-chosen weights.

Shared helpers (majority vote) live in `VerifiedML/Common.lean`.

## What is proved

Proved (checked by the Lean kernel, no `sorry`):

- SVM: a point with a positive decision value is predicted `1`
  (`LinearSVM.predict_eq_one_of_pos`); negative gives `-1`
  (`LinearSVM.predict_eq_neg_one_of_neg`); the RBF kernel of a point
  with itself is `1` (`rbfKernel_self`); the RBF kernel is symmetric
  (`rbfKernel_comm`).
- SVM: **certified robustness**
  (`LinearSVM.predict_stable_of_dist_lt_margin`) — any point closer
  than the margin to `x` gets the same prediction as `x`, i.e. the
  margin is a robustness radius.
- Random forest: a one-tree forest predicts what that tree predicts
  (`RandomForest.predict_singleton`); a leaf always predicts its own
  label (`DecisionTree.predict_leaf`); a strict majority of trees
  decides the forest (`RandomForest.predict_eq_some_of_majority`); a
  unanimous nonempty forest predicts the shared label
  (`RandomForest.predict_eq_some_of_forall`); appending trees that
  agree with the current prediction cannot flip it
  (`RandomForest.predict_append_of_agree`); a tree whose leaves all
  carry one label always predicts it
  (`DecisionTree.predict_eq_of_allLeavesEq`), so a nonempty forest
  of such trees predicts that label
  (`RandomForest.predict_eq_some_of_allLeaves`). Worked example: a
  three-tree forest over two features predicts `"A"` at `(1, -1)`
  (`tinyForest_predict`).
- k-NN: a point is at distance `0` from itself (`euclidDist_self`);
  distance is symmetric (`euclidDist_comm`); with `k = 0` there is no
  prediction (`knnPredict_zero`); with `k = 1`, prediction is the
  label of the nearest point (`knnPredict_one_eq_nearestLabel`).
- Worked example: the toy SVM classifies all four training points
  correctly, its margin at `(2, 0)` is `1`, and every point within
  distance `1` of `(2, 0)` is predicted `+1`
  (`toySVM_stable_near_pos`).
- nanoGPT: softmax weights sum to 1 (`softmax_sum_eq_one`); the
  attention weights at any position sum to 1 (`attnWeights_sum`); a
  head's output at position `j` depends only on the prefix `0..j`
  (`headOut_take`); **full-model causality** — the logits at position
  `i` depend only on the tokens at positions `≤ i` (`logits_take`,
  standard axioms only); the forward pass preserves the sequence
  length and every logit vector has one entry per vocabulary item
  (`logits_length`, `logits_length_each`); the tiny model produces one
  8-entry logit vector per input token (`tiny_logits_length`,
  `tiny_logits_each_length`).

## Still open

Nothing — every stated theorem is proved, and there are no `sorry`s
in the codebase.

## Build

Requires Lean v4.27.0-rc1 and Mathlib (pinned in `lean-toolchain` and
`lakefile.lean` / `lake-manifest.json` — the same versions tiny-bayes
uses). With [elan](https://github.com/leanprover/elan) installed:

```sh
lake update           # fetch Mathlib at the pinned revision
lake exe cache get    # fetch prebuilt Mathlib oleans (or build from source)
lake build            # builds the VerifiedML library
lake build verified_ml  # also builds the executable (first run compiles
                        # Mathlib's C code and takes a while)
```

`lake build` must pass with no errors and no `sorry`s.

## Status

Early scaffold, November 2026. The classifiers are defined and their
main theorems proved, and nanoGPT is defined with its attention,
causality, and shape properties proved. No `sorry`s remain — every
stated theorem is proved.
