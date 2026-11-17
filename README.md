# verified-ml

Classical machine learning classifiers, defined in Lean 4 — towards
verified machine learning.

This is the "definitions first" scaffold for a proof-of-work project in
the spirit of [tiny-bayes](https://github.com/leochlon/tiny-bayes)
(Pythia-14M defined exactly in Lean 4). The idea is the same, applied
to the classics instead of a transformer: take a trained model, freeze
its weights, define its prediction function exactly, and then prove
things about that fixed function. No training procedure is formalised
here — only the trained classifiers.

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

Shared helpers (majority vote) live in `VerifiedML/Common.lean`.

## What is proved vs. what is still open

Proved (checked by the Lean kernel, no `sorry`):

- SVM: a point with a positive decision value is predicted `1`
  (`LinearSVM.predict_eq_one_of_pos`); negative gives `-1`
  (`LinearSVM.predict_eq_neg_one_of_neg`); the RBF kernel of a point
  with itself is `1` (`rbfKernel_self`); the RBF kernel is symmetric
  (`rbfKernel_comm`).
- Random forest: a one-tree forest predicts what that tree predicts
  (`RandomForest.predict_singleton`); a leaf always predicts its own
  label (`DecisionTree.predict_leaf`).
- k-NN: a point is at distance `0` from itself (`euclidDist_self`);
  distance is symmetric (`euclidDist_comm`); with `k = 0` there is no
  prediction (`knnPredict_zero`).

Stated but not yet proved (each is a `sorry` with a `TODO` comment
explaining the proof outline — nothing is hidden):

- SVM: **certified robustness** — `LinearSVM.predict_stable_of_dist_lt_margin`:
  any point closer than the margin to `x` gets the same prediction as
  `x`, i.e. the margin is a robustness radius. Proof outline: the
  decision values differ by `∑ wᵢ (yᵢ - xᵢ)`, Cauchy–Schwarz bounds
  that by `‖w‖ · dist`, which is `< |decision x|`, so the sign cannot
  change.
- k-NN: **1-NN is the nearest neighbour** —
  `knnPredict_one_eq_nearestLabel`: with `k = 1`, prediction equals
  the label of the nearest point. Needs the sorted-list fact that the
  first element of a distance-sorted list is a nearest point.

## Build

Requires Lean v4.27.0-rc1 and Mathlib (pinned in `lean-toolchain` and
`lakefile.lean` / `lake-manifest.json` — the same versions tiny-bayes
uses). With [elan](https://github.com/leanprover/elan) installed:

```sh
lake update           # fetch Mathlib at the pinned revision
lake exe cache get    # fetch prebuilt Mathlib oleans (or build from source)
lake build            # builds VerifiedML + the executable
```

`lake build` must pass with no errors. The two `sorry`s above are the
only ones in the codebase, both marked `TODO`.

## Status

Early scaffold, November 2026. Definitions are real Lean; the two
headline theorems are stated, not yet proved. Next steps: prove the
SVM robustness radius, prove the 1-NN statement, then add a small
trained example (e.g. a toy SVM on the Iris dataset) with its weights
as literal data.
