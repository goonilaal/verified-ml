import Mathlib.Data.Real.Basic
import VerifiedML.Common

namespace VerifiedML

/-!
# Random forest

A decision tree over points in `Fin n → ℝ`, and a forest as a list of
trees whose predictions are combined by majority vote. Labels live in
an arbitrary type `Label` (with decidable equality, so votes can be
counted).
-/

/-- A decision tree: a leaf carrying a label, or an internal node that
    tests feature `i` against a threshold and branches left/right. -/
inductive DecisionTree (n : ℕ) (Label : Type) where
  | leaf : Label → DecisionTree n Label
  | node : Fin n → ℝ → DecisionTree n Label → DecisionTree n Label → DecisionTree n Label

/-- Predict the label of a point by walking down the tree. -/
noncomputable def DecisionTree.predict {n : ℕ} {Label : Type} :
    DecisionTree n Label → (Fin n → ℝ) → Label
  | .leaf a, _ => a
  | .node i t l r, x => if x i ≤ t then l.predict x else r.predict x

/-- A random forest is a list of decision trees. -/
abbrev RandomForest (n : ℕ) (Label : Type) := List (DecisionTree n Label)

/-- Forest prediction: majority vote over the trees' predictions.
    Returns `none` for the empty forest. -/
noncomputable def RandomForest.predict {n : ℕ} {Label : Type} [DecidableEq Label]
    (forest : RandomForest n Label) (x : Fin n → ℝ) : Option Label :=
  majorityVote (forest.map (fun t => t.predict x))

/-- A forest of one tree predicts exactly what that tree predicts. -/
theorem RandomForest.predict_singleton {n : ℕ} {Label : Type} [DecidableEq Label]
    (t : DecisionTree n Label) (x : Fin n → ℝ) :
    RandomForest.predict [t] x = some (t.predict x) := by
  simp [RandomForest.predict, majorityVote_singleton]

/-- A leaf tree always predicts its own label, whatever the point. -/
theorem DecisionTree.predict_leaf {n : ℕ} {Label : Type} (a : Label) (x : Fin n → ℝ) :
    (DecisionTree.leaf a : DecisionTree n Label).predict x = a :=
  rfl

end VerifiedML
