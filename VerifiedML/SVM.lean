import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Real.Sign
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

open scoped BigOperators

namespace VerifiedML

/-!
# Support vector machine

A linear SVM over `ℝⁿ`, with points represented as `Fin n → ℝ`.
The trained model is just data — a weight vector and a bias — so the
definitions below are the whole model: no training procedure is
formalised here, only the fixed, trained classifier (the same approach
as tiny-bayes takes for a trained neural network).

The decision function is `w ⬝ x + b`; the prediction is its sign.
The geometric margin `|w ⬝ x + b| / ‖w‖` is the distance from `x` to the
decision boundary, and it is also the certified robustness radius: any
point closer than the margin to `x` gets the same prediction
(see `LinearSVM.predict_stable_of_dist_lt_margin`).
-/

/-- A linear SVM over `ℝⁿ`: a weight vector and a bias. -/
structure LinearSVM (n : ℕ) where
  weights : Fin n → ℝ
  bias : ℝ

/-- Raw decision value `w ⬝ x + b`. -/
noncomputable def LinearSVM.decision {n : ℕ} (svm : LinearSVM n) (x : Fin n → ℝ) : ℝ :=
  ∑ i, svm.weights i * x i + svm.bias

/-- Predicted label: `1`, `-1`, or `0` (exactly on the decision boundary). -/
noncomputable def LinearSVM.predict {n : ℕ} (svm : LinearSVM n) (x : Fin n → ℝ) : ℝ :=
  Real.sign (svm.decision x)

/-- Euclidean norm of the weight vector. -/
noncomputable def LinearSVM.weightNorm {n : ℕ} (svm : LinearSVM n) : ℝ :=
  Real.sqrt (∑ i, svm.weights i ^ 2)

/-- Geometric margin of a point: its distance to the decision boundary. -/
noncomputable def LinearSVM.margin {n : ℕ} (svm : LinearSVM n) (x : Fin n → ℝ) : ℝ :=
  |svm.decision x| / svm.weightNorm

/-- The RBF (Gaussian) kernel `K(x, y) = exp(-γ ‖x - y‖²)`. -/
noncomputable def rbfKernel {n : ℕ} (γ : ℝ) (x y : Fin n → ℝ) : ℝ :=
  Real.exp (-γ * ∑ i, (x i - y i) ^ 2)

/-- A point with a positive decision value is predicted `1`. -/
theorem LinearSVM.predict_eq_one_of_pos {n : ℕ} (svm : LinearSVM n) {x : Fin n → ℝ}
    (h : 0 < svm.decision x) : svm.predict x = 1 :=
  Real.sign_of_pos h

/-- A point with a negative decision value is predicted `-1`. -/
theorem LinearSVM.predict_eq_neg_one_of_neg {n : ℕ} (svm : LinearSVM n) {x : Fin n → ℝ}
    (h : svm.decision x < 0) : svm.predict x = -1 :=
  Real.sign_of_neg h

/-- The RBF kernel of a point with itself is `1`. -/
theorem rbfKernel_self {n : ℕ} (γ : ℝ) (x : Fin n → ℝ) : rbfKernel γ x x = 1 := by
  simp [rbfKernel]

/-- The RBF kernel is symmetric. -/
theorem rbfKernel_comm {n : ℕ} (γ : ℝ) (x y : Fin n → ℝ) :
    rbfKernel γ x y = rbfKernel γ y x := by
  have h : ∀ i : Fin n, (y i - x i) ^ 2 = (x i - y i) ^ 2 := by
    intro i
    rw [pow_two, pow_two, show y i - x i = -(x i - y i) from (neg_sub _ _).symm,
      neg_mul_neg]
  simp only [rbfKernel, Finset.sum_congr rfl (fun i _ => h i)]

/-- **Certified robustness.** If `y` is closer to `x` than `x`'s margin,
    then `y` gets the same prediction as `x`. In other words the margin
    is a robustness radius: no perturbation smaller than the margin can
    flip the classification. -/
theorem LinearSVM.predict_stable_of_dist_lt_margin {n : ℕ} (svm : LinearSVM n)
    (hw : 0 < svm.weightNorm) {x y : Fin n → ℝ}
    (h : Real.sqrt (∑ i, (x i - y i) ^ 2) < svm.margin x) :
    svm.predict y = svm.predict x := by
  -- TODO: prove this. Proof outline:
  -- `decision y - decision x = ∑ i, w i * (y i - x i)`, so by Cauchy–Schwarz
  -- (`Finset.sum_mul_sq_le_sq_mul_sq`) we get
  -- `|decision y - decision x| ≤ weightNorm * dist x y < |decision x|`,
  -- which forces `decision y` to have the same sign as `decision x`.
  sorry

end VerifiedML
