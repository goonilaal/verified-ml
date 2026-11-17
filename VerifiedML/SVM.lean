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
  -- The margin is `|decision x| / weightNorm`, so the hypothesis gives
  -- `dist * weightNorm < |decision x|`.
  have h2 : Real.sqrt (∑ i, (x i - y i) ^ 2) < |svm.decision x| / svm.weightNorm := h
  have hmargin : Real.sqrt (∑ i, (x i - y i) ^ 2) * svm.weightNorm < |svm.decision x| :=
    (lt_div_iff₀ hw).mp h2
  -- The decision values differ by the weighted sum of the displacement.
  have hdiff : svm.decision y - svm.decision x =
      ∑ i, svm.weights i * (y i - x i) := by
    simp only [LinearSVM.decision]
    rw [add_sub_add_right_eq_sub, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  -- Cauchy–Schwarz bounds that difference by `weightNorm * dist`.
  have hCS : (∑ i, svm.weights i * (y i - x i)) ^ 2 ≤
      (∑ i, svm.weights i ^ 2) * (∑ i, (y i - x i) ^ 2) :=
    Finset.sum_mul_sq_le_sq_mul_sq Finset.univ _ _
  have hw2nn : 0 ≤ ∑ i, svm.weights i ^ 2 :=
    Finset.sum_nonneg fun i _ => sq_nonneg _
  have habs : |svm.decision y - svm.decision x| ≤
      svm.weightNorm * Real.sqrt (∑ i, (x i - y i) ^ 2) := by
    have e1 : |svm.decision y - svm.decision x| =
        Real.sqrt ((∑ i, svm.weights i * (y i - x i)) ^ 2) := by
      rw [hdiff, Real.sqrt_sq_eq_abs]
    have e2 : svm.weightNorm * Real.sqrt (∑ i, (x i - y i) ^ 2) =
        Real.sqrt ((∑ i, svm.weights i ^ 2) * (∑ i, (x i - y i) ^ 2)) := by
      show Real.sqrt (∑ i, svm.weights i ^ 2) * Real.sqrt (∑ i, (x i - y i) ^ 2) = _
      rw [← Real.sqrt_mul hw2nn]
    rw [e1, e2]
    apply Real.sqrt_le_sqrt
    calc (∑ i, svm.weights i * (y i - x i)) ^ 2
        ≤ (∑ i, svm.weights i ^ 2) * (∑ i, (y i - x i) ^ 2) := hCS
      _ = (∑ i, svm.weights i ^ 2) * (∑ i, (x i - y i) ^ 2) := by
        congr 1
        exact Finset.sum_congr rfl fun i _ => by
          rw [show x i - y i = -(y i - x i) from (neg_sub _ _).symm, neg_sq]
  -- So the decision value cannot change sign.
  have hlt : |svm.decision y - svm.decision x| < |svm.decision x| :=
    lt_of_le_of_lt habs (by rw [mul_comm]; exact hmargin)
  have hd0 : svm.decision x ≠ 0 := abs_pos.mp (lt_of_le_of_lt (abs_nonneg _) hlt)
  rcases lt_or_gt_of_ne hd0 with hneg | hpos
  · have hy : svm.decision y < 0 := by
      have h1 : svm.decision y - svm.decision x ≤ |svm.decision y - svm.decision x| :=
        le_abs_self _
      have h2 : |svm.decision x| = -svm.decision x := abs_of_neg hneg
      linarith
    rw [LinearSVM.predict_eq_neg_one_of_neg svm hy,
      LinearSVM.predict_eq_neg_one_of_neg svm hneg]
  · have hy : 0 < svm.decision y := by
      have h1 : -(svm.decision y - svm.decision x) ≤ |svm.decision y - svm.decision x| :=
        neg_le_abs _
      have h2 : |svm.decision x| = svm.decision x := abs_of_pos hpos
      linarith
    rw [LinearSVM.predict_eq_one_of_pos svm hy, LinearSVM.predict_eq_one_of_pos svm hpos]

end VerifiedML
