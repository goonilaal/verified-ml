import Mathlib.Data.Fin.VecNotation
import Mathlib.Algebra.BigOperators.Fin
import VerifiedML.SVM

open scoped BigOperators

namespace VerifiedML

/-!
# A toy trained SVM

A tiny 2-D example with literal weights, so the definitions in
`VerifiedML/SVM.lean` can be seen working on concrete numbers.

The dataset is four points, two per class:

* label `-1`: `(-1, 0)`, `(0, 1)`
* label `+1`: `(2, 0)`, `(2, 1)`

The weights were chosen by hand, not fitted by an algorithm: the line
`x₁ = 1` separates the two classes, so weights `(1, 0)` and bias `-1`
give decision value `x₁ - 1`. The closest points sit at distance `1`
from the boundary, so the margin is `1` there. The file proves the
model classifies all four training points correctly, and that the
certified-robustness theorem applies around `(2, 0)`: every point
within distance `1` of it is predicted `+1`.
-/

/-- Toy 2-D SVM: weights `(1, 0)`, bias `-1`. Decision boundary `x₁ = 1`. -/
noncomputable def toySVM : LinearSVM 2 :=
  ⟨![1, 0], -1⟩

/-- Decision value at `(2, 0)`. -/
theorem toySVM_decision_pos : toySVM.decision ![2, 0] = 1 := by
  simp [LinearSVM.decision, Fin.sum_univ_two, toySVM]; norm_num

/-- Decision value at `(2, 1)`. -/
theorem toySVM_decision_pos' : toySVM.decision ![2, 1] = 1 := by
  simp [LinearSVM.decision, Fin.sum_univ_two, toySVM]; norm_num

/-- Decision value at `(-1, 0)`. -/
theorem toySVM_decision_neg : toySVM.decision ![-1, 0] = -2 := by
  simp [LinearSVM.decision, Fin.sum_univ_two, toySVM]; norm_num

/-- Decision value at `(0, 1)`. -/
theorem toySVM_decision_neg' : toySVM.decision ![0, 1] = -1 := by
  simp [LinearSVM.decision, Fin.sum_univ_two, toySVM]

/-- The weight norm is `1`. -/
theorem toySVM_weightNorm : toySVM.weightNorm = 1 := by
  simp [LinearSVM.weightNorm, Fin.sum_univ_two, toySVM]

/-- The margin at `(2, 0)` is `1`. -/
theorem toySVM_margin_pos : toySVM.margin ![2, 0] = 1 := by
  unfold LinearSVM.margin
  rw [toySVM_decision_pos, toySVM_weightNorm]
  norm_num

/-- `(2, 0)` is predicted `+1`. -/
theorem toySVM_predict_pos : toySVM.predict ![2, 0] = 1 := by
  apply LinearSVM.predict_eq_one_of_pos
  rw [toySVM_decision_pos]
  norm_num

/-- `(2, 1)` is predicted `+1`. -/
theorem toySVM_predict_pos' : toySVM.predict ![2, 1] = 1 := by
  apply LinearSVM.predict_eq_one_of_pos
  rw [toySVM_decision_pos']
  norm_num

/-- `(-1, 0)` is predicted `-1`. -/
theorem toySVM_predict_neg : toySVM.predict ![-1, 0] = -1 := by
  apply LinearSVM.predict_eq_neg_one_of_neg
  rw [toySVM_decision_neg]
  norm_num

/-- `(0, 1)` is predicted `-1`. -/
theorem toySVM_predict_neg' : toySVM.predict ![0, 1] = -1 := by
  apply LinearSVM.predict_eq_neg_one_of_neg
  rw [toySVM_decision_neg']
  norm_num

/-- Certified robustness in action: every point within distance `1` of
    `(2, 0)` — the margin there — is predicted `+1`. -/
theorem toySVM_stable_near_pos (y : Fin 2 → ℝ)
    (h : Real.sqrt (∑ i, (![2, 0] i - y i) ^ 2) < 1) :
    toySVM.predict y = 1 := by
  have hw : 0 < toySVM.weightNorm := by
    rw [toySVM_weightNorm]
    norm_num
  have hstab := LinearSVM.predict_stable_of_dist_lt_margin toySVM hw
    (x := ![2, 0]) (y := y) (by rwa [toySVM_margin_pos])
  rw [hstab, toySVM_predict_pos]

end VerifiedML
