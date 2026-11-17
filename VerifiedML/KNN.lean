import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import VerifiedML.Common

open scoped BigOperators

namespace VerifiedML

/-!
# k-nearest neighbours

Points in `Fin n → ℝ` with Euclidean distance. The classifier takes the
`k` nearest labelled points to a query and predicts by majority vote.
-/

/-- Euclidean distance between two points. -/
noncomputable def euclidDist {n : ℕ} (x y : Fin n → ℝ) : ℝ :=
  Real.sqrt (∑ i, (x i - y i) ^ 2)

/-- A point is at distance `0` from itself. -/
theorem euclidDist_self {n : ℕ} (x : Fin n → ℝ) : euclidDist x x = 0 := by
  simp [euclidDist]

/-- Euclidean distance is symmetric. -/
theorem euclidDist_comm {n : ℕ} (x y : Fin n → ℝ) : euclidDist x y = euclidDist y x := by
  have h : ∀ i : Fin n, (y i - x i) ^ 2 = (x i - y i) ^ 2 := by
    intro i
    rw [pow_two, pow_two, show y i - x i = -(x i - y i) from (neg_sub _ _).symm,
      neg_mul_neg]
  simp only [euclidDist, Finset.sum_congr rfl (fun i _ => h i)]

/-- A labelled dataset: points paired with their labels. -/
abbrev Dataset (n : ℕ) (Label : Type) := List ((Fin n → ℝ) × Label)

/-- The label of the single nearest neighbour, if the dataset is nonempty. -/
noncomputable def nearestLabel {n : ℕ} {Label : Type} (x : Fin n → ℝ)
    (data : Dataset n Label) : Option Label :=
  (data.argmax (fun p => -euclidDist x p.1)).map Prod.snd

/-- The labels of the `k` nearest neighbours, nearest first. -/
noncomputable def kNearestLabels {n : ℕ} {Label : Type} (k : ℕ) (x : Fin n → ℝ)
    (data : Dataset n Label) : List Label :=
  ((data.mergeSort (fun a b => decide (euclidDist x a.1 ≤ euclidDist x b.1))).take k).map Prod.snd

/-- k-NN prediction: majority vote over the `k` nearest neighbours' labels.
    Returns `none` when `k = 0` or the dataset is empty. -/
noncomputable def knnPredict {n : ℕ} {Label : Type} [DecidableEq Label]
    (k : ℕ) (x : Fin n → ℝ) (data : Dataset n Label) : Option Label :=
  majorityVote (kNearestLabels k x data)

/-- With `k = 0` there are no neighbours, so there is no prediction. -/
theorem knnPredict_zero {n : ℕ} {Label : Type} [DecidableEq Label]
    (x : Fin n → ℝ) (data : Dataset n Label) : knnPredict 0 x data = none := by
  simp [knnPredict, kNearestLabels, majorityVote_nil]

/-- With `k = 1`, k-NN predicts the label of the nearest neighbour. -/
theorem knnPredict_one_eq_nearestLabel {n : ℕ} {Label : Type} [DecidableEq Label]
    (x : Fin n → ℝ) (data : Dataset n Label) :
    knnPredict 1 x data = nearestLabel x data := by
  -- TODO: prove this. It needs that the first element of `mergeSort` by
  -- distance is an `argmax` of the negated distance, i.e. the nearest
  -- point — via `List.Sorted` / `List.Perm` of `mergeSort` plus the fact
  -- that sorting by `≤` puts a minimal element first.
  sorry

end VerifiedML
