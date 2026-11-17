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

/-- Stability of `List.mergeSort`, in the form needed for 1-NN: the head
    of the distance-sorted list is the *first* minimal element, i.e. its
    first occurrence in the original list is no later than that of any
    other minimal element. Proved by induction on the list, using
    `List.mergeSort_cons` to see where the head element is inserted. -/
private theorem mergeSort_head_idxOf_le {α : Type} [DecidableEq α] (dist : α → ℝ) :
    ∀ (l : List α) {a : α} {rest : List α},
      l.mergeSort (fun p q => decide (dist p ≤ dist q)) = a :: rest →
      ∀ b ∈ l, dist b ≤ dist a → l.idxOf a ≤ l.idxOf b := by
  intro l
  induction l with
  | nil =>
    intro a rest hsort
    simp at hsort
  | cons c t ih =>
    intro a rest hsort b hb hdist
    obtain ⟨l₁, l₂, h1, h2, h3⟩ := List.mergeSort_cons
      (le := fun p q => decide (dist p ≤ dist q))
      (fun p q r hp hq => by
        simp only [decide_eq_true_eq] at hp hq ⊢
        exact le_trans hp hq)
      (fun p q => by
        rcases le_total (dist p) (dist q) with h | h
        · simp [h]
        · simp [h])
      c t
    rw [h1] at hsort
    cases l₁ with
    | nil =>
      simp only [List.nil_append] at hsort
      obtain ⟨rfl, -⟩ := List.cons.inj hsort
      simp [List.idxOf_cons]
    | cons d l₁' =>
      simp only [List.cons_append] at hsort
      obtain ⟨rfl, -⟩ := List.cons.inj hsort
      have hsort_t : t.mergeSort (fun p q => decide (dist p ≤ dist q)) =
          d :: (l₁' ++ l₂) := by
        rw [h2, List.cons_append]
      have hnd : ¬ dist c ≤ dist d := by
        have h4 := h3 d List.mem_cons_self
        simpa using h4
      have hdc : d ≠ c := by
        rintro rfl
        exact hnd (le_refl _)
      have hbc : b ≠ c := by
        rintro rfl
        exact hnd hdist
      have hbt : b ∈ t := by
        rcases List.mem_cons.mp hb with h | h
        · exact absurd h hbc
        · exact h
      have hle := ih (a := d) (rest := l₁' ++ l₂) hsort_t b hbt hdist
      have hcd : (c == d) = false := decide_eq_false (fun h => hdc h.symm)
      have hcb : (c == b) = false := decide_eq_false (fun h => hbc h.symm)
      have e1 : (c :: t).idxOf d = t.idxOf d + 1 := by
        simp [List.idxOf_cons, hcd]
      have e2 : (c :: t).idxOf b = t.idxOf b + 1 := by
        simp [List.idxOf_cons, hcb]
      rw [e1, e2]
      exact Nat.add_le_add_right hle 1

/-- With `k = 1`, k-NN predicts the label of the nearest neighbour. -/
theorem knnPredict_one_eq_nearestLabel {n : ℕ} {Label : Type} [DecidableEq Label]
    (x : Fin n → ℝ) (data : Dataset n Label) :
    knnPredict 1 x data = nearestLabel x data := by
  classical
  have htrans : ∀ a b c : (Fin n → ℝ) × Label,
      (fun p q => decide (euclidDist x p.1 ≤ euclidDist x q.1)) a b = true →
      (fun p q => decide (euclidDist x p.1 ≤ euclidDist x q.1)) b c = true →
      (fun p q => decide (euclidDist x p.1 ≤ euclidDist x q.1)) a c = true := by
    intro a b c hab hbc
    simp only [decide_eq_true_eq] at hab hbc ⊢
    exact le_trans hab hbc
  have htotal : ∀ a b : (Fin n → ℝ) × Label,
      ((fun p q => decide (euclidDist x p.1 ≤ euclidDist x q.1)) a b ||
        (fun p q => decide (euclidDist x p.1 ≤ euclidDist x q.1)) b a) = true := by
    intro a b
    rcases le_total (euclidDist x a.1) (euclidDist x b.1) with h | h
    · simp [h]
    · simp [h]
  cases data with
  | nil =>
    simp [knnPredict, kNearestLabels, nearestLabel, majorityVote_nil]
  | cons p data' =>
    have hperm := List.mergeSort_perm (p :: data')
      (fun a b => decide (euclidDist x a.1 ≤ euclidDist x b.1))
    cases hs : (p :: data').mergeSort
        (fun a b => decide (euclidDist x a.1 ≤ euclidDist x b.1)) with
    | nil =>
      exfalso
      rw [hs] at hperm
      exact absurd (List.Perm.nil_eq hperm) (by simp)
    | cons a s' =>
      have hLHS : knnPredict 1 x (p :: data') = some a.2 := by
        simp [knnPredict, kNearestLabels, hs, majorityVote_singleton]
      have hpair : ((p :: data').mergeSort
          (fun a b => decide (euclidDist x a.1 ≤ euclidDist x b.1))).Pairwise
          (fun u v => (fun a b => decide (euclidDist x a.1 ≤ euclidDist x b.1)) u v
            = true) :=
        List.pairwise_mergeSort htrans htotal _
      rw [hs] at hpair
      have hpair' := (List.pairwise_cons.mp hpair).1
      have ha_mem : a ∈ p :: data' := by
        have h : a ∈ (p :: data').mergeSort
            (fun a b => decide (euclidDist x a.1 ≤ euclidDist x b.1)) := by
          rw [hs]
          exact List.mem_cons_self
        exact (List.Perm.mem_iff hperm).mp h
      have hmin : ∀ b ∈ p :: data', euclidDist x a.1 ≤ euclidDist x b.1 := by
        intro b hb
        have hb_s : b ∈ (p :: data').mergeSort
            (fun a b => decide (euclidDist x a.1 ≤ euclidDist x b.1)) :=
          (List.Perm.mem_iff hperm).mpr hb
        rw [hs] at hb_s
        rcases List.mem_cons.mp hb_s with rfl | hb'
        · exact le_refl _
        · have h := hpair' b hb'
          simpa only [decide_eq_true_eq] using h
      have harg : (p :: data').argmax (fun q => -euclidDist x q.1) = some a := by
        rw [List.argmax_eq_some_iff]
        refine ⟨ha_mem, ?_, ?_⟩
        · intro b hb
          simp only [neg_le_neg_iff]
          exact hmin b hb
        · intro b hb hle
          simp only [neg_le_neg_iff] at hle
          exact mergeSort_head_idxOf_le (fun q => euclidDist x q.1) _ hs b hb hle
      rw [hLHS]
      simp [nearestLabel, harg]

end VerifiedML
