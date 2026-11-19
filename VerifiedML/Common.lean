import Mathlib.Data.List.MinMax

namespace VerifiedML

/-- Majority vote over a list of labels: the label occurring most often.
    Returns `none` for the empty list. Ties are broken by `List.argmax`,
    which returns the first maximal element. -/
noncomputable def majorityVote [DecidableEq α] (xs : List α) : Option α :=
  xs.eraseDups.argmax (fun a => xs.count a)

/-- The majority vote of an empty list is `none`. -/
theorem majorityVote_nil [DecidableEq α] : majorityVote ([] : List α) = none := by
  simp [majorityVote]

/-- The majority vote of a single label is that label. -/
theorem majorityVote_singleton [DecidableEq α] (a : α) :
    majorityVote [a] = some a := by
  have h1 : ([a].eraseDups) = [a] := rfl
  rw [majorityVote, h1]
  exact List.argmax_singleton

/-- Membership is unchanged by deduplication. -/
theorem mem_eraseDups {α : Type} [DecidableEq α] {a : α} {xs : List α} :
    a ∈ xs.eraseDups ↔ a ∈ xs := by
  have aux : ∀ n : ℕ, ∀ ys : List α, ys.length ≤ n → (a ∈ ys.eraseDups ↔ a ∈ ys) := by
    intro n
    induction n with
    | zero =>
      intro ys h
      have h0 : ys = [] := List.eq_nil_of_length_eq_zero (Nat.eq_zero_of_le_zero h)
      simp [h0]
    | succ n ih =>
      intro ys h
      cases ys with
      | nil => simp
      | cons x xs =>
        have hxs : xs.length ≤ n := by
          have hle : (x :: xs).length ≤ n + 1 := h
          simpa using hle
        have hflt : (xs.filter (fun b => !b == x)).length ≤ n :=
          le_trans (List.length_filter_le _ _) hxs
        rw [List.eraseDups_cons]
        have ihf := ih _ hflt
        simp only [List.mem_cons, List.mem_filter, ihf]
        by_cases hax : a = x
        · subst hax
          simp
        · constructor
          · rintro (rfl | ⟨h1, -⟩)
            · exact absurd rfl hax
            · exact Or.inr h1
          · rintro (rfl | h1)
            · exact absurd rfl hax
            · exact Or.inr ⟨h1, by simp [hax]⟩
  exact aux xs.length xs (le_refl _)

/-- The counts of two distinct labels together cannot exceed the length. -/
theorem count_add_count_le_of_ne {α : Type} [DecidableEq α] {a b : α} (h : a ≠ b)
    (xs : List α) : xs.count a + xs.count b ≤ xs.length := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    by_cases hx : x = a
    · subst hx
      rw [List.count_cons_self, List.count_cons_of_ne h]
      simp only [List.length_cons]
      omega
    · by_cases hx2 : x = b
      · subst hx2
        rw [List.count_cons_self, List.count_cons_of_ne (Ne.symm h)]
        simp only [List.length_cons]
        omega
      · rw [List.count_cons_of_ne hx, List.count_cons_of_ne hx2]
        simp only [List.length_cons]
        omega

/-- If `l` is the unique strict maximiser of the count, it wins the vote. -/
theorem majorityVote_eq_some_of_forall_count_lt {α : Type} [DecidableEq α] {xs : List α}
    {l : α} (hl : l ∈ xs) (hmax : ∀ a ∈ xs, a ≠ l → xs.count a < xs.count l) :
    majorityVote xs = some l := by
  have hl' : l ∈ xs.eraseDups := mem_eraseDups.mpr hl
  have hne : xs.eraseDups ≠ [] := by
    intro h
    rw [h] at hl'
    exact absurd hl' (by simp)
  obtain ⟨m, hm⟩ : ∃ m, xs.eraseDups.argmax (fun a => xs.count a) = some m := by
    rcases harg : xs.eraseDups.argmax (fun a => xs.count a) with _ | m
    · exact absurd (List.argmax_eq_none.mp harg) hne
    · exact ⟨m, rfl⟩
  have hm_mem : m ∈ xs.eraseDups := by
    apply List.argmax_mem (f := fun a => xs.count a)
    rw [hm]
    simp
  have hle : xs.count l ≤ xs.count m := by
    apply List.le_of_mem_argmax (f := fun a => xs.count a) hl'
    rw [hm]
    simp
  have hml : m = l := by
    by_contra hne'
    have hmem : m ∈ xs := mem_eraseDups.mp hm_mem
    have hlt : xs.count m < xs.count l := hmax m hmem hne'
    omega
  rw [hml] at hm
  exact hm

/-- A strict majority wins the vote. -/
theorem majorityVote_eq_some_of_count_gt_half {α : Type} [DecidableEq α] {xs : List α}
    {l : α} (h : xs.length < 2 * xs.count l) : majorityVote xs = some l := by
  apply majorityVote_eq_some_of_forall_count_lt
  · by_contra hmem
    have hz : xs.count l = 0 := List.count_eq_zero.mpr hmem
    omega
  · intro a ha hne
    have hle : xs.count a + xs.count l ≤ xs.length :=
      count_add_count_le_of_ne hne xs
    omega

/-- The winner of a vote has a count at least as large as any other label. -/
theorem count_le_of_majorityVote_eq_some {α : Type} [DecidableEq α] {xs : List α} {l : α}
    (h : majorityVote xs = some l) (a : α) (ha : a ∈ xs) : xs.count a ≤ xs.count l := by
  have ha' : a ∈ xs.eraseDups := mem_eraseDups.mpr ha
  have harg : xs.eraseDups.argmax (fun b => xs.count b) = some l := h
  apply List.le_of_mem_argmax ha'
  rw [harg]
  simp

/-- A unanimous nonempty list votes for its only label. -/
theorem majorityVote_eq_some_of_forall_eq {α : Type} [DecidableEq α] {xs : List α} {l : α}
    (hne : xs ≠ []) (hall : ∀ a ∈ xs, a = l) : majorityVote xs = some l := by
  apply majorityVote_eq_some_of_forall_count_lt
  · cases xs with
    | nil => exact absurd rfl hne
    | cons y ys =>
      have hy := hall y (by simp)
      rw [← hy]
      exact List.mem_cons_self
  · intro a ha hne'
    exact absurd (hall a ha) hne'

end VerifiedML
