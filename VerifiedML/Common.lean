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

end VerifiedML
