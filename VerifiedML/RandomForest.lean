import Mathlib.Data.Real.Basic
import Mathlib.Data.Fin.VecNotation
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

/-- If a strict majority of the trees predict `l`, so does the forest. -/
theorem RandomForest.predict_eq_some_of_majority {n : ℕ} {Label : Type} [DecidableEq Label]
    {forest : RandomForest n Label} {x : Fin n → ℝ} {l : Label}
    (h : (forest.map (fun t => t.predict x)).count l * 2 > forest.length) :
    RandomForest.predict forest x = some l := by
  apply majorityVote_eq_some_of_count_gt_half
  rw [List.length_map]
  omega

/-- If every tree predicts `l` (and the forest is nonempty), so does the forest. -/
theorem RandomForest.predict_eq_some_of_forall {n : ℕ} {Label : Type} [DecidableEq Label]
    {forest : RandomForest n Label} {x : Fin n → ℝ} {l : Label}
    (hne : forest ≠ []) (hall : ∀ t ∈ forest, t.predict x = l) :
    RandomForest.predict forest x = some l := by
  apply majorityVote_eq_some_of_forall_eq
  · cases forest with
    | nil => exact absurd rfl hne
    | cons t ts => simp
  · intro a ha
    obtain ⟨t, ht, htl⟩ := List.mem_map.mp ha
    have htl' : t.predict x = a := htl
    exact htl'.symm.trans (hall t ht)

/-- Adding trees that all agree with the forest's prediction cannot flip it. -/
theorem RandomForest.predict_append_of_agree {n : ℕ} {Label : Type} [DecidableEq Label]
    {forest forest' : RandomForest n Label} {x : Fin n → ℝ} {l : Label}
    (h : RandomForest.predict forest x = some l)
    (hne : forest' ≠ [])
    (hall : ∀ t ∈ forest', t.predict x = l) :
    RandomForest.predict (forest ++ forest') x = some l := by
  apply majorityVote_eq_some_of_forall_count_lt
  · rw [List.map_append]
    apply List.mem_append_right
    cases forest' with
    | nil => exact absurd rfl hne
    | cons t ts =>
      have ht : t.predict x = l := hall t (by simp)
      simp [ht]
  · intro a ha hne'
    rw [List.map_append, List.count_append, List.count_append]
    have ha_forest : a ∈ forest.map (fun t => t.predict x) := by
      rw [List.map_append, List.mem_append] at ha
      rcases ha with h1 | h1
      · exact h1
      · exfalso
        obtain ⟨t, ht, htl⟩ := List.mem_map.mp h1
        have htl' : t.predict x = a := htl
        exact hne' (htl'.symm.trans (hall t ht))
    have hle := count_le_of_majorityVote_eq_some h a ha_forest
    have hzero : (forest'.map (fun t => t.predict x)).count a = 0 := by
      rw [List.count_eq_zero]
      intro hmem
      obtain ⟨t, ht, htl⟩ := List.mem_map.mp hmem
      have htl' : t.predict x = a := htl
      exact hne' (htl'.symm.trans (hall t ht))
    have hcnt : (forest'.map (fun t => t.predict x)).count l = forest'.length := by
      rw [← List.length_map, List.count_eq_length]
      intro b hb
      obtain ⟨t, ht, htl⟩ := List.mem_map.mp hb
      have htl' : t.predict x = b := htl
      rw [← htl']
      exact (hall t ht).symm
    have hlen : 0 < forest'.length :=
      Nat.pos_of_ne_zero (fun hz => hne (List.eq_nil_of_length_eq_zero hz))
    omega

/-- A tree is uniformly labelled `l` when every leaf carries `l`. -/
def DecisionTree.allLeavesEq {n : ℕ} {Label : Type} (l : Label) :
    DecisionTree n Label → Prop
  | .leaf a => a = l
  | .node _ _ s r => s.allLeavesEq l ∧ r.allLeavesEq l

/-- A uniformly labelled tree always predicts its label. -/
theorem DecisionTree.predict_eq_of_allLeavesEq {n : ℕ} {Label : Type}
    {t : DecisionTree n Label} {x : Fin n → ℝ} {l : Label}
    (h : t.allLeavesEq l) : t.predict x = l := by
  induction t with
  | leaf a => exact h
  | node i th s r ihs ihr =>
    have h' : s.allLeavesEq l ∧ r.allLeavesEq l := h
    simp only [DecisionTree.predict]
    by_cases hc : x i ≤ th
    · rw [if_pos hc]; exact ihs h'.1
    · rw [if_neg hc]; exact ihr h'.2

/-- A nonempty forest of uniformly labelled trees predicts their label. -/
theorem RandomForest.predict_eq_some_of_allLeaves {n : ℕ} {Label : Type} [DecidableEq Label]
    {forest : RandomForest n Label} {x : Fin n → ℝ} {l : Label}
    (hne : forest ≠ []) (hall : ∀ t ∈ forest, t.allLeavesEq l) :
    RandomForest.predict forest x = some l := by
  apply RandomForest.predict_eq_some_of_forall hne
  intro t ht
  exact DecisionTree.predict_eq_of_allLeavesEq (hall t ht)

/-- A tiny three-tree forest over two features: two stumps split on one
    feature each, plus a constant tree. -/
def tinyForest : RandomForest 2 String :=
  [ DecisionTree.node 0 0 (DecisionTree.leaf "A") (DecisionTree.leaf "B"),
    DecisionTree.node 1 0 (DecisionTree.leaf "A") (DecisionTree.leaf "B"),
    DecisionTree.leaf "A" ]

/-- The tiny forest predicts `"A"` at `(1, -1)`: the first stump says
    `"B"` (feature 0 is above its threshold), the other two say `"A"`. -/
theorem tinyForest_predict :
    RandomForest.predict tinyForest ![1, -1] = some "A" := by
  have h1 : (DecisionTree.node (0 : Fin 2) (0 : ℝ) (DecisionTree.leaf "A")
      (DecisionTree.leaf "B") : DecisionTree 2 String).predict ![1, -1] = "B" := by
    show (if ![1, -1] (0 : Fin 2) ≤ (0 : ℝ) then "A" else "B") = "B"
    rw [if_neg (by norm_num [Matrix.cons_val_zero])]
  have h2 : (DecisionTree.node (1 : Fin 2) (0 : ℝ) (DecisionTree.leaf "A")
      (DecisionTree.leaf "B") : DecisionTree 2 String).predict ![1, -1] = "A" := by
    show (if ![1, -1] (1 : Fin 2) ≤ (0 : ℝ) then "A" else "B") = "A"
    rw [if_pos (by norm_num [Matrix.cons_val_one, Matrix.head_cons])]
  have h3 : (DecisionTree.leaf "A" : DecisionTree 2 String).predict ![1, -1] = "A" := rfl
  show majorityVote (tinyForest.map (fun t => t.predict ![1, -1])) = some "A"
  unfold tinyForest
  simp only [List.map_cons, List.map_nil, h1, h2, h3]
  decide

end VerifiedML
