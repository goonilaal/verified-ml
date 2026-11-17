import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.List.GetD
import Mathlib.Data.Real.Sqrt
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

namespace VerifiedML

/-!
# nanoGPT, defined exactly in Lean

Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT, `model.py`),
defined exactly over ℝ, following the same approach as tiny-bayes
takes for Pythia-14M: the trained model is just data, so the
definitions below are the whole model — no training procedure is
formalised, only the fixed, trained network, in inference mode
(dropout is the identity).

Architecture, as in `model.py` (defaults: block_size 1024, vocab_size
50304, n_layer 12, n_head 12, n_embd 768, dropout 0.0, bias true):

* token embedding + learned position embedding, added at the input;
* each block is pre-LayerNorm with sequential residuals:
  `x = x + attn (ln_1 x)`, then `x = x + mlp (ln_2 x)`;
* attention is causal multi-head self-attention: one fused `c_attn`
  projection gives q, k, v; scores are `q ⬝ k / √head_dim`, masked to
  the left (the causal prefix), softmaxed; heads are concatenated and
  passed through `c_proj`;
* the MLP expands to `4 * n_embd`, applies exact GELU (`nn.GELU()`,
  the erf-based form: `x * Φ x`), and projects back;
* LayerNorm is PyTorch `F.layer_norm` with `eps = 1e-5` (biased
  variance); the final LayerNorm feeds `lm_head`, which nanoGPT ties
  to the token embedding (here it is a separate matrix, so the
  definition also covers the untied case — tying is the special case
  where its rows equal the token-embedding rows).

Two modelling notes: bias vectors are always present in the parameter
structures below; nanoGPT's `bias := false` is the special case where
they are all zero. And the forward pass folds over whatever blocks
`θ.blocks` holds; a faithful instantiation has
`θ.blocks.length = cfg.nLayer`.
-/

noncomputable section

/-- Dot product of two real vectors. -/
def dot (x y : List ℝ) : ℝ := (List.zipWith (· * ·) x y).sum

/-- Matrix (list of rows) times vector. -/
def mv (W : List (List ℝ)) (x : List ℝ) : List ℝ := W.map fun row => dot row x

/-- Componentwise sum. -/
def vadd (x y : List ℝ) : List ℝ := List.zipWith (· + ·) x y

/-- LayerNorm with gain `g`, bias `b`: `g * (x - mean) / sqrt (var + 1e-5) + b`,
    biased variance — exactly PyTorch `F.layer_norm` with `eps = 1e-5`. -/
def layerNorm (g b x : List ℝ) : List ℝ :=
  let d : ℝ := x.length
  let μ := x.sum / d
  let var := (x.map fun a => (a - μ) ^ 2).sum / d
  let s := Real.sqrt (var + 1 / 100000)
  vadd (List.zipWith (fun gi a => gi * ((a - μ) / s)) g x) b

/-- Standard normal CDF `Φ x = 1/2 + (1 / √(2π)) ∫₀ˣ exp(-t²/2) dt`. -/
def Phi (x : ℝ) : ℝ :=
  1 / 2 + (∫ t in (0 : ℝ)..x, Real.exp (-(t ^ 2) / 2)) / Real.sqrt (2 * Real.pi)

/-- Exact GELU, as in PyTorch `nn.GELU()` (the erf-based form): `x * Φ x`. -/
def gelu (x : ℝ) : ℝ := x * Phi x

/-- Softmax of a list of scores. -/
def softmax (s : List ℝ) : List ℝ := s.map fun x => Real.exp x / (s.map Real.exp).sum

/-- nanoGPT configuration, with Karpathy's defaults. -/
structure GPTConfig where
  blockSize : ℕ := 1024
  vocabSize : ℕ := 50304
  nLayer : ℕ := 12
  nHead : ℕ := 12
  nEmbd : ℕ := 768
  dropout : ℝ := 0
  bias : Bool := true

/-- Width of one attention head: `n_embd / n_head` (nanoGPT asserts
    `n_embd % n_head == 0`). -/
def GPTConfig.headDim (cfg : GPTConfig) : ℕ := cfg.nEmbd / cfg.nHead

/-- The parameters of one transformer block. Matrices are lists of rows. -/
structure BlockParams where
  ln1Weight : List ℝ
  ln1Bias : List ℝ
  /-- `attn.c_attn.weight`, `3 * n_embd` rows × `n_embd`. -/
  attnWeight : List (List ℝ)
  attnBias : List ℝ
  /-- `attn.c_proj.weight`, `n_embd` × `n_embd`. -/
  projWeight : List (List ℝ)
  projBias : List ℝ
  ln2Weight : List ℝ
  ln2Bias : List ℝ
  /-- `mlp.c_fc.weight`, `4 * n_embd` rows × `n_embd`. -/
  fcWeight : List (List ℝ)
  fcBias : List ℝ
  /-- `mlp.c_proj.weight`, `n_embd` × `4 * n_embd`. -/
  fcProjWeight : List (List ℝ)
  fcProjBias : List ℝ

/-- All parameters: embeddings, blocks, final LayerNorm, language-model head. -/
structure GPTParams where
  wte : ℕ → List ℝ
  wpe : ℕ → List ℝ
  blocks : List BlockParams
  lnfWeight : List ℝ
  lnfBias : List ℝ
  /-- `lm_head.weight`, `vocab_size` rows × `n_embd`; nanoGPT ties this
      to `wte`. -/
  lmHeadWeight : List (List ℝ)

/-- Slice of a vector belonging to head `h`: coordinates `[h * hs, (h+1) * hs)`. -/
def headSlice (x : List ℝ) (h hs : ℕ) : List ℝ := (x.drop (h * hs)).take hs

/-- `c_attn` applied to one (layer-normed) position: `3 * n_embd` coordinates. -/
def qkv (L : BlockParams) (u : List ℝ) : List ℝ := vadd (mv L.attnWeight u) L.attnBias

/-- Query part of a `qkv` output: the first `n_embd` coordinates. -/
def qOf (cfg : GPTConfig) (q : List ℝ) : List ℝ := q.take cfg.nEmbd

/-- Key part of a `qkv` output: the middle `n_embd` coordinates. -/
def kOf (cfg : GPTConfig) (q : List ℝ) : List ℝ := (q.drop cfg.nEmbd).take cfg.nEmbd

/-- Value part of a `qkv` output: the last `n_embd` coordinates. -/
def vOf (cfg : GPTConfig) (q : List ℝ) : List ℝ := q.drop (2 * cfg.nEmbd)

/-- The (unnormalised) attention scores of head `h` at position `j`:
    the scaled dot products of the query at `j` with the keys at
    positions `0..j` — the causal prefix. -/
def attnScores (cfg : GPTConfig) (L : BlockParams) (U : List (List ℝ)) (j h : ℕ) :
    List ℝ :=
  let hs := cfg.headDim
  let q := headSlice (qOf cfg (qkv L (U.getD j []))) h hs
  ((U.take (j + 1)).map fun u => headSlice (kOf cfg (qkv L u)) h hs).map fun k =>
    dot q k / Real.sqrt hs

/-- Attention weights of head `h` at position `j`: the softmax of the scores. -/
def attnWeights (cfg : GPTConfig) (L : BlockParams) (U : List (List ℝ)) (j h : ℕ) :
    List ℝ :=
  softmax (attnScores cfg L U j h)

/-- Attention-weighted sum of the value vectors of head `h`. -/
def mix (ws : List ℝ) (vs : List (List ℝ)) (hs : ℕ) : List ℝ :=
  (List.range hs).map fun d => (List.zipWith (fun w v => w * v.getD d 0) ws vs).sum

/-- Output of head `h` at position `j`. -/
def headOut (cfg : GPTConfig) (L : BlockParams) (U : List (List ℝ)) (j h : ℕ) : List ℝ :=
  let hs := cfg.headDim
  let vs := (U.take (j + 1)).map fun u => headSlice (vOf cfg (qkv L u)) h hs
  mix (attnWeights cfg L U j h) vs hs

/-- Attention output at position `j`: all heads concatenated, then `c_proj`. -/
def attnOut (cfg : GPTConfig) (L : BlockParams) (U : List (List ℝ)) (j : ℕ) : List ℝ :=
  vadd (mv L.projWeight ((List.range cfg.nHead).flatMap fun h => headOut cfg L U j h))
    L.projBias

/-- MLP output for one (layer-normed) position. -/
def mlpOut (L : BlockParams) (u : List ℝ) : List ℝ :=
  vadd (mv L.fcProjWeight ((vadd (mv L.fcWeight u) L.fcBias).map gelu)) L.fcProjBias

/-- One nanoGPT block, pre-LayerNorm with sequential residuals:
    `x = x + attn (ln_1 x)`, then `x = x + mlp (ln_2 x)`. -/
def block (cfg : GPTConfig) (L : BlockParams) (H : List (List ℝ)) : List (List ℝ) :=
  let U := H.map (layerNorm L.ln1Weight L.ln1Bias)
  let H1 := (List.range H.length).map fun j =>
    vadd (H.getD j []) (attnOut cfg L U j)
  let V := H1.map (layerNorm L.ln2Weight L.ln2Bias)
  (List.range H1.length).map fun j =>
    vadd (H1.getD j []) (mlpOut L (V.getD j []))

/-- Token + position embeddings: position `i` gets `wte[tokᵢ] + wpe[i]`. -/
def embed (θ : GPTParams) (toks : List ℕ) : List (List ℝ) :=
  List.zipWith vadd (toks.map θ.wte) ((List.range toks.length).map θ.wpe)

/-- Residual streams after all blocks. -/
def transformer (cfg : GPTConfig) (θ : GPTParams) (toks : List ℕ) : List (List ℝ) :=
  θ.blocks.foldl (fun H L => block cfg L H) (embed θ toks)

/-- Final hidden states, after the last LayerNorm. -/
def finalHidden (cfg : GPTConfig) (θ : GPTParams) (toks : List ℕ) : List (List ℝ) :=
  (transformer cfg θ toks).map (layerNorm θ.lnfWeight θ.lnfBias)

/-- Logits at every position. -/
def logits (cfg : GPTConfig) (θ : GPTParams) (toks : List ℕ) : List (List ℝ) :=
  (finalHidden cfg θ toks).map fun h => mv θ.lmHeadWeight h

/-- Next-token logits after the prompt: the logits at the last position. -/
def nextLogits (cfg : GPTConfig) (θ : GPTParams) (toks : List ℕ) : List ℝ :=
  (logits cfg θ toks).getD (toks.length - 1) []

/-- A sum of `exp` terms divided by a constant is the sum divided by it. -/
theorem sum_map_div_const (S : ℝ) (s : List ℝ) :
    (s.map (fun x => Real.exp x / S)).sum = (s.map Real.exp).sum / S := by
  induction s with
  | nil => simp
  | cons a s ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [ih, add_div]

/-- Softmax weights sum to 1 (for a nonempty score list). -/
theorem softmax_sum_eq_one {s : List ℝ} (hs : s ≠ []) :
    (softmax s).sum = 1 := by
  have hS : (s.map Real.exp).sum ≠ 0 := by
    have hpos : 0 < (s.map Real.exp).sum := by
      apply List.sum_pos
      · intro x hx
        obtain ⟨y, _, rfl⟩ := List.mem_map.mp hx
        exact Real.exp_pos y
      · intro h
        exact hs (List.map_eq_nil_iff.mp h)
    exact ne_of_gt hpos
  unfold softmax
  rw [sum_map_div_const, div_self hS]

/-- The attention weights at any position sum to 1. -/
theorem attnWeights_sum (cfg : GPTConfig) (L : BlockParams) {U : List (List ℝ)} {j h : ℕ}
    (hj : j < U.length) : (attnWeights cfg L U j h).sum = 1 := by
  unfold attnWeights
  apply softmax_sum_eq_one
  -- The score list has one entry per position `0..j`, so it is nonempty.
  have htake : (U.take (j + 1)) ≠ [] := by
    intro hcon
    have hlen : (U.take (j + 1)).length = j + 1 := by
      rw [List.length_take, Nat.min_eq_left (Nat.succ_le_of_lt hj)]
    rw [hcon] at hlen
    simp at hlen
  unfold attnScores
  intro hcon
  apply htake
  exact List.map_eq_nil_iff.mp (List.map_eq_nil_iff.mp hcon)

/-- Reading position `j` off the prefix `take (j + 1)` is the same as
    reading it off the whole list. -/
theorem getD_take_succ {α : Type} (U : List α) (j : ℕ) (d : α) :
    (U.take (j + 1)).getD j d = U.getD j d := by
  induction U generalizing j with
  | nil => simp
  | cons a U ih =>
    cases j with
    | zero => simp
    | succ j => simp

/-- **Causality at the attention level.** The output of head `h`
    at position `j` depends only on the prefix `U.take (j + 1)`: if two
    sequences agree on that prefix, the head output agrees. The query
    reads `U.getD j`, and `j` is the last index of that prefix, so the
    query is determined by the prefix too (`getD_take_succ`). -/
theorem headOut_take (cfg : GPTConfig) (L : BlockParams) {U V : List (List ℝ)} {j h : ℕ}
    (hUV : U.take (j + 1) = V.take (j + 1)) :
    headOut cfg L U j h = headOut cfg L V j h := by
  have hget : U.getD j [] = V.getD j [] := by
    rw [← getD_take_succ U j, ← getD_take_succ V j, hUV]
  simp only [headOut, attnWeights, attnScores, hUV, hget]


/-- Matrix-vector multiplication preserves the row count. -/
theorem mv_length (W : List (List ℝ)) (x : List ℝ) : (mv W x).length = W.length := by
  simp [mv]

/-- Embedding preserves the sequence length. -/
theorem embed_length (θ : GPTParams) (toks : List ℕ) :
    (embed θ toks).length = toks.length := by
  simp [embed, List.length_zipWith]

/-- One block preserves the sequence length. -/
theorem block_length (cfg : GPTConfig) (L : BlockParams) (H : List (List ℝ)) :
    (block cfg L H).length = H.length := by
  simp [block]

/-- Folding blocks preserves the sequence length. -/
theorem foldl_block_length (cfg : GPTConfig) (ls : List BlockParams) (H : List (List ℝ)) :
    (ls.foldl (fun H L => block cfg L H) H).length = H.length := by
  induction ls generalizing H with
  | nil => simp
  | cons L ls ih => simp [List.foldl_cons, block_length, ih]

/-- The transformer preserves the sequence length. -/
theorem transformer_length (cfg : GPTConfig) (θ : GPTParams) (toks : List ℕ) :
    (transformer cfg θ toks).length = toks.length := by
  simp [transformer, foldl_block_length, embed_length]

/-- The final hidden states have one entry per token. -/
theorem finalHidden_length (cfg : GPTConfig) (θ : GPTParams) (toks : List ℕ) :
    (finalHidden cfg θ toks).length = toks.length := by
  simp [finalHidden, transformer_length]

/-- The logits have one entry per token. -/
theorem logits_length (cfg : GPTConfig) (θ : GPTParams) (toks : List ℕ) :
    (logits cfg θ toks).length = toks.length := by
  simp [logits, finalHidden_length]

/-- Every logit vector has one entry per vocabulary item. -/
theorem logits_length_each (cfg : GPTConfig) (θ : GPTParams) (toks : List ℕ)
    (hW : θ.lmHeadWeight.length = cfg.vocabSize) :
    (logits cfg θ toks).map List.length = List.replicate toks.length cfg.vocabSize := by
  have key : ∀ l : List (List ℝ),
      (l.map (fun h => mv θ.lmHeadWeight h)).map List.length
        = l.map (fun _ => cfg.vocabSize) := by
    intro l
    induction l with
    | nil => simp
    | cons a l ih =>
      simp only [List.map_cons]
      rw [mv_length, hW, ih]
  have h1 : (logits cfg θ toks).map List.length =
      (finalHidden cfg θ toks).map (fun _ => cfg.vocabSize) := by
    unfold logits
    exact key _
  rw [h1, ← finalHidden_length cfg θ toks]
  exact List.map_const

/-- `getD` of a `map` at an in-range index. -/
theorem getD_map_of_lt {α β : Type} (f : α → β) (l : List α) {j : ℕ} (hj : j < l.length)
    (d : β) (dα : α) : (l.map f).getD j d = f (l.getD j dα) := by
  rw [List.getD_eq_getElem _ _ (by simpa using hj), List.getElem_map,
    List.getD_eq_getElem _ _ hj]

/-- Two lists agree on their first `n` entries when they agree pointwise there. -/
theorem take_eq_of_getElem_agree {α : Type} {l l' : List α} {n : ℕ}
    (hn : n ≤ l.length) (hn' : n ≤ l'.length)
    (h : ∀ j (hj : j < n), l[j]'(Nat.lt_of_lt_of_le hj hn) = l'[j]'(Nat.lt_of_lt_of_le hj hn')) :
    l.take n = l'.take n := by
  apply List.ext_getElem
  · simp [List.length_take, Nat.min_eq_left hn, Nat.min_eq_left hn']
  · intro j hj1 hj2
    have hjn : j < n := by
      have h1 : j < (l.take n).length := hj1
      rwa [List.length_take, Nat.min_eq_left hn] at h1
    simp only [List.getElem_take]
    exact h j hjn

/-- Attention output at position `j` depends only on the prefix `U.take (j + 1)`. -/
theorem attnOut_take (cfg : GPTConfig) (L : BlockParams) {U V : List (List ℝ)} {j : ℕ}
    (hUV : U.take (j + 1) = V.take (j + 1)) :
    attnOut cfg L U j = attnOut cfg L V j := by
  unfold attnOut
  congr 1
  congr 1
  apply List.flatMap_congr
  intro h _
  exact headOut_take cfg L hUV

/-- Reading position `j` off a prefix is the same as reading it off the whole list,
    whenever `j` lies inside the prefix. -/
theorem getD_take_of_lt {α : Type} (l : List α) {j n : ℕ} (hj : j < n) (d : α) :
    (l.take n).getD j d = l.getD j d := by
  by_cases hl : j < l.length
  · have hlen : j < (l.take n).length := by
      rw [List.length_take]
      exact Nat.lt_min.mpr ⟨hj, hl⟩
    rw [List.getD_eq_getElem _ _ hlen, List.getD_eq_getElem _ _ hl, List.getElem_take]
  · have hle : (l.take n).length ≤ j := by
      rw [List.length_take]
      exact Nat.le_trans (Nat.min_le_right _ _) (Nat.le_of_not_lt hl)
    rw [List.getD_eq_default _ _ hle, List.getD_eq_default _ _ (Nat.le_of_not_lt hl)]

/-- The embedding at position `j ≤ i` depends only on the tokens at positions `≤ i`. -/
theorem embed_getD (θ : GPTParams) {toks toks' : List ℕ} {i : ℕ}
    (h : toks.take (i + 1) = toks'.take (i + 1)) (hi : i < toks.length) :
    ∀ j, j ≤ i → (embed θ toks).getD j [] = (embed θ toks').getD j [] := by
  intro j hj
  have hjT : j < toks.length := lt_of_le_of_lt hj hi
  have hi' : i < toks'.length := by
    have hlen : (toks'.take (i + 1)).length = i + 1 := by
      rw [← h, List.length_take, Nat.min_eq_left (Nat.succ_le_of_lt hi)]
    have hle : i + 1 ≤ toks'.length := by
      rw [← hlen, List.length_take]
      exact Nat.min_le_right _ _
    exact Nat.lt_of_succ_le hle
  have hjT' : j < toks'.length := lt_of_le_of_lt hj hi'
  have htokD : toks.getD j 0 = toks'.getD j 0 := by
    have e : (toks.take (i + 1)).getD j 0 = (toks'.take (i + 1)).getD j 0 := by rw [h]
    rwa [getD_take_of_lt _ (Nat.lt_succ_of_le hj) 0,
      getD_take_of_lt _ (Nat.lt_succ_of_le hj) 0] at e
  have htok : toks[j]'(hjT) = toks'[j]'(hjT') := by
    rw [← List.getD_eq_getElem toks 0 hjT, ← List.getD_eq_getElem toks' 0 hjT']
    exact htokD
  simp only [embed, List.getD_eq_getElem, List.getElem_zipWith, List.getElem_map,
    List.getElem_range, List.length_zipWith, List.length_map, List.length_range,
    min_self, hjT, hjT']
  rw [htok]

/-- One block preserves agreement at every position `≤ i`. -/
theorem block_getD (cfg : GPTConfig) (L : BlockParams) {H H' : List (List ℝ)} {i : ℕ}
    (hH : ∀ j, j ≤ i → H.getD j [] = H'.getD j [])
    (hHi : i < H.length) (hHi' : i < H'.length) :
    ∀ j, j ≤ i → (block cfg L H).getD j [] = (block cfg L H').getD j [] := by
  intro j hj
  have hjH : j < H.length := lt_of_le_of_lt hj hHi
  have hjH' : j < H'.length := lt_of_le_of_lt hj hHi'
  -- The layer-normed streams agree on the prefix `0..j`.
  have hU : (H.map (layerNorm L.ln1Weight L.ln1Bias)).take (j + 1) =
      (H'.map (layerNorm L.ln1Weight L.ln1Bias)).take (j + 1) := by
    apply take_eq_of_getElem_agree (by simpa using hjH) (by simpa using hjH')
    intro k hk
    have e1 : (H.map (layerNorm L.ln1Weight L.ln1Bias))[k]'(by simpa using Nat.lt_of_lt_of_le hk hjH)
        = layerNorm L.ln1Weight L.ln1Bias (H.getD k []) := by
      rw [List.getElem_map, List.getD_eq_getElem _ _ (Nat.lt_of_lt_of_le hk hjH)]
    have e2 : (H'.map (layerNorm L.ln1Weight L.ln1Bias))[k]'(by simpa using Nat.lt_of_lt_of_le hk hjH')
        = layerNorm L.ln1Weight L.ln1Bias (H'.getD k []) := by
      rw [List.getElem_map, List.getD_eq_getElem _ _ (Nat.lt_of_lt_of_le hk hjH')]
    rw [e1, e2, hH k (le_trans (Nat.lt_succ_iff.mp hk) hj)]
  simp only [block, List.getD_eq_getElem, List.getElem_map, List.getElem_range,
    List.length_map, List.length_range, hjH, hjH']
  simp only [← List.getD_eq_getElem H [] hjH, ← List.getD_eq_getElem H' [] hjH']
  rw [hH j hj, attnOut_take cfg L hU]

/-- Folding blocks preserves agreement at every position `≤ i`. -/
theorem foldl_block_getD (cfg : GPTConfig) (ls : List BlockParams)
    {H H' : List (List ℝ)} {i : ℕ}
    (hH : ∀ j, j ≤ i → H.getD j [] = H'.getD j [])
    (hHi : i < H.length) (hHi' : i < H'.length) :
    ∀ j, j ≤ i → (ls.foldl (fun H L => block cfg L H) H).getD j [] =
      (ls.foldl (fun H L => block cfg L H) H').getD j [] := by
  induction ls generalizing H H' with
  | nil => exact hH
  | cons L ls ih =>
    simp only [List.foldl_cons]
    apply ih
    · exact block_getD cfg L hH hHi hHi'
    · rw [block_length]; exact hHi
    · rw [block_length]; exact hHi'

/-- **Causality of the full model.** The logits at position `i`
    depend only on the tokens at positions `≤ i`: if two prompts agree
    on their first `i + 1` tokens, their logits agree at position `i`.
    Proof: induction over the blocks — the embedding at position
    `i` depends only on token `i`; each block's output at position `i`
    depends only on its inputs at positions `≤ i`, since attention at
    `i` uses only the prefix `0..i` (`headOut_take`) and LayerNorm/MLP
    act per position; the final LayerNorm and `lm_head` act per
    position. -/
theorem logits_take (cfg : GPTConfig) (θ : GPTParams) {toks toks' : List ℕ} {i : ℕ}
    (h : toks.take (i + 1) = toks'.take (i + 1)) (hi : i < toks.length) :
    (logits cfg θ toks).getD i [] = (logits cfg θ toks').getD i [] := by
  have hi' : i < toks'.length := by
    have hlen : (toks'.take (i + 1)).length = i + 1 := by
      rw [← h, List.length_take, Nat.min_eq_left (Nat.succ_le_of_lt hi)]
    have hle : i + 1 ≤ toks'.length := by
      rw [← hlen, List.length_take]
      exact Nat.min_le_right _ _
    exact Nat.lt_of_succ_le hle
  have hT : ∀ j, j ≤ i → (transformer cfg θ toks).getD j [] =
      (transformer cfg θ toks').getD j [] := by
    show ∀ j, j ≤ i →
      (θ.blocks.foldl (fun H L => block cfg L H) (embed θ toks)).getD j [] =
        (θ.blocks.foldl (fun H L => block cfg L H) (embed θ toks')).getD j []
    apply foldl_block_getD cfg θ.blocks
    · exact embed_getD θ h hi
    · rw [embed_length]; exact hi
    · rw [embed_length]; exact hi'
  have hF : (finalHidden cfg θ toks).getD i [] = (finalHidden cfg θ toks').getD i [] := by
    have e1 : (finalHidden cfg θ toks).getD i [] =
        layerNorm θ.lnfWeight θ.lnfBias ((transformer cfg θ toks).getD i []) := by
      unfold finalHidden
      rw [getD_map_of_lt _ _ (by rw [transformer_length]; exact hi) _ []]
    have e2 : (finalHidden cfg θ toks').getD i [] =
        layerNorm θ.lnfWeight θ.lnfBias ((transformer cfg θ toks').getD i []) := by
      unfold finalHidden
      rw [getD_map_of_lt _ _ (by rw [transformer_length]; exact hi') _ []]
    rw [e1, e2, hT i le_rfl]
  have e1 : (logits cfg θ toks).getD i [] =
      mv θ.lmHeadWeight ((finalHidden cfg θ toks).getD i []) := by
    unfold logits
    rw [getD_map_of_lt _ _ (by rw [finalHidden_length]; exact hi) _ []]
  have e2 : (logits cfg θ toks').getD i [] =
      mv θ.lmHeadWeight ((finalHidden cfg θ toks').getD i []) := by
    unfold logits
    rw [getD_map_of_lt _ _ (by rw [finalHidden_length]; exact hi') _ []]
  rw [e1, e2, hF]

/-!
# A tiny nanoGPT

A miniature config — vocab 8, block size 4, 1 layer, 2 heads, width 8
— with hand-chosen weights, so the definitions above can be checked on
concrete shapes. The weights are not trained: the projections are
identity-like (query, key and value each copy the input; the MLP
expands by repeating it four times and averages back), LayerNorm gains
are 1 with zero bias, and the token/position embeddings are small
closed-form values. The `lm_head` is the identity, so the logits are
the final hidden states.
-/

/-- The `n × n` identity matrix, as a list of rows. -/
def idMat (n : ℕ) : List (List ℝ) :=
  (List.range n).map fun i => (List.range n).map fun j => if i = j then 1 else 0

/-- A matrix with `k * n` rows whose rows repeat the `n × n` identity
    `k` times. -/
def repIdMat (k n : ℕ) : List (List ℝ) :=
  (List.range (k * n)).map fun i => (List.range n).map fun j => if i % n = j then 1 else 0

/-- An `n × (k * n)` matrix averaging `k` copies: row `i` has `1 / k`
    at columns `j` with `j % n = i`, else `0`. -/
def avgMat (n k : ℕ) : List (List ℝ) :=
  (List.range n).map fun i => (List.range (k * n)).map fun j =>
    if j % n = i then 1 / k else 0

/-- Miniature nanoGPT config: vocab 8, block size 4, 1 layer, 2 heads, width 8. -/
def tinyConfig : GPTConfig :=
  { blockSize := 4, vocabSize := 8, nLayer := 1, nHead := 2, nEmbd := 8, dropout := 0,
    bias := true }

/-- The single block of the tiny model. -/
def tinyBlock : BlockParams where
  ln1Weight := List.replicate 8 1
  ln1Bias := List.replicate 8 0
  attnWeight := repIdMat 3 8
  attnBias := List.replicate 24 0
  projWeight := idMat 8
  projBias := List.replicate 8 0
  ln2Weight := List.replicate 8 1
  ln2Bias := List.replicate 8 0
  fcWeight := repIdMat 4 8
  fcBias := List.replicate 32 0
  fcProjWeight := avgMat 8 4
  fcProjBias := List.replicate 8 0

/-- Tiny nanoGPT parameters: closed-form embeddings, one block, identity head. -/
def tinyParams : GPTParams where
  wte := fun t => (List.range 8).map fun j => (t : ℝ) * 0.1 + j
  wpe := fun p => (List.range 8).map fun _ => (p : ℝ) * 0.01
  blocks := [tinyBlock]
  lnfWeight := List.replicate 8 1
  lnfBias := List.replicate 8 0
  lmHeadWeight := idMat 8

/-- The tiny model produces one logit vector per input token. -/
theorem tiny_logits_length :
    (logits tinyConfig tinyParams [1, 2, 0]).length = 3 :=
  logits_length _ _ _

/-- Every logit vector of the tiny model has one entry per vocabulary item. -/
theorem tiny_logits_each_length :
    (logits tinyConfig tinyParams [1, 2, 0]).map List.length = [8, 8, 8] := by
  have hW : tinyParams.lmHeadWeight.length = tinyConfig.vocabSize := by
    simp [tinyParams, idMat, tinyConfig]
  rw [logits_length_each tinyConfig tinyParams [1, 2, 0] hW]
  rfl

end

end VerifiedML

