import Lake
open Lake DSL

package verified_ml where
  version := v!"0.1.0"

-- Pinned to the Mathlib revision used by tiny-bayes
-- (https://github.com/leochlon/tiny-bayes), known to work with
-- Lean v4.27.0-rc1.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "fdddb3ea"

@[default_target]
lean_lib VerifiedML

lean_exe verified_ml where
  root := `Main
