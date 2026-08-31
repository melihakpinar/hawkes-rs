# Derivations

One file per formula, written **before** the code that implements it (CLAUDE.md §4).

A derivation must contain:

1. The mathematics, in full — not a sketch.
2. The index conventions, stated explicitly, including the exact range of every sum.
3. Source citations by equation number (`[Ozaki1979, eq. 7]`) or by `tick` path and
   line range.
4. The exact expression intended for the code, in a form a reader can compare
   line-by-line against the implementation.

Having written it, **stop and hand it to the repository owner for approval.** Do not
implement in the same turn. An index error is invisible in code and obvious in a
derivation; that separation is the entire point.

## Contents

| File | Status |
| --- | --- |
| `conventions.md` | Awaiting owner approval. Pins the CLAUDE.md §1.3 convention hazards; C1, C3 and C8 are grounded in experiments, not docstrings. |
| `univariate_loglikelihood.md` | Awaiting owner approval. Intensity, compensator, and the O(n) recursion with ties handled. |
| `univariate_gradient.md` | Awaiting owner approval. Partials w.r.t. `mu`, `alpha`, `beta`, including the recursive state derivative. |
| `check_univariate_derivation.py` | Throwaway consistency check for the two above. Not `hawkes` code; `python3` with no dependencies. |
| `multivariate_loglikelihood.md` | Awaiting owner approval. `d` dimensions: intensity, compensator, and the recursion grouped over distinct times pooled across all components. |
| `multivariate_gradient.md` | Awaiting owner approval. Partials w.r.t. `mu[i]`, `alpha[i][j]` and `beta`, with the per-component state-derivative recursion. |
| `check_multivariate_derivation.py` | Throwaway consistency check for the two above, including the `d = 1` bitwise reduction. |

Anything that a source does not settle does not belong here. It belongs in
`docs/open-questions.md`.
