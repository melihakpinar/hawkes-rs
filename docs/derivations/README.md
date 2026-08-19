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
| `conventions.md` | Findings recorded, awaiting owner approval. Pins the CLAUDE.md §1.3 convention hazards to `tick`'s source. |

Anything that a source does not settle does not belong here. It belongs in
`docs/open-questions.md`.
