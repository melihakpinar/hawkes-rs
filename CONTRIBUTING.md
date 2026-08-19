# Contributing

Read [`CLAUDE.md`](CLAUDE.md) first. It is the authority; this file is a summary of
the parts you will hit immediately.

## The one rule that matters

**Never implement a formula, convention, index range or numerical choice you cannot
trace to a named source.** A source is a paper cited by *equation number*, `tick`'s
source cited by *path and line range*, an approved derivation in
`docs/derivations/`, or a test that empirically pins the behaviour.

"Standard practice" is not a source. Neither is a strong prior.

If you catch yourself writing a comment like `// assuming the kernel is normalized
here`, stop. That comment is a bug report you are filing against yourself. Either
find the source and delete the hedge, or file the question in
`docs/open-questions.md` and work on something else.

## Derivation before code

Before implementing any formula, write the derivation to
`docs/derivations/<name>.md`: the mathematics, the index conventions, the citations,
and the exact expression you intend to code. **Then stop and get it approved.** Do
not implement in the same change. An index error is invisible in code and obvious in
a derivation.

## Verification

No estimator lands without an oracle that could have caught it being wrong. Adding a
feature means adding to the oracles in `hawk/tests/`.

**Sabotage rule.** When you add an oracle, prove it works: deliberately break the
code it guards, confirm the test goes red, revert, and record it in
`docs/verification-log.md`. An oracle that has never gone red is not known to be an
oracle.

**Do not write a test whose expected value you computed with the code under test.**
Expected values come from a paper, from `tick`, or from a hand calculation recorded
in `docs/derivations/`.

## Definition of done

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

All three clean, plus: every new formula traces to a citation, no new unresolved
entry in `docs/open-questions.md`, and a `CHANGELOG.md` entry for any public API
change.

Never commit an `#[ignore]`d or commented-out test to make the suite pass. If a test
fails and you cannot fix it, leave it failing and say so.

## Git

- Work on a branch: `feat/<issue>-<kebab-description>` or `fix/<issue>-<...>`. Never
  commit to the default branch. No issue? Create one first — work is never
  branchless and never issueless.
- [Conventional Commits](https://www.conventionalcommits.org/): `feat`, `fix`,
  `test`, `docs`, `refactor`, `perf`, `chore`, `ci`. Scopes are crate or module
  names: `hawk`, `hawk-python`, `bench`, `fixtures`.
- Summary under 72 characters, body wrapped at 72, explaining *why*. A commit
  encoding a convention decision must cite the source that settled it — the log is
  part of the audit trail.
- **No self-attribution.** No co-author trailers, no generated-by footers. Ever.

## Style

- `f64` everywhere. Never compare floats with `==`; every tolerance is a named
  constant with a comment saying where the number came from.
- No `unwrap`, `expect` or `panic!` in `src/` outside tests and documented invariant
  assertions. Invalid input is an error value.
- Long names are welcome: `branching_ratio_spectral_radius` beats `sr`. Single
  letters only where they mirror a cited equation, and say which equation.
- Duplication is not a defect. Extract a function when the copies are *coupled*, not
  because they look alike.
- No abstraction for a single caller: no trait with one implementor, no generic with
  one instantiation, no config struct with one field.
