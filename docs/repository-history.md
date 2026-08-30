# Why the history is not rewritten

Short version: **do not run `git filter-repo`, BFG, or `filter-branch` on this
repository.** There are three stale binary blobs in the history. They are staying. This
file exists so that whoever finds them next does not helpfully remove them.

## What is in there

`hawk-python/python/hawk/_hawk.abi3.so` — the compiled extension module — was tracked
for part of M3. `maturin develop` writes it into the package source tree, so every local
build dirtied the working tree and three commits picked up a copy:

| Commit | |
| --- | --- |
| `84b7cb4` | `feat(hawk-python): PyO3 bindings, and fix a fixture-parsing defect they exposed` |
| `18b2819` | `feat(hawk-python): input contract, array policy, and two Rust bugs they exposed` |
| `7a7e3f7` | `refactor(hawk): return Result from the multivariate dimension guard` |

Three distinct blobs, 8.7 MB uncompressed between them. It is now in `.gitignore` and
untracked, so no further copies will accumulate.

## Why they are not being removed

Because the cost of removing them is much higher than the cost of keeping them, and the
cost is of a kind this repository cannot afford.

**This project's audit trail is the commit history.** That is not incidental to how it
works — it is load-bearing:

- `docs/verification-log.md` records, for every oracle, that it was deliberately broken
  and observed going red. Those entries refer to specific code at specific commits.
- Derivations are approved before implementation (CLAUDE.md §4), and which commit
  implemented which approved derivation is part of the argument that the implementation
  is a transcription rather than an invention.
- `CLAUDE.md` §9 requires a commit encoding a convention decision to cite the source
  that settled it, so the log is where those decisions are recorded.
- The fixtures are byte-identical reference data from a pinned `tick` image, and the
  proof that they have not drifted is that the history shows them unchanged.
- Every merged PR body cites commits, and CI runs are pinned to SHAs.

`git filter-repo` rewrites **every SHA in the repository**. Every commit reference in
`verification-log.md`, in `open-questions.md`, in the derivations, in PR descriptions and
in issue comments becomes a dangling pointer to an object that no longer exists. GitHub's
PR and review history cannot be rewritten to match, so the two would disagree
permanently. The audit trail would still exist, but it would no longer resolve, and an
audit trail you cannot follow is a filing cabinet.

Three binary blobs are cheap by comparison. The measured cost is:

```text
$ du -sh .git
7.4M
```

That is the **entire repository** — history, fixtures, every branch. The blobs delta and
compress well, and 7.4 MB is not a number anyone needs to act on.

## If the repository does get large one day

Look at what is actually large before reaching for history rewriting. In this repository
the plausible candidates are the fixture corpus and any future reference data, not these
three blobs. And if a rewrite ever does become genuinely necessary, it is a decision for
the repository owner that has to be taken together with a plan for the references it
breaks — not a cleanup someone performs in passing because `git count-objects` looked
untidy.

## Decision

Taken by the repository owner, 2026-08-30, on issue #31. The blobs stay in history; the
file is untracked and ignored going forward.
