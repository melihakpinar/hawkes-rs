# References

Papers cited by this project, as PDFs.

**Primary citable reference** (CLAUDE.md §2). Citations must resolve to an equation in
a freely accessible source:

| Key | Paper | Where to get it |
| --- | --- | --- |
| `Laub2015` | Laub, P. J., Taimre, T. & Pollett, P. K. (2015). *Hawkes Processes.* arXiv:1507.02822v1 | <https://arxiv.org/abs/1507.02822> — file `1507.02822v1.pdf` |
| `Bacry2015` | Bacry, E., Mastromatteo, I. & Muzy, J.-F. (2015). *Hawkes processes in finance.* arXiv:1502.04592 | <https://arxiv.org/abs/1502.04592> — PDF not committed |

From `Bacry2015`: Proposition 1 (stationarity, spectral radius of the kernel-norm
matrix below 1), Proposition 4 eq. 21 with eq. 17 (stationary mean intensity). Cited by
`multivariate_loglikelihood.md` §7. Freely accessible on arXiv, which is what CLAUDE.md
§2 requires; the PDF is not committed because `.gitignore` excludes PDFs here.

The equations relied on so far: eq. 4 (exponential-kernel intensity), eq. 5 (branching
ratio), eq. 6 (stationary mean intensity), Theorem 3 (likelihood on `[0, T]`), eq. 17
(log-likelihood), eq. 18 (compensator), eq. 19 (`O(n^2)` form), eq. 20 (Ozaki
recursion), eq. 21 (`O(n)` form), Theorem 4 (random time change), Algorithm 2 (Ogata
thinning).

**Note the parametrization.** [Laub2015, eq. 4] uses `alpha_L * exp(-beta t)` while
`hawkes` and `tick` use `alpha * beta * exp(-beta t)`. The map is
`alpha_L = alpha * beta`. See `docs/derivations/conventions.md` C1 and
`univariate_loglikelihood.md` §1.1 — getting it backwards makes every downstream
equation wrong by a factor of `beta`.

The remaining papers below are **not committed**: they are under publisher copyright,
and `.gitignore` excludes PDFs in this directory. They are cited for provenance only,
without equation numbers, which CLAUDE.md §2 permits when no PDF is present. Drop them
here locally if you have access.

Cite a paper by author, year and **equation number**, e.g. `[Ozaki1979, eq. 7]`.
A citation without an equation number does not satisfy CLAUDE.md §1: it does not let
a reader check the formula.

Expected contents (CLAUDE.md §2):

| Key | Paper |
| --- | --- |
| `Hawkes1971` | Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika* 58(1), 83-90. |
| `HawkesOakes1974` | Hawkes, A. G. & Oakes, D. (1974). A cluster process representation of a self-exciting process. *J. Appl. Probab.* 11(3), 493-503. |
| `Ozaki1979` | Ozaki, T. (1979). Maximum likelihood estimation of Hawkes' self-exciting point processes. *Ann. Inst. Statist. Math.* 31(1), 145-155. |
| `Ogata1981` | Ogata, Y. (1981). On Lewis' simulation method for point processes. *IEEE Trans. Inf. Theory* 27(1), 23-31. |
| `Ogata1988` | Ogata, Y. (1988). Statistical models for earthquake occurrences and residual analysis for point processes. *JASA* 83(401), 9-27. |
| `DassiosZhao2013` | Dassios, A. & Zhao, H. (2013). Exact simulation of Hawkes process with exponentially decaying intensity. *Electron. Commun. Probab.* 18(62), 1-13. |
| `Bacry2015` | Bacry, E., Mastromatteo, I. & Muzy, J.-F. (2015). Hawkes processes in finance. *Market Microstructure and Liquidity* 1(1), 1550005. |
| `Bacry2018` | Bacry, E. et al. (2018). tick: a Python library for statistical learning. *JMLR* 18(214), 1-5. |
| `Bacry2020` | Bacry, E., Bompaire, M., Gaiffas, S. & Muzy, J.-F. (2020). Sparse and low-rank multivariate Hawkes processes. *JMLR* 21(50), 1-32. |

Suggested filename: `<Key>.pdf`, e.g. `Ozaki1979.pdf`.
