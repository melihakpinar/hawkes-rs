# What the Python bindings do with numpy arrays

Status: **decisions recorded before the tests exist**, per M3 step 5.

numpy has a default for every case below — a silent cast, a silent copy, a silent
reinterpretation. Per CLAUDE.md §1 a default is not a decision. Each case here is
chosen, with the reason, and each has a test that names the rule it enforces.

The governing principle is the repository's: the product is correct numbers, and a
boundary that quietly changes them is worse than one that refuses, because the caller
cannot tell.

---

## 1. dtype — only `float64` is accepted

Anything else raises `TypeError`. No cast is performed, not even a widening one.

### Why not accept `float32` and widen it

A `float32` → `float64` cast is exact, so this looks safe. It is not, because the
question is not whether the cast loses information but whether the information was
already lost.

Near a realistic horizon `float32` cannot separate events:

| `t` | `ulp(float32)` | `ulp(float64)` |
| --- | --- | --- |
| 1e3 | 6.10e-05 | 1.14e-13 |
| 1e5 | 7.81e-03 | 1.46e-11 |
| 1e6 | 6.25e-02 | 1.16e-10 |
| 1e7 | **1.0** | 1.86e-09 |

At `t = 1e7` the spacing is a whole time unit. Three events `0.03` apart at `t = 1e6`
become

```
[1000000.0, 1000000.0, 1000000.0625]
```

— a **tie that was not in the data**, and the sequence is no longer strictly
increasing.

That is not a rounding nuisance. `conventions.md` C8 admits ties, and
`univariate_loglikelihood.md` §3.1 records that on tied data the objective is not a
likelihood at all: the maximum-likelihood asymptotics do not apply, and `fit`'s
standard errors mean nothing. A silent widening cast would hand a caller a fitted
model with invalid guarantees and no indication.

So the caller converts explicitly. `np.asarray(t, dtype=np.float64)` is one line, and
writing it is the moment they decide their data really is `float64`.

### Why not accept integer arrays

Same rule, same reason in reverse: `int64` timestamps convert exactly, but accepting
them means the binding has an implicit conversion policy, and the next question is
`int32`, then `float16`. One rule — `float64` only — has no edge.

### Why not accept lists

`hawk.univariate.negative_log_likelihood(params, [1.0, 2.0], 3.0)` would require
inferring a dtype, and `[1, 2]` infers `int64`, which rule 1 then rejects. A caller
would see a list of numbers refused for a reason invisible in the source. Array
arguments take numpy arrays; the error message says so.

---

## 2. Contiguity — non-contiguous input is accepted, and copied

`t[::2]` and views into larger buffers are ordinary numpy. Rust needs a contiguous
`&[f64]`, so the binding copies.

Accepted rather than rejected because a copy changes no value: it costs `O(n)` time
and memory and carries no numerical risk. Refusing would be hostile for no gain.

The copy is documented in the docstrings so a caller sizing a large fit knows it
happens.

---

## 3. Memory order of the excitation matrix — logical indexing always wins

A 2-D `excitation` array is accepted in C order or Fortran order. **The value at
logical position `[i][j]` is always the value `hawk` uses for `alpha[i][j]`.** An
F-ordered array is copied into C order; its buffer is never reinterpreted.

This is the most dangerous case in this document, and the only one where the wrong
choice is silent and wrong rather than merely surprising.

`hawk` stores `excitation` row-major and reads `excitation[i * d + j]` as "j excites
i" (`conventions.md` C6). Handing Rust the raw buffer of an F-ordered array without
copying would deliver **the transpose** — a plausible-looking matrix that is wrong,
and undetectable on symmetric input. It is exactly the failure C6 exists to prevent
and that `S33`, `S37` and the elementwise round-trip test were built to catch on the
Rust side; the boundary must not reintroduce it.

`np.asfortranarray(a)` and `a` differ only in layout, never in `a[i, j]`. The binding
preserves that, which means honouring numpy's indexing rather than its strides.

---

## 4. Writability — read-only arrays are accepted

The bindings only read their inputs. Read-only arrays arise from `np.frombuffer`,
memory-mapped files, shared memory and `flags.writeable = False`, and there is no
reason to refuse them.

No input array is ever mutated. That is a promise, not an implementation detail: a
caller may pass the same array to `fit` and keep using it.

---

## 5. Shape and dimensionality — strict

`ValueError` for the wrong number of dimensions, a non-square `excitation`, or a
`baseline` whose length does not match. No broadcasting, no squeezing of length-1
axes, no promoting a scalar to a vector.

A 1-component multivariate process is written with shape `(1,)` and `(1, 1)`, not with
scalars. Silently accepting a scalar where a vector belongs is how a `d`-component call
becomes a 1-component one without anybody noticing.

---

## 6. Returned arrays belong to the caller

Anything the bindings return — simulated event times, a fitted excitation matrix, a
gradient — is a fresh array that the caller may mutate freely, and that never aliases
live Rust state. Nothing can dangle: the buffer is numpy's to free.

The precise mechanism differs by call, and `OWNDATA` is **not** a reliable way to check
this. A matrix built row by row owns its buffer outright and reports `OWNDATA == True`.
An array handed over from a Rust `Vec` is adopted by numpy through a base object, so
`OWNDATA` reports `False` while the memory is still numpy's and the `Vec` no longer
exists. Both are safe; only the second looks unusual.

What is promised, and what the tests assert, is the behaviour rather than the flag:
the array is writable, mutating it changes nothing else, and its lifetime does not
depend on any Rust value outliving it.

---

## 7. What none of this changes

Values. Rules 2 and 3 copy; rules 1, 4, 5 accept or refuse. Nothing here rounds,
casts, rescales or reorders a `float64` that reaches the Rust side. That is what makes
M3 step 3's bitwise equality across the boundary achievable, and that test is what
proves this document is describing the implementation rather than an aspiration.
