//! Multivariate Hawkes processes: simulation and maximum-likelihood estimation.
//!
//! # Status
//!
//! Pre-alpha. [`univariate`] and [`multivariate`] implement the exponential-kernel
//! process in one and `d` dimensions respectively. At `d = 1` the two agree bitwise;
//! `univariate` is kept because it is measurably cheaper and is the reference the
//! multivariate path is checked against.
//!
//! # Conventions
//!
//! Every convention this crate encodes is pinned by experiment in
//! `docs/derivations/conventions.md`, and every formula is transcribed from an
//! approved derivation. The two that matter most to a caller:
//!
//! - The kernel is `alpha * beta * exp(-beta * t)`, so `alpha` is the **branching
//!   ratio** directly, not `alpha / beta` (C1, C2). [Laub2015] uses the other
//!   parametrization; the map is `alpha_Laub = alpha * beta`.
//! - The observation window `[0, T]` is supplied by the caller and never inferred
//!   from the data (C5). Inferring it biases the baseline upward.

#![forbid(unsafe_code)]

mod error;
pub mod multivariate;
pub mod univariate;

pub use error::Error;
