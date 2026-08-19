//! Multivariate Hawkes processes: simulation and maximum-likelihood estimation.
//!
//! # Status
//!
//! Pre-alpha. Only the univariate exponential-kernel process is implemented; see
//! [`univariate`]. The multivariate case arrives in M2.
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
pub mod univariate;

pub use error::Error;
