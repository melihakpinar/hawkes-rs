//! Multivariate Hawkes processes: simulation and maximum-likelihood estimation.
//!
//! # Status
//!
//! Pre-alpha. This crate is currently empty of algorithms by design: milestone M0
//! builds the verification harnesses (differential tests against `tick`, round-trip
//! property tests, finite-difference gradient checks) *before* there is anything to
//! verify. See `docs/verification-log.md` for evidence that those harnesses detect
//! the failures they are meant to detect.
//!
//! No intensity, likelihood, simulator or estimator lives here yet. They arrive in
//! M1, each accompanied by an approved derivation under `docs/derivations/`.

#![forbid(unsafe_code)]
