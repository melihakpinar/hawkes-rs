//! Error type for the crate.
//!
//! Invalid input is an error value, never a panic (CLAUDE.md §5).

use thiserror::Error;

/// Anything a caller can get wrong.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Error {
    /// `mu`, `alpha` and `beta` are all strictly positive
    /// (`docs/derivations/univariate_loglikelihood.md` §1).
    #[error("{name} must be strictly positive, got {value}")]
    NonPositiveParameter { name: &'static str, value: f64 },

    #[error("{name} must be finite, got {value}")]
    NonFiniteParameter { name: &'static str, value: f64 },

    /// Timestamps must be ascending within a component
    /// (`docs/derivations/conventions.md` C8).
    ///
    /// `hawk` rejects rather than silently sorting. `tick` accepts unsorted input and
    /// returns a different, wrong number for each ordering, with no warning; a caller
    /// who supplies unsorted data has probably misunderstood it, and sorting on their
    /// behalf hides that.
    #[error(
        "timestamps must be ascending: index {index} is {current}, \
         which is before index {previous_index} at {previous}"
    )]
    UnsortedEvents {
        index: usize,
        previous_index: usize,
        previous: f64,
        current: f64,
    },

    /// Every timestamp lies in `[0, horizon]`, endpoints included
    /// (`docs/derivations/conventions.md` C8).
    #[error("timestamp {time} at index {index} is outside the window [0, {horizon}]")]
    EventOutsideWindow {
        index: usize,
        time: f64,
        horizon: f64,
    },

    #[error("timestamp at index {index} is not finite")]
    NonFiniteEvent { index: usize },

    /// The observation window is supplied by the caller and never inferred
    /// (`docs/derivations/conventions.md` C5).
    #[error("the observation horizon must be strictly positive and finite, got {horizon}")]
    InvalidHorizon { horizon: f64 },
}
