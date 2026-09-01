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
    /// `hawkes` rejects rather than silently sorting. `tick` accepts unsorted input and
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

    /// Fitting needs events. With none, the likelihood is monotone decreasing in
    /// `mu` and has no interior maximum, and `alpha` and `beta` do not appear in it
    /// at all.
    #[error("cannot fit: {events} events is not enough to identify three parameters")]
    InsufficientData { events: usize },

    /// The optimizer failed outright, as opposed to stopping without converging.
    #[error("the optimizer failed: {message}")]
    OptimizerFailed { message: String },

    /// A multivariate process needs at least one component.
    #[error("the process must have at least one component")]
    EmptyProcess,

    /// `baseline` has one entry per component and `excitation` is square in the same
    /// dimension.
    #[error(
        "dimension mismatch: {what} has length {actual}, expected {expected} \
         for a {dimension}-component process"
    )]
    DimensionMismatch {
        what: &'static str,
        actual: usize,
        expected: usize,
        dimension: usize,
    },

    /// The parameters and the observation are supplied independently and must
    /// describe the same process.
    ///
    /// Distinct from `DimensionMismatch`, which is about the internal shape of a
    /// single `Parameters` value: `baseline` against `excitation`. This one is two
    /// separate values disagreeing with each other.
    #[error(
        "parameters describe a {parameters}-component process but the observation \
         has {observation} components; they must agree"
    )]
    ProcessDimensionMismatch {
        parameters: usize,
        observation: usize,
    },

    /// Excitation entries may be zero — a component that excites nothing is
    /// ordinary — but never negative.
    #[error("excitation[{row}][{column}] must be non-negative and finite, got {value}")]
    InvalidExcitation {
        row: usize,
        column: usize,
        value: f64,
    },
}

impl Error {
    /// Not enough data to identify the parameters.
    pub(crate) fn insufficient_data(events: usize) -> Self {
        Error::InsufficientData { events }
    }
}
