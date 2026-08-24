//! Python bindings for `hawk`.
//!
//! The Python API mirrors the Rust one deliberately and adds nothing. Anything that
//! exists on one side and not the other would be outside the reach of the fixture
//! corpus, which is what polices both.
//!
//! # What this layer must not do
//!
//! Change a number. Every array policy is decided in `docs/python-array-handling.md`
//! and none of them casts, rounds, rescales or reorders an `f64` on its way to Rust.
//! `hawk-python/tests/test_bit_identity.py` proves it: the negative log-likelihood
//! computed through Python is compared **bit for bit** against the value the Rust
//! test suite produces for the same input.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use hawk::Error;

/// Maps a `hawk::Error` onto a Python exception.
///
/// Every variant is listed explicitly rather than falling through a catch-all, so a
/// new variant is a compile error here rather than a silent `ValueError` in the
/// interpreter. The mapping is documented in `python/hawk/__init__.py`.
fn to_py_error(error: Error) -> PyErr {
    let message = error.to_string();
    match error {
        // Contract violations a caller can fix by changing their input.
        Error::NonPositiveParameter { .. }
        | Error::NonFiniteParameter { .. }
        | Error::UnsortedEvents { .. }
        | Error::EventOutsideWindow { .. }
        | Error::NonFiniteEvent { .. }
        | Error::InvalidHorizon { .. }
        | Error::InsufficientData { .. }
        | Error::EmptyProcess
        | Error::DimensionMismatch { .. }
        | Error::InvalidExcitation { .. } => PyValueError::new_err(message),
        // The optimizer failed as a solver, not because the input was invalid.
        Error::OptimizerFailed { .. } => PyRuntimeError::new_err(message),
    }
}

use pyo3::exceptions::PyRuntimeError;

/// Reads a 1-D `float64` array into an owned `Vec<f64>`.
///
/// Copies unconditionally. Non-contiguous input is accepted and copied
/// (`python-array-handling.md` §2); read-only input is accepted (§4). `dtype` is
/// enforced by the signature: `PyReadonlyArray1<f64>` refuses anything that is not
/// already `float64`, which is §1.
fn vector_from(array: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    array.as_array().iter().copied().collect()
}

/// Reads a 2-D `float64` array into a row-major `Vec<f64>`.
///
/// **Indexed logically, never by stride.** `as_array()` yields an `ndarray` view that
/// honours the array's own indexing, so an F-ordered input is read as `[i, j]` and
/// written out in C order. The buffer is never reinterpreted, because doing so would
/// hand Rust the transpose of the excitation matrix — the exact failure
/// `conventions.md` C6 exists to prevent. See `python-array-handling.md` §3.
fn matrix_from(array: &PyReadonlyArray2<'_, f64>) -> (usize, usize, Vec<f64>) {
    let view = array.as_array();
    let rows = view.nrows();
    let columns = view.ncols();
    let mut flat = Vec::with_capacity(rows * columns);
    for i in 0..rows {
        for j in 0..columns {
            flat.push(view[[i, j]]);
        }
    }
    (rows, columns, flat)
}

// ------------------------------------------------------------------- univariate

/// Parameters of a univariate exponential-kernel Hawkes process.
#[pyclass(
    module = "hawk._hawk",
    name = "UnivariateParameters",
    frozen,
    from_py_object
)]
#[derive(Clone)]
struct UnivariateParameters {
    inner: hawk::univariate::Parameters,
}

#[pymethods]
impl UnivariateParameters {
    #[new]
    fn new(baseline: f64, excitation: f64, decay: f64) -> PyResult<Self> {
        Ok(Self {
            inner: hawk::univariate::Parameters::new(baseline, excitation, decay)
                .map_err(to_py_error)?,
        })
    }

    #[getter]
    fn baseline(&self) -> f64 {
        self.inner.baseline()
    }

    #[getter]
    fn excitation(&self) -> f64 {
        self.inner.excitation()
    }

    #[getter]
    fn decay(&self) -> f64 {
        self.inner.decay()
    }

    fn branching_ratio(&self) -> f64 {
        self.inner.branching_ratio()
    }

    fn is_stationary(&self) -> bool {
        self.inner.is_stationary()
    }

    fn stationary_mean_intensity(&self) -> Option<f64> {
        self.inner.stationary_mean_intensity()
    }

    fn __repr__(&self) -> String {
        format!(
            "UnivariateParameters(baseline={:?}, excitation={:?}, decay={:?})",
            self.inner.baseline(),
            self.inner.excitation(),
            self.inner.decay()
        )
    }
}

/// Outcome of a univariate fit.
#[pyclass(
    module = "hawk._hawk",
    name = "UnivariateFit",
    frozen,
    skip_from_py_object
)]
struct UnivariateFit {
    #[pyo3(get)]
    parameters: UnivariateParameters,
    #[pyo3(get)]
    negative_log_likelihood: f64,
    #[pyo3(get)]
    iterations: u64,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    gradient_norm: f64,
    #[pyo3(get)]
    objective_evaluations: u64,
    #[pyo3(get)]
    gradient_evaluations: u64,
}

#[pymethods]
impl UnivariateFit {
    fn branching_ratio(&self) -> f64 {
        self.parameters.inner.branching_ratio()
    }

    fn is_stationary(&self) -> bool {
        self.parameters.inner.is_stationary()
    }
}

#[pyfunction]
#[pyo3(name = "univariate_negative_log_likelihood")]
fn univariate_negative_log_likelihood(
    parameters: &UnivariateParameters,
    times: PyReadonlyArray1<'_, f64>,
    horizon: f64,
) -> PyResult<f64> {
    let owned = vector_from(&times);
    let observation = hawk::univariate::Observation::new(&owned, horizon).map_err(to_py_error)?;
    Ok(hawk::univariate::negative_log_likelihood(
        &parameters.inner,
        &observation,
    ))
}

#[pyfunction]
#[pyo3(name = "univariate_negative_log_likelihood_and_gradient")]
fn univariate_negative_log_likelihood_and_gradient(
    parameters: &UnivariateParameters,
    times: PyReadonlyArray1<'_, f64>,
    horizon: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    let owned = vector_from(&times);
    let observation = hawk::univariate::Observation::new(&owned, horizon).map_err(to_py_error)?;
    let (value, gradient) =
        hawk::univariate::negative_log_likelihood_and_gradient(&parameters.inner, &observation);
    Ok((
        value,
        gradient.baseline,
        gradient.excitation,
        gradient.decay,
    ))
}

#[pyfunction]
#[pyo3(name = "univariate_simulate")]
fn univariate_simulate<'py>(
    py: Python<'py>,
    parameters: &UnivariateParameters,
    horizon: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let times =
        hawk::univariate::simulate(&parameters.inner, horizon, &mut rng).map_err(to_py_error)?;
    Ok(PyArray1::from_vec(py, times))
}

#[pyfunction]
#[pyo3(name = "univariate_fit")]
fn univariate_fit(times: PyReadonlyArray1<'_, f64>, horizon: f64) -> PyResult<UnivariateFit> {
    let owned = vector_from(&times);
    let observation = hawk::univariate::Observation::new(&owned, horizon).map_err(to_py_error)?;
    let fit = hawk::univariate::fit(&observation).map_err(to_py_error)?;
    Ok(UnivariateFit {
        parameters: UnivariateParameters {
            inner: fit.parameters,
        },
        negative_log_likelihood: fit.negative_log_likelihood,
        iterations: fit.iterations,
        converged: fit.converged,
        gradient_norm: fit.gradient_norm,
        objective_evaluations: fit.objective_evaluations,
        gradient_evaluations: fit.gradient_evaluations,
    })
}

// ----------------------------------------------------------------- multivariate

/// Parameters of a `d`-component exponential-kernel Hawkes process.
#[pyclass(
    module = "hawk._hawk",
    name = "MultivariateParameters",
    frozen,
    from_py_object
)]
#[derive(Clone)]
struct MultivariateParameters {
    inner: hawk::multivariate::Parameters,
}

#[pymethods]
impl MultivariateParameters {
    #[new]
    fn new(
        baseline: PyReadonlyArray1<'_, f64>,
        excitation: PyReadonlyArray2<'_, f64>,
        decay: f64,
    ) -> PyResult<Self> {
        let baseline = vector_from(&baseline);
        let (rows, columns, flat) = matrix_from(&excitation);
        if rows != columns {
            return Err(PyValueError::new_err(format!(
                "excitation must be square, got shape ({rows}, {columns})"
            )));
        }
        Ok(Self {
            inner: hawk::multivariate::Parameters::new(baseline, flat, decay)
                .map_err(to_py_error)?,
        })
    }

    #[getter]
    fn baseline<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.baseline())
    }

    /// The excitation matrix, `excitation[i][j]` meaning "j excites i"
    /// (`conventions.md` C6). Returned in C order as a fresh array.
    #[getter]
    fn excitation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = self.inner.dimension();
        let flat = self.inner.excitation().to_vec();
        PyArray1::from_vec(py, flat).reshape([d, d])
    }

    #[getter]
    fn decay(&self) -> f64 {
        self.inner.decay()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn branching_ratio_spectral_radius(&self) -> f64 {
        self.inner.branching_ratio_spectral_radius()
    }

    fn is_stationary(&self) -> bool {
        self.inner.is_stationary()
    }

    fn stationary_mean_intensity<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .stationary_mean_intensity()
            .map(|v| PyArray1::from_vec(py, v))
    }

    fn __repr__(&self) -> String {
        format!(
            "MultivariateParameters(dimension={}, decay={:?})",
            self.inner.dimension(),
            self.inner.decay()
        )
    }
}

/// Outcome of a multivariate fit.
#[pyclass(
    module = "hawk._hawk",
    name = "MultivariateFit",
    frozen,
    skip_from_py_object
)]
struct MultivariateFit {
    #[pyo3(get)]
    parameters: MultivariateParameters,
    #[pyo3(get)]
    negative_log_likelihood: f64,
    #[pyo3(get)]
    iterations: u64,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    gradient_norm: f64,
    #[pyo3(get)]
    objective_evaluations: u64,
    #[pyo3(get)]
    gradient_evaluations: u64,
}

#[pymethods]
impl MultivariateFit {
    fn branching_ratio_spectral_radius(&self) -> f64 {
        self.parameters.inner.branching_ratio_spectral_radius()
    }

    fn is_stationary(&self) -> bool {
        self.parameters.inner.is_stationary()
    }
}

/// Converts a Python sequence of 1-D `float64` arrays into per-component events.
///
/// Each component is read with the same policy as any other array
/// (`python-array-handling.md`): `float64` only, copied, read-only accepted.
fn events_from(events: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    let mut out = Vec::new();
    for item in events.try_iter()? {
        let item = item?;
        let array: PyReadonlyArray1<'_, f64> = item.extract().map_err(|_| {
            PyTypeError::new_err(
                "each component must be a 1-D numpy array of dtype float64; convert \
                 with np.asarray(component, dtype=np.float64)",
            )
        })?;
        out.push(vector_from(&array));
    }
    if out.is_empty() {
        return Err(to_py_error(Error::EmptyProcess));
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "multivariate_negative_log_likelihood")]
fn multivariate_negative_log_likelihood(
    parameters: &MultivariateParameters,
    events: &Bound<'_, PyAny>,
    horizon: f64,
) -> PyResult<f64> {
    let owned = events_from(events)?;
    let observation = hawk::multivariate::Observation::new(&owned, horizon).map_err(to_py_error)?;
    Ok(hawk::multivariate::negative_log_likelihood(
        &parameters.inner,
        &observation,
    ))
}

/// `(value, d/d baseline, d/d excitation, d/d decay)` — the fields of the Rust
/// `multivariate::Gradient`, in order. The Python layer gives them names.
type MultivariateGradientTuple<'py> = (
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
);

#[pyfunction]
#[pyo3(name = "multivariate_negative_log_likelihood_and_gradient")]
fn multivariate_negative_log_likelihood_and_gradient<'py>(
    py: Python<'py>,
    parameters: &MultivariateParameters,
    events: &Bound<'_, PyAny>,
    horizon: f64,
) -> PyResult<MultivariateGradientTuple<'py>> {
    let owned = events_from(events)?;
    let observation = hawk::multivariate::Observation::new(&owned, horizon).map_err(to_py_error)?;
    let (value, gradient) =
        hawk::multivariate::negative_log_likelihood_and_gradient(&parameters.inner, &observation);
    let d = parameters.inner.dimension();
    let excitation = PyArray1::from_vec(py, gradient.excitation).reshape([d, d])?;
    Ok((
        value,
        PyArray1::from_vec(py, gradient.baseline),
        excitation,
        gradient.decay,
    ))
}

#[pyfunction]
#[pyo3(name = "multivariate_compensator_at_events")]
fn multivariate_compensator_at_events<'py>(
    py: Python<'py>,
    parameters: &MultivariateParameters,
    events: &Bound<'_, PyAny>,
    horizon: f64,
) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
    let owned = events_from(events)?;
    let observation = hawk::multivariate::Observation::new(&owned, horizon).map_err(to_py_error)?;
    Ok(
        hawk::multivariate::compensator_at_events(&parameters.inner, &observation)
            .into_iter()
            .map(|component| PyArray1::from_vec(py, component))
            .collect(),
    )
}

#[pyfunction]
#[pyo3(name = "multivariate_simulate")]
fn multivariate_simulate<'py>(
    py: Python<'py>,
    parameters: &MultivariateParameters,
    horizon: f64,
    seed: u64,
) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let events =
        hawk::multivariate::simulate(&parameters.inner, horizon, &mut rng).map_err(to_py_error)?;
    Ok(events
        .into_iter()
        .map(|component| PyArray1::from_vec(py, component))
        .collect())
}

#[pyfunction]
#[pyo3(name = "multivariate_fit")]
fn multivariate_fit(events: &Bound<'_, PyAny>, horizon: f64) -> PyResult<MultivariateFit> {
    let owned = events_from(events)?;
    let observation = hawk::multivariate::Observation::new(&owned, horizon).map_err(to_py_error)?;
    let fit = hawk::multivariate::fit(&observation).map_err(to_py_error)?;
    Ok(MultivariateFit {
        parameters: MultivariateParameters {
            inner: fit.parameters,
        },
        negative_log_likelihood: fit.negative_log_likelihood,
        iterations: fit.iterations,
        converged: fit.converged,
        gradient_norm: fit.gradient_norm,
        objective_evaluations: fit.objective_evaluations,
        gradient_evaluations: fit.gradient_evaluations,
    })
}

#[pymodule]
fn _hawk(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<UnivariateParameters>()?;
    module.add_class::<UnivariateFit>()?;
    module.add_class::<MultivariateParameters>()?;
    module.add_class::<MultivariateFit>()?;
    module.add_function(wrap_pyfunction!(
        univariate_negative_log_likelihood,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        univariate_negative_log_likelihood_and_gradient,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(univariate_simulate, module)?)?;
    module.add_function(wrap_pyfunction!(univariate_fit, module)?)?;
    module.add_function(wrap_pyfunction!(
        multivariate_negative_log_likelihood,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        multivariate_negative_log_likelihood_and_gradient,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        multivariate_compensator_at_events,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(multivariate_simulate, module)?)?;
    module.add_function(wrap_pyfunction!(multivariate_fit, module)?)?;
    Ok(())
}
