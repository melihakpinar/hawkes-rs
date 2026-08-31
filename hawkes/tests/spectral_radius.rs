//! Regression tests for `branching_ratio_spectral_radius` on **reducible** matrices.
//!
//! The routine originally returned the midpoint of the Collatz-Wielandt bracket. That
//! is correct for irreducible matrices, where both bounds converge to the Perron root,
//! and wrong for reducible ones, where the lower bound need not converge at all: for a
//! diagonal matrix `(A x)_i / x_i` equals `A[i][i]` at every step, so the bracket is
//! `[min A[i][i], max A[i][i]]` forever and its midpoint is not the spectral radius.
//!
//! That bug was found by a hand-written case, not by a sabotage. This file pins the
//! class so it cannot come back, with every expected value computed by hand.
//!
//! Reducible structures reachable in a Hawkes model, all of them ordinary:
//!
//! - **Diagonal** — components excite only themselves; `d` independent processes.
//! - **Block diagonal** — two groups that do not interact.
//! - **Triangular** — one group feeds another and gets nothing back, which is what a
//!   causal or hierarchical model looks like.
//!
//! # Sabotage
//!
//! Restoring the midpoint turned this file red on every reducible case while leaving
//! the irreducible ones green — the exact signature of the original defect. Recorded in
//! `docs/verification-log.md`.

use hawkes::multivariate::Parameters;

/// Agreement required against a hand-computed spectral radius.
///
/// The routine iterates to a fixed point, so for the cases below it is exact to
/// several more digits than this. The bound is loose enough not to be a convergence
/// test in disguise and far tighter than any of the errors it exists to catch: the
/// midpoint bug moves `diag(0.2, 0.7, 0.4)` from `0.7` to `0.45`.
const TOLERANCE: f64 = 1e-9;

fn radius(dimension: usize, excitation: Vec<f64>) -> f64 {
    Parameters::new(vec![1.0; dimension], excitation, 1.0)
        .expect("valid parameters")
        .branching_ratio_spectral_radius()
}

fn check(label: &str, dimension: usize, excitation: Vec<f64>, expected: f64) {
    let actual = radius(dimension, excitation);
    assert!(
        (actual - expected).abs() <= TOLERANCE,
        "{label}: got {actual:?}, hand calculation gives {expected:?} \
         (difference {:e})",
        (actual - expected).abs()
    );
}

/// A diagonal matrix's eigenvalues are its diagonal entries, so the spectral radius is
/// the largest of them. This is the case the original bug was found on.
#[test]
fn diagonal_matrices() {
    check("diag(0.5)", 1, vec![0.5], 0.5);
    check("diag(0.1, 0.9)", 2, vec![0.1, 0.0, 0.0, 0.9], 0.9);
    check("diag(0.9, 0.1)", 2, vec![0.9, 0.0, 0.0, 0.1], 0.9);
    check(
        "diag(0.2, 0.7, 0.4)",
        3,
        vec![0.2, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.4],
        0.7,
    );
    // The midpoint of the bracket here would be (0.05 + 0.95)/2 = 0.5.
    check("diag(0.05, 0.95)", 2, vec![0.05, 0.0, 0.0, 0.95], 0.95);
    check("zero matrix", 3, vec![0.0; 9], 0.0);
}

/// Two groups that never interact. The spectral radius is the larger of the blocks'.
#[test]
fn block_diagonal_matrices() {
    // Block A = [[0.1, 0.6], [0.05, 0.15]], eigenvalues 0.3 and -0.05, radius 0.3.
    // Block B = [0.8]. Overall radius 0.8.
    check(
        "blocks {0.3} then {0.8}",
        3,
        vec![0.10, 0.60, 0.0, 0.05, 0.15, 0.0, 0.0, 0.0, 0.80],
        0.8,
    );
    // Same blocks, other order: the answer must not depend on where the dominant
    // block sits, which a power iteration started at a uniform vector could get wrong.
    check(
        "blocks {0.8} then {0.3}",
        3,
        vec![0.80, 0.0, 0.0, 0.0, 0.10, 0.60, 0.0, 0.05, 0.15],
        0.8,
    );
    // Dominant block is the 2x2 this time: eigenvalues of [[0.2, 0.9], [0.4, 0.3]] are
    // (0.5 +- sqrt(0.01 + 1.44))/2 = (0.5 +- 1.2041594578792296)/2, radius
    // 0.8520797289396148.
    check(
        "blocks {0.85207...} then {0.1}",
        3,
        vec![0.20, 0.90, 0.0, 0.40, 0.30, 0.0, 0.0, 0.0, 0.10],
        0.8520797289396148,
    );
}

/// A triangular excitation matrix is a model where influence flows one way. Its
/// eigenvalues are its diagonal entries, however large the off-diagonal terms are —
/// which is what makes it a good test: the row sums are nowhere near the answer.
#[test]
fn triangular_matrices() {
    // Upper triangular: component 1 excites component 0, not the reverse. Row sums are
    // 10.5 and 0.2, and the radius is 0.5.
    check(
        "upper triangular, big off-diagonal",
        2,
        vec![0.5, 10.0, 0.0, 0.2],
        0.5,
    );
    // Lower triangular, the same the other way round.
    check(
        "lower triangular, big off-diagonal",
        2,
        vec![0.3, 0.0, 5.0, 0.6],
        0.6,
    );
    // 3x3 upper triangular; radius is the largest diagonal entry, 0.6.
    check(
        "3x3 upper triangular",
        3,
        vec![0.60, 2.0, 1.0, 0.0, 0.20, 3.0, 0.0, 0.0, 0.45],
        0.6,
    );
    // Dominant entry at the bottom rather than the top.
    check(
        "3x3 upper triangular, dominant last",
        3,
        vec![0.10, 2.0, 1.0, 0.0, 0.20, 3.0, 0.0, 0.0, 0.75],
        0.75,
    );
}

/// Agreement required for **defective** matrices — a repeated eigenvalue with too few
/// eigenvectors.
///
/// Power iteration converges sublinearly on these, like `1/k` rather than
/// geometrically, so the routine lands within about `3e-4` at its iteration cap rather
/// than at `1e-9`. That is a property of the method, not a defect, and it is
/// immaterial for deciding `rho < 1`.
///
/// It is pinned anyway. A test that simply omitted these cases would leave the
/// accuracy claim in the doc comment unchecked, and this is precisely the regime the
/// original bug lived in.
const DEFECTIVE_TOLERANCE: f64 = 1e-3;

/// Repeated eigenvalues, where the matrix has no basis of eigenvectors.
///
/// These were not in the first version of this file. Adding them — widening the case
/// set to the regime where the method is weakest, per CLAUDE.md §3 — found a second
/// bug: the routine returned `1.0` for the nilpotent matrix below, whose spectral
/// radius is `0`. It had a "the upper bound stopped moving" early exit, and on that
/// matrix the upper bound sits at 2 for two iterations before descending. A trivially
/// stationary process was being reported as explosive.
#[test]
fn defective_matrices() {
    let cases: [(&str, usize, Vec<f64>, f64); 5] = [
        (
            "strictly upper [[0, 3], [0, 0]]",
            2,
            vec![0.0, 3.0, 0.0, 0.0],
            0.0,
        ),
        (
            "Jordan block at 0.4, large coupling",
            2,
            vec![0.4, 7.0, 0.0, 0.4],
            0.4,
        ),
        (
            "Jordan block at 0.4, small coupling",
            2,
            vec![0.4, 0.01, 0.0, 0.4],
            0.4,
        ),
        (
            "3x3 equal diagonal, upper triangular",
            3,
            vec![0.5, 2.0, 1.0, 0.0, 0.5, 2.0, 0.0, 0.0, 0.5],
            0.5,
        ),
        (
            "nilpotent 3x3",
            3,
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            0.0,
        ),
    ];
    for (label, dimension, excitation, expected) in cases {
        let actual = radius(dimension, excitation);
        assert!(
            (actual - expected).abs() <= DEFECTIVE_TOLERANCE,
            "{label}: got {actual:?}, hand calculation gives {expected:?} \
             (difference {:e}, defective tolerance {DEFECTIVE_TOLERANCE:e})",
            (actual - expected).abs()
        );
    }
}

/// The consequence, which is what actually matters: a nilpotent excitation matrix is a
/// finite cascade that dies out, and the process is stationary.
#[test]
fn a_nilpotent_cascade_is_stationary() {
    let p = Parameters::new(
        vec![0.5, 0.6, 0.7],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        1.0,
    )
    .unwrap();
    assert!(
        p.is_stationary(),
        "a nilpotent excitation matrix has spectral radius 0; this reported {}",
        p.branching_ratio_spectral_radius()
    );
    let mean = p.stationary_mean_intensity().expect("stationary");
    // (I - N)^-1 = I + N + N^2 for a nilpotent N with N^3 = 0, so
    // Lambda = mu + N mu + N^2 mu componentwise:
    //   Lambda_0 = 0.5 + 0.6 + 0.7 = 1.8
    //   Lambda_1 = 0.6 + 0.7       = 1.3
    //   Lambda_2 = 0.7
    assert!((mean[0] - 1.8).abs() < 1e-9, "got {mean:?}");
    assert!((mean[1] - 1.3).abs() < 1e-9, "got {mean:?}");
    assert!((mean[2] - 0.7).abs() < 1e-9, "got {mean:?}");
}

/// Irreducible matrices, where both Collatz-Wielandt bounds converge. These must stay
/// green under the midpoint sabotage, or the test above proves nothing about which
/// class the fix applies to.
#[test]
fn irreducible_matrices_are_unaffected() {
    // Eigenvalues 0.3 and -0.05.
    check("2x2 irreducible", 2, vec![0.1, 0.6, 0.05, 0.15], 0.3);
    // Periodic: eigenvalues +-sqrt(1.8). Radius 1.3416407864998738. This is the case
    // that forced the `+ I` shift.
    check(
        "periodic 2x2",
        2,
        vec![0.0, 2.0, 0.9, 0.0],
        1.3416407864998738,
    );
    // Circulant with row sums 0.45; for a non-negative circulant matrix the Perron
    // root is the row sum.
    let d = 10;
    let mut excitation = vec![0.0; d * d];
    for i in 0..d {
        excitation[i * d + i] += 0.05;
        excitation[i * d + (i + 1) % d] += 0.30;
        excitation[i * d + (i + 3) % d] += 0.10;
    }
    check("circulant d=10", d, excitation, 0.45);
}

/// Stationarity is what the radius is used for, so the verdict is pinned too.
#[test]
fn stationarity_verdicts_on_reducible_matrices() {
    // Triangular with a diagonal entry above 1: not stationary, however small the
    // other entries are.
    let explosive = Parameters::new(vec![1.0, 1.0], vec![0.2, 0.0, 3.0, 1.4], 1.0).unwrap();
    assert!(
        !explosive.is_stationary(),
        "radius {} should be 1.4",
        explosive.branching_ratio_spectral_radius()
    );
    assert_eq!(explosive.stationary_mean_intensity(), None);

    // Diagonal, all entries below 1: stationary, and the mean intensity is
    // mu_i / (1 - alpha_ii) component by component because the components are
    // independent.
    let independent = Parameters::new(vec![0.6, 0.8], vec![0.5, 0.0, 0.0, 0.2], 1.0).unwrap();
    assert!(independent.is_stationary());
    let mean = independent.stationary_mean_intensity().unwrap();
    assert!((mean[0] - 0.6 / 0.5).abs() < 1e-12, "got {mean:?}");
    assert!((mean[1] - 0.8 / 0.8).abs() < 1e-12, "got {mean:?}");
}
