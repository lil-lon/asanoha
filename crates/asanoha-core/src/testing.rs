//! Dependency-free helpers for tests: a seeded RNG, central differences and
//! [`assert_close!`](crate::assert_close).

use crate::Real;

/// Marsaglia's xorshift64 (shifts 13, 7, 17). Deterministic for a given seed.
#[derive(Debug, Clone)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Creates the generator. Panics if `seed == 0`, which is a fixed point.
    pub fn new(seed: u64) -> Self {
        assert_ne!(seed, 0, "xorshift64 seed must be nonzero");
        Self { state: seed }
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal sample (Box–Muller; one sample per call).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }

    /// Uniformly distributed rotation matrix, from a unit quaternion drawn
    /// uniformly on the 3-sphere.
    pub fn rotation(&mut self) -> [[f64; 3]; 3] {
        let q = [self.normal(), self.normal(), self.normal(), self.normal()];
        let norm = q.iter().map(|c| c * c).sum::<f64>().sqrt();
        let [w, x, y, z] = q.map(|c| c / norm);
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }

    /// Random permutation of `0..n` (Fisher–Yates).
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        p
    }
}

/// Gradient of `f` at `x` by central differences with step `h`.
pub fn finite_difference(mut f: impl FnMut(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<f64> {
    let mut xh = x.to_vec();
    (0..x.len())
        .map(|k| {
            xh[k] = x[k] + h;
            let plus = f(&xh);
            xh[k] = x[k] - h;
            let minus = f(&xh);
            xh[k] = x[k];
            (plus - minus) / (2.0 * h)
        })
        .collect()
}

/// Values accepted by [`assert_close!`](crate::assert_close): a scalar or a
/// slice-like collection of [`Real`].
pub trait CloseOperand {
    /// The values widened to `f64`.
    fn values(&self) -> Vec<f64>;
}

impl CloseOperand for f32 {
    fn values(&self) -> Vec<f64> {
        vec![self.to_f64()]
    }
}

impl CloseOperand for f64 {
    fn values(&self) -> Vec<f64> {
        vec![*self]
    }
}

impl<T: Real> CloseOperand for [T] {
    fn values(&self) -> Vec<f64> {
        self.iter().map(|x| x.to_f64()).collect()
    }
}

impl<T: Real, const N: usize> CloseOperand for [T; N] {
    fn values(&self) -> Vec<f64> {
        self.as_slice().values()
    }
}

impl<T: Real> CloseOperand for Vec<T> {
    fn values(&self) -> Vec<f64> {
        self.as_slice().values()
    }
}

impl<A: CloseOperand + ?Sized> CloseOperand for &A {
    fn values(&self) -> Vec<f64> {
        (**self).values()
    }
}

/// Panics unless `|a − b| ≤ atol + rtol·|b|` elementwise; the message names
/// the worst index. Prefer the [`assert_close!`](crate::assert_close) macro.
#[track_caller]
pub fn assert_close<A, B>(a: &A, b: &B, atol: f64, rtol: f64)
where
    A: CloseOperand + ?Sized,
    B: CloseOperand + ?Sized,
{
    let (a, b) = (a.values(), b.values());
    assert_eq!(a.len(), b.len(), "length mismatch");
    let mut worst: Option<(usize, f64)> = None;
    for (k, (&x, &y)) in a.iter().zip(&b).enumerate() {
        let excess = (x - y).abs() - (atol + rtol * y.abs());
        let bad = excess > 0.0 || excess.is_nan();
        if bad && worst.is_none_or(|(_, e)| excess > e || excess.is_nan()) {
            worst = Some((k, excess));
        }
    }
    if let Some((k, _)) = worst {
        panic!(
            "not close at index {k}: {} vs {} (|diff| = {:e}, atol = {atol:e}, rtol = {rtol:e})",
            a[k],
            b[k],
            (a[k] - b[k]).abs()
        );
    }
}

/// `assert_close!(a, b, atol, rtol)`: scalars, slices, arrays or `Vec`s of
/// [`Real`] are close elementwise (`|a − b| ≤ atol + rtol·|b|`).
#[macro_export]
macro_rules! assert_close {
    ($a:expr, $b:expr, $atol:expr, $rtol:expr $(,)?) => {
        $crate::testing::assert_close(&$a, &$b, $atol, $rtol)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xorshift_reference_sequence() {
        // x = 1: 1 ^ (1 << 13) = 8193; 8193 ^ (8193 >> 7) = 8257;
        // 8257 ^ (8257 << 17) = 8257 + 8257 · 2^17 = 1082269761.
        let mut rng = XorShift64::new(1);
        assert_eq!(rng.next_u64(), 1_082_269_761);
    }

    #[test]
    fn next_f64_in_unit_interval() {
        let mut rng = XorShift64::new(42);
        let mut sum = 0.0;
        for _ in 0..10_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
            sum += x;
        }
        assert!((sum / 10_000.0 - 0.5).abs() < 0.02);
    }

    // Statistical bounds below are five standard errors of the estimate.

    #[test]
    fn normal_moments() {
        let n = 200_000;
        for seed in [1, 42, 12345] {
            let mut rng = XorShift64::new(seed);
            let (mut s1, mut s2) = (0.0, 0.0);
            for _ in 0..n {
                let z = rng.normal();
                s1 += z;
                s2 += z * z;
            }
            let mean = s1 / n as f64;
            let var = s2 / n as f64 - mean * mean;
            assert!(mean.abs() < 5.0 * (1.0 / n as f64).sqrt(), "mean {mean}");
            assert!(
                (var - 1.0).abs() < 5.0 * (2.0 / n as f64).sqrt(),
                "var {var}"
            );
        }
    }

    #[test]
    fn rotation_is_proper_orthogonal() {
        for seed in [1, 42, 12345] {
            let mut rng = XorShift64::new(seed);
            for _ in 0..1000 {
                let r = rng.rotation();
                for i in 0..3 {
                    for j in 0..3 {
                        let dot: f64 = (0..3).map(|k| r[i][k] * r[j][k]).sum();
                        let id = if i == j { 1.0 } else { 0.0 };
                        crate::assert_close!(dot, id, 2e-15, 0.0);
                    }
                }
                let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                    - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                    + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
                crate::assert_close!(det, 1.0, 2e-15, 0.0);
            }
        }
    }

    #[test]
    fn rotation_moments_match_uniform() {
        // Under the uniform measure each entry has mean 0, and its square
        // follows Beta(1/2, 1): mean 1/3, variance 4/45.
        let n = 30_000;
        let mut rng = XorShift64::new(7);
        let (mut s1, mut s2) = ([[0.0; 3]; 3], [[0.0; 3]; 3]);
        for _ in 0..n {
            let r = rng.rotation();
            for i in 0..3 {
                for j in 0..3 {
                    s1[i][j] += r[i][j];
                    s2[i][j] += r[i][j] * r[i][j];
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                let m1 = s1[i][j] / n as f64;
                let m2 = s2[i][j] / n as f64;
                assert!(
                    m1.abs() < 5.0 * (1.0 / 3.0 / n as f64).sqrt(),
                    "E[R{i}{j}] = {m1}"
                );
                assert!(
                    (m2 - 1.0 / 3.0).abs() < 5.0 * (4.0 / 45.0 / n as f64).sqrt(),
                    "E[R{i}{j}^2] = {m2}"
                );
            }
        }
    }

    #[test]
    fn permutation_is_a_permutation() {
        for seed in [1, 42, 12345] {
            let mut rng = XorShift64::new(seed);
            for n in [0, 1, 2, 7, 100] {
                let mut p = rng.permutation(n);
                p.sort_unstable();
                assert_eq!(p, (0..n).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn permutation_frequencies_match_uniform() {
        let n = 60_000;
        let mut rng = XorShift64::new(3);
        let mut counts = std::collections::BTreeMap::new();
        for _ in 0..n {
            *counts.entry(rng.permutation(3)).or_insert(0usize) += 1;
        }
        assert_eq!(counts.len(), 6);
        let expected = n as f64 / 6.0;
        let bound = 5.0 * (n as f64 * (1.0 / 6.0) * (5.0 / 6.0)).sqrt();
        for (p, &c) in &counts {
            assert!((c as f64 - expected).abs() < bound, "{p:?}: {c}");
        }
    }

    #[test]
    fn finite_difference_matches_analytic_gradient() {
        let f = |x: &[f64]| x[0].sin() * x[1] * x[1] + x[2].exp();
        let grad = |x: &[f64]| {
            [
                x[0].cos() * x[1] * x[1],
                2.0 * x[0].sin() * x[1],
                x[2].exp(),
            ]
        };
        let mut rng = XorShift64::new(11);
        for _ in 0..100 {
            let x: Vec<f64> = (0..3).map(|_| rng.normal()).collect();
            crate::assert_close!(finite_difference(f, &x, 1e-5), grad(&x), 1e-9, 0.0);
        }
    }

    #[test]
    fn close_passes_on_scalars_and_slices() {
        crate::assert_close!(1.0f64, 1.0 + 1e-13, 1e-12, 0.0);
        crate::assert_close!(1.0f32, 1.0f64, 0.0, 0.0);
        crate::assert_close!([1.0f64, 2.0], vec![1.0, 2.0 + 1e-9], 0.0, 1e-9);
        crate::assert_close!(vec![1.0f32][..], [1.0f32], 0.0, 0.0);
    }

    #[test]
    fn close_accepts_borrowed_slices() {
        let v = vec![1.0f64, 2.0];
        let a: &[f64] = &v;
        crate::assert_close!(a, v.as_slice(), 0.0, 0.0);

        let f = crate::Features::<f32>::zeros("2x0e".parse().unwrap(), crate::Layout::IrMul, 1);
        crate::assert_close!(f.block(0, 0), [0.0f32, 0.0], 0.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "not close at index 1")]
    fn close_reports_worst_index() {
        crate::assert_close!([1.0f64, 5.0, 2.1], [1.0f64, 2.0, 2.0], 1e-12, 0.0);
    }

    #[test]
    #[should_panic(expected = "not close at index 0")]
    fn close_rejects_nan() {
        crate::assert_close!(f64::NAN, 0.0f64, 1.0, 1.0);
    }
}
