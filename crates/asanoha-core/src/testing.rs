//! Dependency-free helpers for tests: a seeded RNG and [`assert_close!`](crate::assert_close).

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
