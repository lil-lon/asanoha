use core::fmt::{Debug, Display};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Floating-point scalar used by every kernel; implemented for `f32` and `f64`.
pub trait Real:
    Copy
    + Send
    + Sync
    + PartialOrd
    + Default
    + Debug
    + Display
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + 'static
{
    /// `0`
    const ZERO: Self;
    /// `1`
    const ONE: Self;
    /// `2`
    const TWO: Self;
    /// `0.5`
    const HALF: Self;
    /// Machine epsilon.
    const EPSILON: Self;
    /// π
    const PI: Self;

    /// Converts from `f64`, rounding to nearest.
    fn from_f64(x: f64) -> Self;
    /// Converts to `f64` (exact for both implementations).
    fn to_f64(self) -> f64;
    /// Converts from `usize`, rounding to nearest.
    fn from_usize(n: usize) -> Self;

    /// Square root.
    fn sqrt(self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// Largest integer not greater than `self`.
    fn floor(self) -> Self;
    /// Smallest integer not less than `self`.
    fn ceil(self) -> Self;
    /// `e^self`
    fn exp(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// Sine.
    fn sin(self) -> Self;
    /// Cosine.
    fn cos(self) -> Self;
    /// Hyperbolic tangent.
    fn tanh(self) -> Self;
    /// Integer power.
    fn powi(self, n: i32) -> Self;
    /// `self * a + b`, fused where the target supports it.
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// Euclidean remainder, always in `[0, |m|)`.
    fn rem_euclid(self, m: Self) -> Self;
    /// Minimum of two values.
    fn min(self, o: Self) -> Self;
    /// Maximum of two values.
    fn max(self, o: Self) -> Self;
    /// `true` if neither infinite nor NaN.
    fn is_finite(self) -> bool;

    /// Logistic sigmoid `1 / (1 + e^-x)`.
    fn sigmoid(self) -> Self {
        Self::ONE / (Self::ONE + (-self).exp())
    }
    /// SiLU `x · sigmoid(x)`.
    fn silu(self) -> Self {
        self * self.sigmoid()
    }
}

macro_rules! impl_real {
    ($t:ident) => {
        impl Real for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const TWO: Self = 2.0;
            const HALF: Self = 0.5;
            const EPSILON: Self = $t::EPSILON;
            const PI: Self = core::$t::consts::PI;

            fn from_f64(x: f64) -> Self {
                x as $t
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn from_usize(n: usize) -> Self {
                n as $t
            }

            fn sqrt(self) -> Self {
                $t::sqrt(self)
            }
            fn abs(self) -> Self {
                $t::abs(self)
            }
            fn floor(self) -> Self {
                $t::floor(self)
            }
            fn ceil(self) -> Self {
                $t::ceil(self)
            }
            fn exp(self) -> Self {
                $t::exp(self)
            }
            fn ln(self) -> Self {
                $t::ln(self)
            }
            fn sin(self) -> Self {
                $t::sin(self)
            }
            fn cos(self) -> Self {
                $t::cos(self)
            }
            fn tanh(self) -> Self {
                $t::tanh(self)
            }
            fn powi(self, n: i32) -> Self {
                $t::powi(self, n)
            }
            fn mul_add(self, a: Self, b: Self) -> Self {
                $t::mul_add(self, a, b)
            }
            fn rem_euclid(self, m: Self) -> Self {
                $t::rem_euclid(self, m)
            }
            fn min(self, o: Self) -> Self {
                $t::min(self, o)
            }
            fn max(self, o: Self) -> Self {
                $t::max(self, o)
            }
            fn is_finite(self) -> bool {
                $t::is_finite(self)
            }
        }
    };
}

impl_real!(f32);
impl_real!(f64);

#[cfg(test)]
mod tests {
    use super::Real;

    fn check<T: Real>(eps: f64) {
        let close = |a: T, b: f64| (a.to_f64() - b).abs() <= eps * b.abs().max(1.0);

        assert!(close(T::ONE + T::ONE, 2.0));
        assert!(close(T::TWO * T::HALF, 1.0));
        assert!(close(T::PI, core::f64::consts::PI));
        assert!(close(T::from_usize(7), 7.0));
        assert!(close(T::from_f64(2.0).sqrt(), 2f64.sqrt()));
        assert!(close(T::from_f64(-1.5).floor(), -2.0));
        assert!(close(T::from_f64(-1.5).ceil(), -1.0));
        assert!(close(T::from_f64(-0.5).rem_euclid(T::ONE), 0.5));
        assert!(close(T::from_f64(3.0).powi(3), 27.0));
        assert!(close(
            T::from_f64(2.0).mul_add(T::from_f64(3.0), T::ONE),
            7.0
        ));
        assert!(close(T::ZERO.sigmoid(), 0.5));
        assert!(close(T::ZERO.silu(), 0.0));
        assert!(close(T::ONE.silu(), 1.0 / (1.0 + (-1f64).exp())));
        assert!(close(T::ONE.tanh(), 1f64.tanh()));
        assert!(close(T::ONE.exp().ln(), 1.0));

        assert!(T::ONE.is_finite());
        assert!(!(T::ONE / T::ZERO).is_finite());
        assert!(!T::from_f64(f64::NAN).is_finite());
        assert!(T::from_f64(-0.0).abs().to_f64().is_sign_positive());
    }

    #[test]
    fn f32_ops() {
        check::<f32>(1e-6);
    }

    #[test]
    fn f64_ops() {
        check::<f64>(1e-15);
    }

    #[test]
    fn silu_saturates_without_nan() {
        assert_eq!((-1000f64).silu(), -0.0);
        assert_eq!(1000f64.silu(), 1000.0);
        assert_eq!((-200f32).silu(), -0.0);
    }
}
