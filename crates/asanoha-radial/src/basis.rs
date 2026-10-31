use asanoha_core::Real;

/// `R_n(r) = B_n(r) · f_c(r / r_max)` for `n = 1..=n_basis`.
///
/// The Bessel basis is `B_n(r) = (2 / r_max) · sin(n·π·r / r_max) / r`, with
/// its limit `B_n(0) = (2 / r_max) · n·π / r_max`. The polynomial cutoff is
/// `f_c(x) = 1 − (p+1)(p+2)/2 · x^p + p(p+2) · x^{p+1} − p(p+1)/2 · x^{p+2}`
/// for `x < 1` and zero otherwise.
#[derive(Debug, Clone)]
pub struct RadialBasis<T> {
    /// Cutoff radius.
    pub r_max: T,
    /// Number of basis functions.
    pub n_basis: usize,
    /// Exponent of the polynomial cutoff.
    pub p: i32,
}

impl<T: Real> RadialBasis<T> {
    /// `f_c(r / r_max)` and its derivative with respect to `r`.
    fn cutoff(&self, r: T) -> (T, T) {
        let x = r / self.r_max;
        if x >= T::ONE {
            return (T::ZERO, T::ZERO);
        }
        let p = T::from_f64(f64::from(self.p));
        let a = (p + T::ONE) * (p + T::TWO) * T::HALF;
        let b = p * (p + T::TWO);
        let c = p * (p + T::ONE) * T::HALF;
        let xp1 = x.powi(self.p - 1);
        let xp = xp1 * x;
        let f = T::ONE - a * xp + b * xp * x - c * xp * x * x;
        let df = xp1 * (-a * p + b * (p + T::ONE) * x - c * (p + T::TWO) * x * x) / self.r_max;
        (f, df)
    }

    /// `n·π / r_max` of the `n`-th basis function, `n ≥ 1`.
    fn wavenumber(&self, n: usize) -> T {
        T::from_usize(n) * T::PI / self.r_max
    }

    /// Writes `R_n(r)` into `out[..n_basis]`.
    pub fn eval(&self, r: T, out: &mut [T]) {
        let (f, _) = self.cutoff(r);
        let pre = T::TWO / self.r_max;
        for (n, o) in out[..self.n_basis].iter_mut().enumerate() {
            let k = self.wavenumber(n + 1);
            let b = if r == T::ZERO {
                pre * k
            } else {
                pre * (k * r).sin() / r
            };
            *o = b * f;
        }
    }

    /// Writes `R_n(r)` into `out[..n_basis]` and `dR_n/dr` into
    /// `d_out[..n_basis]`.
    pub fn eval_with_grad(&self, r: T, out: &mut [T], d_out: &mut [T]) {
        let (f, df) = self.cutoff(r);
        let pre = T::TWO / self.r_max;
        let pairs = out[..self.n_basis]
            .iter_mut()
            .zip(&mut d_out[..self.n_basis]);
        for (n, (o, d)) in pairs.enumerate() {
            let k = self.wavenumber(n + 1);
            let (b, db) = if r == T::ZERO {
                (pre * k, T::ZERO)
            } else {
                let (s, c) = ((k * r).sin(), (k * r).cos());
                (pre * s / r, pre * (k * r * c - s) / (r * r))
            };
            *o = b * f;
            *d = db * f + b * df;
        }
    }

    /// `R_n` of every distance: `out` is `[dist.len()][n_basis]`.
    pub fn eval_batch(&self, dist: &[T], out: &mut [T]) {
        assert_eq!(out.len(), dist.len() * self.n_basis);
        for (&r, o) in dist.iter().zip(out.chunks_exact_mut(self.n_basis)) {
            self.eval(r, o);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asanoha_core::assert_close;
    use asanoha_core::testing::{XorShift64, finite_difference};

    /// nequix-mp-1's radial basis.
    fn basis<T: Real>() -> RadialBasis<T> {
        RadialBasis {
            r_max: T::from_f64(6.0),
            n_basis: 8,
            p: 6,
        }
    }

    fn eval(r: f64) -> [f64; 8] {
        let mut out = [0.0; 8];
        basis().eval(r, &mut out);
        out
    }

    fn eval_with_grad(r: f64) -> ([f64; 8], [f64; 8]) {
        let (mut out, mut d_out) = ([0.0; 8], [0.0; 8]);
        basis().eval_with_grad(r, &mut out, &mut d_out);
        (out, d_out)
    }

    #[test]
    fn known_values_at_half_cutoff() {
        // B_n(3) = (2/6) · sin(n·π/2) / 3, f_c(1/2) = 1 − 28/64 + 48/128 − 21/256.
        let fc = 219.0 / 256.0;
        let expected: Vec<f64> = (1..=8)
            .map(|n| (n as f64 * core::f64::consts::FRAC_PI_2).sin() / 9.0 * fc)
            .collect();
        assert_close!(eval(3.0), expected, 1e-16, 0.0);
    }

    #[test]
    fn zero_from_cutoff() {
        for r in [6.0, 6.0 + 1e-12, 7.0, 100.0] {
            assert_eq!(eval_with_grad(r), ([0.0; 8], [0.0; 8]));
        }
    }

    #[test]
    fn vanishes_smoothly_at_cutoff() {
        // f_c, f_c' and f_c'' vanish at x = 1, so R ~ (1 − x)³ and R' ~ (1 − x)².
        // Here |R| ≤ 7e-19 and |R'| ≤ 5e-14; a cutoff ~ (1 − x)² gives
        // |R'| ≈ 7e-11.
        let (out, d_out) = eval_with_grad(6.0 * (1.0 - 1e-5));
        assert_close!(out, [0.0; 8], 2e-18, 0.0);
        assert_close!(d_out, [0.0; 8], 1e-13, 0.0);
    }

    #[test]
    fn origin_is_the_limit() {
        let (out, d_out) = eval_with_grad(0.0);
        assert_close!(out, eval(1e-7), 1e-12, 0.0);
        assert_eq!(d_out, [0.0; 8]);
    }

    // Optimized builds may round `sin` differently in the two code paths.
    #[test]
    fn values_match_without_grad() {
        let mut rng = XorShift64::new(1);
        for _ in 0..100 {
            let r = 6.5 * rng.next_f64();
            assert_close!(eval_with_grad(r).0, eval(r), 1e-15, 0.0);
        }
    }

    #[test]
    fn grad_matches_finite_differences() {
        let mut rng = XorShift64::new(2);
        for _ in 0..100 {
            let r = 0.01 + 5.98 * rng.next_f64();
            let (_, d_out) = eval_with_grad(r);
            for (n, &d) in d_out.iter().enumerate() {
                let fd = finite_difference(|v| eval(v[0])[n], &[r], 1e-6);
                assert_close!(fd, [d], 1e-8, 0.0);
            }
        }
    }

    #[test]
    fn batch_matches_single() {
        let dist = [0.0, 0.7, 2.5, 5.9, 6.0, 8.0];
        let mut out = vec![0.0; dist.len() * 8];
        basis().eval_batch(&dist, &mut out);
        let single: Vec<f64> = dist.iter().flat_map(|&r| eval(r)).collect();
        assert_eq!(out, single);
    }

    // The tolerances are about twice the largest errors measured.
    #[test]
    fn f32_matches_f64() {
        let mut rng = XorShift64::new(3);
        for _ in 0..100 {
            let r = 0.5 + 5.5 * rng.next_f64();
            let (mut out, mut d_out) = ([0f32; 8], [0f32; 8]);
            basis::<f32>().eval_with_grad(r as f32, &mut out, &mut d_out);
            let (out64, d_out64) = eval_with_grad(f64::from(r as f32));
            assert_close!(out, out64, 2.5e-7, 0.0);
            assert_close!(d_out, d_out64, 1e-6, 0.0);
        }
    }
}
