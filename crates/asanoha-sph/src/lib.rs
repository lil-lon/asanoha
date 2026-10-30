//! Real spherical harmonics `l ≤ 3` and their gradients, as
//! `e3nn_jax.spherical_harmonics("0e + 1o + 2e + 3o", r, normalize=True,
//! normalization="component")` computes them.
//!
//! The 16 outputs are `l = 0, 1, 2, 3` in turn, `2l + 1` components each, in
//! e3nn-jax's order. They depend only on the direction `u = r / |r|`, and
//! `Σ_m Y_lm(u)² = 2l + 1`.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

use asanoha_core::Real;

/// Number of spherical harmonics with `l ≤ 3`.
pub const DIM_LMAX3: usize = 16;

/// Degree `l` of each output component.
const DEGREE: [usize; DIM_LMAX3] = [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3];

/// `√x` rounded to `T`.
fn sqrt<T: Real>(x: f64) -> T {
    T::from_f64(x.sqrt())
}

/// The homogeneous polynomials, of degree `l`, that e3nn-jax evaluates at
/// the unit vector.
fn poly<T: Real>([x, y, z]: [T; 3]) -> [T; DIM_LMAX3] {
    let three = T::from_f64(3.0);
    let four = T::from_f64(4.0);
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let c1 = sqrt::<T>(3.0);
    let c2 = sqrt::<T>(15.0);
    let c3 = sqrt::<T>(70.0) / four;
    let c4 = sqrt::<T>(42.0) / four;
    let c5 = sqrt::<T>(105.0);
    [
        T::ONE,
        c1 * x,
        c1 * y,
        c1 * z,
        c2 * x * z,
        c2 * x * y,
        sqrt::<T>(5.0) * (yy - T::HALF * (xx + zz)),
        c2 * y * z,
        c2 * T::HALF * (zz - xx),
        c3 * x * (three * zz - xx),
        c5 * x * y * z,
        c4 * x * (four * yy - xx - zz),
        sqrt::<T>(7.0) * T::HALF * y * (T::TWO * yy - three * (xx + zz)),
        c4 * z * (four * yy - xx - zz),
        c5 * T::HALF * y * (zz - xx),
        c3 * z * (zz - three * xx),
    ]
}

/// `∂P/∂u_k` of each polynomial of [`poly`].
fn poly_grad<T: Real>([x, y, z]: [T; 3]) -> [[T; 3]; DIM_LMAX3] {
    let zero = T::ZERO;
    let three = T::from_f64(3.0);
    let four = T::from_f64(4.0);
    let six = T::from_f64(6.0);
    let eight = T::from_f64(8.0);
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let c1 = sqrt::<T>(3.0);
    let c2 = sqrt::<T>(15.0);
    let c3 = sqrt::<T>(70.0) / four;
    let c4 = sqrt::<T>(42.0) / four;
    let c5 = sqrt::<T>(105.0);
    let c6 = sqrt::<T>(5.0);
    let c7 = sqrt::<T>(7.0) * T::HALF;
    let c8 = c5 * T::HALF;
    [
        [zero, zero, zero],
        [c1, zero, zero],
        [zero, c1, zero],
        [zero, zero, c1],
        [c2 * z, zero, c2 * x],
        [c2 * y, c2 * x, zero],
        [-c6 * x, T::TWO * c6 * y, -c6 * z],
        [zero, c2 * z, c2 * y],
        [-c2 * x, zero, c2 * z],
        [three * c3 * (zz - xx), zero, six * c3 * x * z],
        [c5 * y * z, c5 * x * z, c5 * x * y],
        [
            c4 * (four * yy - three * xx - zz),
            eight * c4 * x * y,
            -T::TWO * c4 * x * z,
        ],
        [
            -six * c7 * x * y,
            three * c7 * (T::TWO * yy - xx - zz),
            -six * c7 * y * z,
        ],
        [
            -T::TWO * c4 * x * z,
            eight * c4 * y * z,
            c4 * (four * yy - xx - three * zz),
        ],
        [-T::TWO * c8 * x * y, c8 * (zz - xx), T::TWO * c8 * y * z],
        [-six * c3 * x * z, zero, three * c3 * (zz - xx)],
    ]
}

/// `r / |r|` and `|r|`.
fn normalize<T: Real>(r: [T; 3]) -> ([T; 3], T) {
    let norm = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    (r.map(|c| c / norm), norm)
}

/// The 16 spherical harmonics of the direction of `r`, which must be
/// nonzero.
pub fn sh_lmax3<T: Real>(r: [T; 3]) -> [T; DIM_LMAX3] {
    poly(normalize(r).0)
}

/// The spherical harmonics of [`sh_lmax3`] and their gradients
/// `grad[c][k] = ∂Y_c/∂r_k`, which include the derivative of `r / |r|`.
///
/// With `P` of degree `l`, `∂Y/∂r_k = (∂P/∂u_k − l·u_k·P(u)) / |r|`: the
/// derivative of `r / |r|` projects `∇P` onto the tangent plane, and
/// `u·∇P = l·P` for a homogeneous polynomial.
pub fn sh_lmax3_with_grad<T: Real>(r: [T; 3]) -> ([T; DIM_LMAX3], [[T; 3]; DIM_LMAX3]) {
    let (u, norm) = normalize(r);
    let y = poly(u);
    let mut grad = poly_grad(u);
    for (c, g) in grad.iter_mut().enumerate() {
        let ly = T::from_usize(DEGREE[c]) * y[c];
        for k in 0..3 {
            g[k] = (g[k] - ly * u[k]) / norm;
        }
    }
    (y, grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use asanoha_core::assert_close;
    use asanoha_core::testing::{XorShift64, finite_difference};

    /// Random vectors with lengths in `[0.5, 6)`.
    fn vectors(seed: u64, n: usize) -> Vec<[f64; 3]> {
        let mut rng = XorShift64::new(seed);
        (0..n)
            .map(|_| {
                let v = [rng.normal(), rng.normal(), rng.normal()];
                let len = 0.5 + 5.5 * rng.next_f64();
                let (u, _) = normalize(v);
                u.map(|c| c * len)
            })
            .collect()
    }

    #[test]
    fn component_normalization() {
        for r in vectors(1, 100) {
            let y = sh_lmax3(r);
            let mut sums = [0.0; 4];
            for (c, v) in y.iter().enumerate() {
                sums[DEGREE[c]] += v * v;
            }
            assert_close!(sums, [1.0, 3.0, 5.0, 7.0], 1e-14, 0.0);
        }
    }

    #[test]
    fn depends_only_on_direction() {
        for r in vectors(2, 100) {
            assert_close!(sh_lmax3(r.map(|c| c * 3.7)), sh_lmax3(r), 1e-14, 0.0);
        }
    }

    #[test]
    fn values_match_without_grad() {
        for r in vectors(3, 100) {
            assert_eq!(sh_lmax3_with_grad(r).0, sh_lmax3(r));
        }
    }

    #[test]
    fn grad_matches_finite_differences() {
        for r in vectors(4, 100) {
            let (_, grad) = sh_lmax3_with_grad(r);
            for (c, g) in grad.iter().enumerate() {
                let f = |v: &[f64]| sh_lmax3([v[0], v[1], v[2]])[c];
                assert_close!(finite_difference(f, &r, 1e-6), g, 2e-9, 0.0);
            }
        }
    }
}
