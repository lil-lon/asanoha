use core::cmp::Ordering;

use asanoha_core::Real;

use crate::Error;

/// Cell matrix `H` whose rows are the lattice vectors `a, b, c`; Cartesian
/// `r = f·H` for fractional `f`. Left-handed cells (`det H < 0`) are allowed.
#[derive(Debug, Clone, PartialEq)]
pub struct Cell<T: Real> {
    rows: [[T; 3]; 3],
    inverse: [[T; 3]; 3],
    volume: T,
    heights: [T; 3],
}

impl<T: Real> Cell<T> {
    /// Creates the cell from its rows `a, b, c`. Fails with
    /// [`Error::SingularCell`] unless `|det H| > 1e-12·max(|a|, |b|, |c|)³`,
    /// which also rejects non-finite entries.
    pub fn from_rows(rows: [[T; 3]; 3]) -> Result<Self, Error> {
        // Work in f64 on H/s, where s is the power of two obtained by clearing
        // the mantissa of the largest |entry|. Dividing by s is exact and keeps
        // the cubes and squared norms below in range for any finite cell.
        let max = rows
            .iter()
            .flatten()
            .fold(0.0, |m: f64, x| m.max(x.to_f64().abs()));
        let s = f64::from_bits(max.to_bits() & 0x7ff0_0000_0000_0000);
        let [a, b, c] = rows.map(|r| r.map(|x| x.to_f64() / s));
        // Column m of (H/s)⁻¹ is normals[m] / det.
        let normals = [cross(b, c), cross(c, a), cross(a, b)];
        let det = dot(a, normals[0]);
        let scale = norm(a).max(norm(b)).max(norm(c));
        let threshold = 1e-12 * scale.powi(3);
        if det.abs().partial_cmp(&threshold) != Some(Ordering::Greater) {
            return Err(Error::SingularCell);
        }
        Ok(Self {
            rows,
            inverse: core::array::from_fn(|k| {
                core::array::from_fn(|m| T::from_f64(normals[m][k] / det / s))
            }),
            volume: T::from_f64(det.abs() * s * s * s),
            heights: normals.map(|n| T::from_f64(det.abs() / norm(n) * s)),
        })
    }

    /// Cubic cell with edge `l`. Panics if `l` is zero or not finite.
    pub fn cubic(l: T) -> Self {
        let z = T::ZERO;
        Self::from_rows([[l, z, z], [z, l, z], [z, z, l]])
            .expect("cubic cell edge must be nonzero and finite")
    }

    /// The lattice vectors `a, b, c` as rows.
    pub fn rows(&self) -> &[[T; 3]; 3] {
        &self.rows
    }

    /// `|det H|`.
    pub fn volume(&self) -> T {
        self.volume
    }

    /// Distances between adjacent lattice planes:
    /// `[V/|b×c|, V/|c×a|, V/|a×b|]`.
    pub fn heights(&self) -> [T; 3] {
        self.heights
    }

    /// `f = r·H⁻¹`.
    pub fn to_fractional(&self, r: [T; 3]) -> [T; 3] {
        let h = &self.inverse;
        core::array::from_fn(|m| r[0] * h[0][m] + r[1] * h[1][m] + r[2] * h[2][m])
    }

    /// `r = f·H`.
    pub fn to_cartesian(&self, f: [T; 3]) -> [T; 3] {
        let h = &self.rows;
        core::array::from_fn(|k| f[0] * h[0][k] + f[1] * h[1][k] + f[2] * h[2][k])
    }
}

fn dot(u: [f64; 3], v: [f64; 3]) -> f64 {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
}

fn cross(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

fn norm(u: [f64; 3]) -> f64 {
    dot(u, u).sqrt()
}

#[cfg(test)]
mod tests {
    use asanoha_core::assert_close;
    use asanoha_core::testing::XorShift64;

    use super::*;

    fn cell<T: Real>(rows: [[f64; 3]; 3]) -> Result<Cell<T>, Error> {
        Cell::from_rows(rows.map(|r| r.map(T::from_f64)))
    }

    fn vec3<T: Real>(v: [f64; 3]) -> [T; 3] {
        v.map(T::from_f64)
    }

    /// `H·H⁻¹`
    fn product<T: Real>(c: &Cell<T>) -> [[f64; 3]; 3] {
        core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                (0..3)
                    .map(|k| c.rows[i][k].to_f64() * c.inverse[k][j].to_f64())
                    .sum()
            })
        })
    }

    const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn check_cubic<T: Real>(tol: f64) {
        let c = Cell::<T>::cubic(T::from_f64(2.5));
        assert_close!(c.volume().to_f64(), 15.625, 0.0, tol);
        assert_close!(c.heights(), [2.5; 3], 0.0, tol);
        assert_close!(
            c.to_fractional(vec3([1.25, -5.0, 7.5])),
            [0.5, -2.0, 3.0],
            0.0,
            tol
        );
        assert_close!(
            c.to_cartesian(vec3([0.5, -2.0, 3.0])),
            [1.25, -5.0, 7.5],
            0.0,
            tol
        );
    }

    #[test]
    fn cubic() {
        check_cubic::<f32>(1e-6);
        check_cubic::<f64>(1e-15);
    }

    fn check_extreme_edges<T: Real>(edges: [f64; 2], tol: f64) {
        // l⁴, the squared norm of a face normal, leaves the range of T for
        // these edges, while the volume l³ does not.
        for l in edges.map(T::from_f64) {
            let c = Cell::cubic(l);
            assert_close!(c.volume().to_f64(), l.to_f64().powi(3), 0.0, tol);
            assert_close!(c.heights(), [l; 3], 0.0, tol);
            assert_close!(c.to_fractional([l; 3]), [1.0; 3], 0.0, tol);
        }
    }

    #[test]
    fn extreme_edges() {
        check_extreme_edges::<f32>([1e10, 1e-12], 1e-7);
        check_extreme_edges::<f64>([1e100, 1e-100], 1e-15);
    }

    #[test]
    fn orthorhombic() {
        let c = cell::<f64>([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]]).unwrap();
        assert_close!(c.volume(), 72.0, 0.0, 1e-15);
        assert_close!(c.heights(), [3.0, 4.0, 6.0], 0.0, 1e-15);
        assert_close!(
            c.to_fractional([1.5, -1.0, 12.0]),
            [0.5, -0.25, 2.0],
            0.0,
            1e-15
        );
    }

    // a = (5, 0, 0), b = (4, 3, 0), c = (0, 0, 5): b×c = (15, −20, 0),
    // c×a = (0, 25, 0), a×b = (0, 0, 15), det = 75, so the heights are
    // 75/25, 75/25, 75/15 = 3, 3, 5 while every edge has length 5.
    const TRICLINIC: [[f64; 3]; 3] = [[5.0, 0.0, 0.0], [4.0, 3.0, 0.0], [0.0, 0.0, 5.0]];

    fn check_triclinic<T: Real>(tol: f64) {
        let c = cell::<T>(TRICLINIC).unwrap();
        assert_close!(c.volume().to_f64(), 75.0, 0.0, tol);
        assert_close!(c.heights(), [3.0, 3.0, 5.0], 0.0, tol);
        assert_close!(product(&c).concat(), IDENTITY.concat(), tol, 0.0);

        // r = 0.5·a + 0.25·b − 1.5·c
        let f = [0.5, 0.25, -1.5];
        let r = [3.5, 0.75, -7.5];
        assert_close!(c.to_cartesian(vec3(f)), r, 0.0, tol);
        assert_close!(c.to_fractional(vec3(r)), f, 0.0, tol);
    }

    #[test]
    fn triclinic() {
        check_triclinic::<f32>(1e-6);
        check_triclinic::<f64>(1e-15);
    }

    #[test]
    fn left_handed() {
        let [a, b, c] = TRICLINIC;
        let cell = cell::<f64>([b, a, c]).unwrap();
        assert_close!(cell.volume(), 75.0, 0.0, 1e-15);
        assert_close!(cell.heights(), [3.0, 3.0, 5.0], 0.0, 1e-15);
        assert_close!(product(&cell).concat(), IDENTITY.concat(), 1e-15, 0.0);
        assert_close!(
            cell.to_fractional([3.5, 0.75, -7.5]),
            [0.25, 0.5, -1.5],
            0.0,
            1e-15
        );
    }

    #[test]
    fn rejects_singular() {
        let [a, b, _] = TRICLINIC;
        let coplanar = [a, b, [a[0] + b[0], a[1] + b[1], a[2] + b[2]]];
        let flat = [a, b, [0.0, 0.0, 1e-13]];
        for rows in [
            coplanar,
            flat,
            [[0.0; 3]; 3],
            [a, b, [0.0, 0.0, f64::NAN]],
            [a, b, [0.0, 0.0, f64::INFINITY]],
        ] {
            assert_eq!(cell::<f64>(rows), Err(Error::SingularCell), "{rows:?}");
            assert_eq!(cell::<f32>(rows), Err(Error::SingularCell), "{rows:?}");
        }
    }

    #[test]
    fn accepts_thin_cell() {
        let [a, b, _] = TRICLINIC;
        let c = cell::<f64>([a, b, [0.0, 0.0, 1e-9]]).unwrap();
        assert_close!(c.heights(), [3.0, 3.0, 1e-9], 0.0, 1e-15);
    }

    #[test]
    fn random_round_trip() {
        let mut rng = XorShift64::new(5);
        for _ in 0..1000 {
            let rows: [[f64; 3]; 3] = core::array::from_fn(|i| {
                core::array::from_fn(|k| rng.normal() + if i == k { 3.0 } else { 0.0 })
            });
            let c = cell::<f64>(rows).unwrap();
            assert_close!(product(&c).concat(), IDENTITY.concat(), 1e-13, 0.0);
            let f = [rng.normal(), rng.normal(), rng.normal()];
            assert_close!(c.to_fractional(c.to_cartesian(f)), f, 1e-12, 0.0);
            // The height along axis m is the projection of row m on the unit
            // normal of the other two rows.
            for m in 0..3 {
                let n = cross(rows[(m + 1) % 3], rows[(m + 2) % 3]);
                assert_close!(c.heights()[m], dot(rows[m], n).abs() / norm(n), 0.0, 1e-13);
            }
        }
    }
}
