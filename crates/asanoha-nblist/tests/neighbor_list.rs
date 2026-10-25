//! Neighbor-list tests shared by every algorithm. The geometries of the
//! hand-derived cases follow nblist-rs; the expected edges are re-derived for
//! the strict cutoff `|r|² < r_c²` and for shifts relative to the input
//! positions.

use asanoha_core::testing::XorShift64;
use asanoha_core::{Real, assert_close};
use asanoha_nblist::{Algorithm, Cell, Error, NeighborList, Structure};

const ALGORITHMS: [Algorithm; 2] = [Algorithm::Naive, Algorithm::CellList];

const ALL: [bool; 3] = [true; 3];
const SLAB: [bool; 3] = [true, true, false];
const WIRE: [bool; 3] = [true, false, false];
const NONE: [bool; 3] = [false; 3];

/// `(receiver, sender, shift, distance)`
type Edge = (u32, u32, [i32; 3], f64);

fn cell(rows: [[f64; 3]; 3]) -> Cell<f64> {
    Cell::from_rows(rows).unwrap()
}

fn build<T: Real>(
    positions: &[[T; 3]],
    cell: Option<&Cell<T>>,
    pbc: [bool; 3],
    cutoff: T,
    algo: Algorithm,
) -> Result<NeighborList<T>, Error> {
    let s = Structure {
        positions,
        cell,
        pbc,
    };
    NeighborList::build(&s, cutoff, algo)
}

/// `(i, j, S)` of every edge in output order.
fn triples<T: Real>(nl: &NeighborList<T>) -> Vec<(u32, u32, [i32; 3])> {
    (0..nl.n_atoms())
        .flat_map(|i| {
            nl.edges_of(i)
                .map(move |e| (i as u32, nl.senders[e], nl.shifts[e]))
        })
        .collect()
}

/// Each pair in both directions: `(i, j, S, d)` and `(j, i, −S, d)`.
fn both(pairs: &[Edge]) -> Vec<Edge> {
    pairs
        .iter()
        .flat_map(|&(i, j, s, d)| [(i, j, s, d), (j, i, s.map(|x| -x), d)])
        .collect()
}

/// Checks that every edge vector is `(r_j − r_i) + S·H` for the input
/// positions, evaluated in this order, and that every distance is the norm of
/// its vector.
fn check_vectors(nl: &NeighborList<f64>, positions: &[[f64; 3]], cell: Option<&Cell<f64>>) {
    let h = cell.map_or([[0.0; 3]; 3], |c| *c.rows());
    for i in 0..nl.n_atoms() {
        for e in nl.edges_of(i) {
            let s = nl.shifts[e].map(f64::from);
            let (ri, rj) = (positions[i], positions[nl.senders[e] as usize]);
            let expected: [f64; 3] = core::array::from_fn(|k| {
                (rj[k] - ri[k]) + (s[0] * h[0][k] + s[1] * h[1][k] + s[2] * h[2][k])
            });
            assert_eq!(nl.vectors[e], expected);
            let v = nl.vectors[e];
            assert_eq!(
                nl.distances[e],
                (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
            );
        }
    }
}

/// Builds the list with every algorithm and checks it against `expected`,
/// given in any order: the same `(i, j, S)` in canonical order, distances
/// within a few ulp of the coordinates, and consistent vectors.
fn check(
    positions: &[[f64; 3]],
    cell: Option<&Cell<f64>>,
    pbc: [bool; 3],
    cutoff: f64,
    expected: &[Edge],
) {
    let mut expected = expected.to_vec();
    expected.sort_by_key(|e| (e.0, e.1, e.2));
    let triples_expected: Vec<_> = expected.iter().map(|e| (e.0, e.1, e.2)).collect();
    let distances: Vec<f64> = expected.iter().map(|e| e.3).collect();
    for algo in ALGORITHMS {
        let nl = build(positions, cell, pbc, cutoff, algo).unwrap();
        assert_eq!(nl.n_atoms(), positions.len());
        assert_eq!(triples(&nl), triples_expected, "{algo:?}");
        assert_close!(nl.distances, distances, 4e-15, 0.0);
        check_vectors(&nl, positions, cell);
    }
}

#[test]
fn basic() {
    let pos = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [5.0, 5.0, 5.0],
        [8.0, 0.0, 0.0],
    ];
    let expected = both(&[(0, 1, [0, 0, 0], 2.0), (0, 3, [-1, 0, 0], 2.0)]);
    check(&pos, Some(&Cell::cubic(10.0)), ALL, 3.0, &expected);
}

#[test]
fn no_atoms() {
    check(&[], Some(&Cell::cubic(10.0)), ALL, 3.0, &[]);
    check(&[], None, NONE, 3.0, &[]);
}

#[test]
fn extent_beyond_f64_range() {
    // The extent along x, 2e308, overflows f64.
    let pos = [
        [-1e308, 0.0, 0.0],
        [1e308, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ];
    let expected = both(&[(2, 3, [0, 0, 0], 1.0)]);
    check(&pos, None, NONE, 2.0, &expected);
}

#[test]
fn no_neighbors() {
    let pos = [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]];
    check(&pos, Some(&Cell::cubic(20.0)), ALL, 1.0, &[]);
}

#[test]
fn pbc_corner() {
    let pos = [[0.5, 0.5, 0.5], [9.5, 9.5, 9.5]];
    let expected = both(&[(0, 1, [-1, -1, -1], 3f64.sqrt())]);
    check(&pos, Some(&Cell::cubic(10.0)), ALL, 2.0, &expected);
}

#[test]
fn self_image() {
    // Face images at 3; edge-diagonal images at √18 are outside.
    let expected = both(&[
        (0, 0, [1, 0, 0], 3.0),
        (0, 0, [0, 1, 0], 3.0),
        (0, 0, [0, 0, 1], 3.0),
    ]);
    check(&[[0.0; 3]], Some(&Cell::cubic(3.0)), ALL, 3.5, &expected);
}

#[test]
fn multiple_images() {
    // 0→1 with S = (s, t, u) has |r|² = (1 + 4s)² + 16(t² + u²) < 25; the
    // images at exactly 5 (S = (−1, ±1, 0), (1, 0, 0), …) are outside.
    let expected = both(&[
        (0, 1, [0, 0, 0], 1.0),
        (0, 1, [0, 1, 0], 17f64.sqrt()),
        (0, 1, [0, -1, 0], 17f64.sqrt()),
        (0, 1, [0, 0, 1], 17f64.sqrt()),
        (0, 1, [0, 0, -1], 17f64.sqrt()),
        (0, 1, [-1, 0, 0], 3.0),
        (0, 0, [1, 0, 0], 4.0),
        (0, 0, [0, 1, 0], 4.0),
        (0, 0, [0, 0, 1], 4.0),
        (1, 1, [1, 0, 0], 4.0),
        (1, 1, [0, 1, 0], 4.0),
        (1, 1, [0, 0, 1], 4.0),
    ]);
    let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    check(&pos, Some(&Cell::cubic(4.0)), ALL, 5.0, &expected);
}

#[test]
fn orthorhombic() {
    let c = cell([[3.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
    let expected = both(&[
        (0, 1, [0, 0, 0], 1.0),
        (0, 1, [-1, 0, 0], 2.0),
        (0, 0, [1, 0, 0], 3.0),
        (1, 1, [1, 0, 0], 3.0),
    ]);
    let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    check(&pos, Some(&c), ALL, 3.5, &expected);
}

#[test]
fn unwrapped_coordinates() {
    // j wraps by (2, −2, 4); the nearest image of the wrapped j is at
    // S = (−1, 0, 0), so relative to the input S = (−3, 2, −4) and
    // r = (−2, 1, 1).
    let pos = [[1.0, 1.0, 1.0], [14.0, -8.0, 22.0]];
    let expected = both(&[(0, 1, [-3, 2, -4], 6f64.sqrt())]);
    check(&pos, Some(&Cell::cubic(5.0)), ALL, 3.0, &expected);
}

const MONOCLINIC: [[f64; 3]; 3] = [[5.0, 0.0, 0.0], [2.0, 5.0, 0.0], [0.0, 0.0, 5.0]];

#[test]
fn monoclinic_basic() {
    let pos = [[1.0, 1.0, 0.0], [6.0, 4.0, 0.0]];
    let expected = both(&[(0, 1, [-1, -1, 0], 8f64.sqrt())]);
    check(&pos, Some(&cell(MONOCLINIC)), ALL, 2.9, &expected);
}

#[test]
fn monoclinic_pbc_corner() {
    let c = cell(MONOCLINIC);
    let pos = [
        c.to_cartesian([0.05, 0.05, 0.05]),
        c.to_cartesian([0.95, 0.95, 0.95]),
    ];
    let expected = both(&[(0, 1, [-1, -1, -1], 0.99f64.sqrt())]);
    check(&pos, Some(&c), ALL, 1.5, &expected);
}

#[test]
fn monoclinic_unwrapped() {
    // B is at fractional (2.32, 1.2, 0.2) and wraps by (2, 1, 0) to
    // (2, 1, 1), at distance 1 from A without a further shift.
    let pos = [[1.0, 1.0, 1.0], [14.0, 6.0, 1.0]];
    let expected = both(&[(0, 1, [-2, -1, 0], 1.0)]);
    check(&pos, Some(&cell(MONOCLINIC)), ALL, 1.5, &expected);
}

#[test]
fn monoclinic_self_image() {
    // ±a, ±c at 5 and ±b at √29; a − b at √34 is outside.
    let expected = both(&[
        (0, 0, [1, 0, 0], 5.0),
        (0, 0, [0, 1, 0], 29f64.sqrt()),
        (0, 0, [0, 0, 1], 5.0),
    ]);
    check(&[[0.0; 3]], Some(&cell(MONOCLINIC)), ALL, 5.5, &expected);
}

#[test]
fn monoclinic_skewed() {
    // The height along b is 1 while |b| ≈ 9.55, so S_b up to ±3 is needed.
    let c = cell([[10.0, 0.0, 0.0], [9.5, 1.0, 0.0], [0.0, 0.0, 5.0]]);
    let expected = both(&[
        (0, 0, [1, -1, 0], 1.25f64.sqrt()),
        (0, 0, [2, -2, 0], 5f64.sqrt()),
    ]);
    check(&[[0.0; 3]], Some(&c), ALL, 2.5, &expected);
}

#[test]
fn monoclinic_boundary_distance() {
    // The image of B at S = (−1, 0, 0) is at exactly the cutoff, 3.
    let pos = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let expected = both(&[(0, 1, [0, 0, 0], 2.0)]);
    check(&pos, Some(&cell(MONOCLINIC)), ALL, 3.0, &expected);
}

#[test]
fn monoclinic_left_handed() {
    let c = cell([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0], [2.0, 5.0, 0.0]]);
    let pos = [
        c.to_cartesian([0.05, 0.05, 0.05]),
        c.to_cartesian([0.95, 0.95, 0.95]),
    ];
    let expected = both(&[(0, 1, [-1, -1, -1], 0.99f64.sqrt())]);
    check(&pos, Some(&c), ALL, 1.5, &expected);
}

#[test]
fn triclinic_full_3d() {
    let c = cell([[6.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.5, 1.0, 4.0]]);
    let pos = [
        c.to_cartesian([0.1, 0.1, 0.1]),
        c.to_cartesian([0.9, 0.9, 0.9]),
    ];
    // r = (−1.5, −1.2, −0.8) and (−1, −0.2, 3.2).
    let expected = both(&[
        (0, 1, [-1, -1, -1], 4.33f64.sqrt()),
        (0, 1, [-1, -1, 0], 11.28f64.sqrt()),
    ]);
    check(&pos, Some(&c), ALL, 3.5, &expected);
}

#[test]
fn triclinic_full_3d_self_image() {
    let c = cell([[4.0, 0.0, 0.0], [1.0, 4.0, 0.0], [1.0, 1.0, 4.0]]);
    let expected = both(&[
        (0, 0, [1, 0, 0], 4.0),
        (0, 0, [0, 1, 0], 17f64.sqrt()),
        (0, 0, [0, 0, 1], 18f64.sqrt()),
    ]);
    check(&[[0.0; 3]], Some(&c), ALL, 4.3, &expected);
}

#[test]
fn triclinic_multi_height_replicas() {
    // Heights ≈ (0.74, 1, 1): more than one image along every axis.
    let c = cell([[10.0, 0.0, 0.0], [9.5, 1.0, 0.0], [9.5, 0.0, 1.0]]);
    let expected = both(&[
        (0, 0, [1, -1, 0], 1.25f64.sqrt()),
        (0, 0, [1, 0, -1], 1.25f64.sqrt()),
        (0, 0, [0, 1, -1], 2f64.sqrt()),
    ]);
    check(&[[0.0; 3]], Some(&c), ALL, 1.5, &expected);
}

#[test]
fn triclinic_multi_atom() {
    // The pairs 1→2 at S = (0, 0, 0) and (1, −1, 0) are at exactly the
    // cutoff, 2.5.
    let c = cell([[4.0, 0.0, 0.0], [1.0, 4.0, 0.0], [1.0, 1.0, 4.0]]);
    let pos = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 2.0, 0.0]];
    let expected = both(&[
        (0, 1, [0, 0, 0], 2.0),
        (0, 1, [-1, 0, 0], 2.0),
        (0, 2, [0, 0, 0], 4.25f64.sqrt()),
        (0, 2, [0, -1, 0], 4.25f64.sqrt()),
    ]);
    check(&pos, Some(&c), ALL, 2.5, &expected);
}

#[test]
fn triclinic_near_degenerate() {
    // Heights ≈ (0.36, 0.5, 0.5): images up to (4, 3, 3) are searched.
    let c = cell([[5.0, 0.0, 0.0], [4.9, 0.5, 0.0], [4.9, 0.0, 0.5]]);
    let expected = both(&[
        (0, 0, [1, -1, 0], 0.26f64.sqrt()),
        (0, 0, [1, 0, -1], 0.26f64.sqrt()),
        (0, 0, [0, 1, -1], 0.5f64.sqrt()),
        (0, 0, [2, -1, -1], 0.54f64.sqrt()),
        (0, 0, [2, -2, 0], 1.04f64.sqrt()),
        (0, 0, [2, 0, -2], 1.04f64.sqrt()),
    ]);
    check(&[[0.0; 3]], Some(&c), ALL, 1.1, &expected);
}

#[test]
fn slab_self_image() {
    let expected = both(&[(0, 0, [1, 0, 0], 3.0), (0, 0, [0, 1, 0], 3.0)]);
    check(&[[0.0; 3]], Some(&Cell::cubic(3.0)), SLAB, 3.5, &expected);
}

#[test]
fn wire_self_image() {
    let expected = both(&[(0, 0, [1, 0, 0], 3.0)]);
    check(&[[0.0; 3]], Some(&Cell::cubic(3.0)), WIRE, 3.5, &expected);
}

#[test]
fn isolated_no_images() {
    check(&[[0.0; 3]], Some(&Cell::cubic(3.0)), NONE, 3.5, &[]);
    check(&[[0.0; 3]], None, NONE, 3.5, &[]);
}

#[test]
fn slab_two_atoms() {
    // A and C are 9 apart along the non-periodic c axis.
    let pos = [[0.5, 0.5, 0.5], [9.5, 0.5, 0.5], [0.5, 0.5, 9.5]];
    let expected = both(&[(0, 1, [-1, 0, 0], 1.0)]);
    check(&pos, Some(&Cell::cubic(10.0)), SLAB, 2.0, &expected);
}

#[test]
fn isolated_two_atoms() {
    let pos = [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [9.0, 1.0, 1.0]];
    let expected = both(&[(0, 1, [0, 0, 0], 2.0)]);
    check(&pos, Some(&Cell::cubic(10.0)), NONE, 3.0, &expected);
    check(&pos, None, NONE, 3.0, &expected);
}

#[test]
fn slab_no_wrap_non_periodic() {
    // Atom 0 is at fractional c = 1.5 and is not wrapped along c.
    let c = Cell::cubic(5.0);
    let pos = [c.to_cartesian([0.1, 0.1, 1.5]), [0.5, 0.5, 7.0]];
    let expected = both(&[(0, 1, [0, 0, 0], 0.5)]);
    check(&pos, Some(&c), SLAB, 2.0, &expected);
}

#[test]
fn slab_cross_bin_boundary() {
    let pos = [[5.0, 5.0, 3.3], [5.0, 5.0, 3.4]];
    let expected = both(&[(0, 1, [0, 0, 0], 0.1)]);
    check(&pos, Some(&Cell::cubic(10.0)), SLAB, 3.0, &expected);
}

#[test]
fn isolated_atoms_outside_cell() {
    let pos = [[-5.0, 5.0, 5.0], [-3.0, 5.0, 5.0]];
    let expected = both(&[(0, 1, [0, 0, 0], 2.0)]);
    check(&pos, Some(&Cell::cubic(10.0)), NONE, 2.5, &expected);
    check(&pos, None, NONE, 2.5, &expected);
}

#[test]
fn strict_boundary() {
    let below = 2.0 * (1.0 - 1e-12);
    let pos = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, below, 0.0]];
    let expected = both(&[(0, 2, [0, 0, 0], below)]);
    check(&pos, None, NONE, 2.0, &expected);
    // Self-images at exactly the cutoff.
    check(&[[0.0; 3]], Some(&Cell::cubic(3.0)), ALL, 3.0, &[]);
    // Both self-images of an atom off the origin round to the same distance.
    check(&[[0.4, 0.0, 0.0]], Some(&Cell::cubic(1.0)), WIRE, 1.0, &[]);
}

#[test]
fn small_cubic_image_count() {
    // One atom in a cubic cell of edge 2 with cutoff 6: one edge per lattice
    // vector n with 0 < 2|n| < 6.
    let count = (-5i32..=5)
        .flat_map(|x| {
            (-5i32..=5).flat_map(move |y| (-5i32..=5).map(move |z| x * x + y * y + z * z))
        })
        .filter(|&n2| 0 < n2 && n2 < 9)
        .count();
    for algo in ALGORITHMS {
        let nl = build(&[[0.3, -0.2, 0.7]], Some(&Cell::cubic(2.0)), ALL, 6.0, algo).unwrap();
        assert_eq!(nl.n_edges(), count, "{algo:?}");
    }
}

#[test]
fn f32_large_cell_keeps_small_separations() {
    // Both pairs are 1 apart through the cell boundary, while 1e8 ± 1 is not
    // representable in f32.
    let c = Cell::<f32>::cubic(1e8);
    for pos in [
        [[0.0; 3], [-1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [1e8, 0.0, 0.0]],
    ] {
        for algo in ALGORITHMS {
            let nl = build(&pos, Some(&c), ALL, 1e4, algo).unwrap();
            assert_eq!(nl.distances, [1.0, 1.0], "{algo:?}");
        }
    }
}

fn check_skewed_cell_far_atom<T: Real>() {
    // det = 4e-6 is just above 1e-6·|b|³ ≈ 2.8e-6, and the cutoff 1.5e-4 is
    // just above 1e-4·|b|. Atom 0 sits on the image of atom 1 at
    // S = (2¹⁶, 0, 0), the farthest accepted wrap. a − b = (0, −4e-6, 0), so
    // the images at S + k·(1, −1, 0) with |k| ≤ 37 are within the cutoff, for
    // both pairs and for the self-images.
    let rows = [[1.0, 1.0, 0.0], [1.0, 1.0 + 4e-6, 0.0], [0.0, 0.0, 1.0]];
    let c = Cell::from_rows(rows.map(|r| r.map(T::from_f64))).unwrap();
    let pos = [[65536.0, 65536.0, 0.0], [0.0; 3]].map(|r| r.map(T::from_f64));
    let mut expected = Vec::new();
    for (i, j, x) in [(0, 0, 0), (0, 1, 65536), (1, 0, -65536), (1, 1, 0)] {
        for k in -37..=37 {
            if x != 0 || k != 0 {
                expected.push((i, j, [x + k, -k, 0]));
            }
        }
    }
    for algo in ALGORITHMS {
        let nl = build(&pos, Some(&c), ALL, T::from_f64(1.5e-4), algo).unwrap();
        assert_eq!(triples(&nl), expected, "{algo:?}");
    }
}

#[test]
fn skewed_cell_far_atom() {
    check_skewed_cell_far_atom::<f32>();
    check_skewed_cell_far_atom::<f64>();
    // A cell skewed beyond 1e-6·L³ is rejected.
    let rows = [[1e5, 1e5, 0.0], [1e5, 1e5 + 1e-6, 0.0], [0.0, 0.0, 1e5]];
    assert_eq!(Cell::<f64>::from_rows(rows), Err(Error::InvalidCell));
    // So is an atom far along the non-periodic axis of a skewed cell, where
    // r·H⁻¹ would be too inaccurate to place it along the periodic one.
    let c = cell([[1.0, 1.0, 0.0], [1.0, 1.000004, 0.0], [0.0, 0.0, 1.0]]);
    let r = [1e14, 1e14 * 1.000004, 0.0];
    let pos = [r, [r[0] + 2.0, r[1] + 2.0, 0.0]];
    for algo in ALGORITHMS {
        let nl = build(&pos, Some(&c), WIRE, 1.5e-4, algo);
        assert_eq!(nl, Err(Error::InvalidPosition(0)), "{algo:?}");
    }
}

#[test]
fn rejects_cutoff_below_cell_resolution() {
    // A shift of about ±65535 in this cell sums terms of about 4e12, whose
    // rounding (about 4e-4) would exceed a cutoff of 1e-8.
    let c = cell([
        [63251177.551587075, 0.0, 0.0],
        [63250948.51293027, 200.0, 0.0],
        [0.0, 0.0, 63251177.551587075],
    ]);
    let pos = [[0.0; 3], [15010048.373940848, -13107000.0, 0.0]];
    for algo in ALGORITHMS {
        assert_eq!(
            build(&pos, Some(&c), ALL, 1e-8, algo),
            Err(Error::InvalidCutoff)
        );
        // 1e-4·L is accepted; below it is not.
        let c10 = Cell::cubic(10.0);
        assert!(build(&pos[..1], Some(&c10), ALL, 1e-3, algo).is_ok());
        assert_eq!(
            build(&pos[..1], Some(&c10), ALL, 0.99e-3, algo),
            Err(Error::InvalidCutoff)
        );
    }
}

/// Random triclinic cell with edges 3–6 and skew up to ±1, 1–12 atoms at
/// fractional coordinates in [−1.5, 2.5), random periodicity, and a cutoff
/// from 0.3 to 1.5 times the cell heights.
fn random_structure(rng: &mut XorShift64) -> (Vec<[f64; 3]>, Cell<f64>, [bool; 3], f64) {
    let c = random_cell(rng);
    let n = 1 + (rng.next_u64() % 12) as usize;
    let pos = (0..n)
        .map(|_| c.to_cartesian(core::array::from_fn(|_| 4.0 * rng.next_f64() - 1.5)))
        .collect();
    let pbc = core::array::from_fn(|_| rng.next_u64().is_multiple_of(2));
    let h = c.heights();
    let (lo, hi) = (
        0.3 * h[0].min(h[1]).min(h[2]),
        1.5 * h[0].max(h[1]).max(h[2]),
    );
    let cutoff = lo + (hi - lo) * rng.next_f64();
    (pos, c, pbc, cutoff)
}

/// Random triclinic cell with edges 3–6 and skew up to ±1.
fn random_cell(rng: &mut XorShift64) -> Cell<f64> {
    cell(core::array::from_fn(|i| {
        core::array::from_fn(|k| {
            if i == k {
                3.0 + 3.0 * rng.next_f64()
            } else {
                2.0 * rng.next_f64() - 1.0
            }
        })
    }))
}

#[test]
fn cell_list_matches_naive() {
    // Cutoffs log-uniform from 0.03 to 2 times the smallest cell height, so
    // that the bins range from many per axis, most of them empty, to a single
    // bin with several images. Up to 200 atoms below half the height,
    // 12 above, in a box of 0.1 to 4 cells along each axis.
    let mut rng = XorShift64::new(37);
    for _ in 0..100 {
        let c = random_cell(&mut rng);
        let h = c.heights();
        let h_min = h[0].min(h[1]).min(h[2]);
        let cutoff = 0.03 * h_min * (2.0f64 / 0.03).powf(rng.next_f64());
        let max_atoms = if cutoff < 0.5 * h_min { 200 } else { 12 };
        let n = 1 + (rng.next_u64() % max_atoms) as usize;
        let lo: [f64; 3] = core::array::from_fn(|_| 3.0 * rng.next_f64() - 2.0);
        let width: [f64; 3] = core::array::from_fn(|_| 0.1 + 3.9 * rng.next_f64());
        let pos: Vec<[f64; 3]> = (0..n)
            .map(|_| c.to_cartesian(core::array::from_fn(|a| lo[a] + width[a] * rng.next_f64())))
            .collect();
        let pbc = core::array::from_fn(|_| rng.next_u64().is_multiple_of(2));
        for (cell, pbc) in [(Some(&c), pbc), (None, NONE)] {
            let naive = build(&pos, cell, pbc, cutoff, Algorithm::Naive).unwrap();
            let cell_list = build(&pos, cell, pbc, cutoff, Algorithm::CellList).unwrap();
            assert_eq!(cell_list, naive);
        }
    }
}

#[test]
fn csr_invariants() {
    let mut rng = XorShift64::new(17);
    for _ in 0..50 {
        let (pos, c, pbc, cutoff) = random_structure(&mut rng);
        for algo in ALGORITHMS {
            let nl = build(&pos, Some(&c), pbc, cutoff, algo).unwrap();
            assert_eq!(nl.row_offsets.len(), pos.len() + 1);
            assert_eq!(nl.row_offsets[0], 0);
            assert_eq!(*nl.row_offsets.last().unwrap(), nl.n_edges());
            assert!(nl.row_offsets.windows(2).all(|w| w[0] <= w[1]));
            let t = triples(&nl);
            assert!(
                t.windows(2).all(|w| w[0] < w[1]),
                "rows not sorted by (j, S)"
            );
            for (e, &(i, j, s)) in t.iter().enumerate() {
                let r = t
                    .binary_search(&(j, i, s.map(|x| -x)))
                    .expect("missing reverse edge");
                assert_eq!(nl.vectors[r], nl.vectors[e].map(|x| -x));
                assert_eq!(nl.distances[r], nl.distances[e]);
                assert!(nl.distances[e] < cutoff);
            }
            check_vectors(&nl, &pos, Some(&c));
        }
    }
}

#[test]
fn lattice_translation() {
    // Moving atom i by n_i·H changes only the shifts, by n_i − n_j.
    let mut rng = XorShift64::new(23);
    for _ in 0..50 {
        let (pos, c, pbc, cutoff) = random_structure(&mut rng);
        let moves: Vec<[i32; 3]> = pos
            .iter()
            .map(|_| {
                core::array::from_fn(|a| {
                    if pbc[a] {
                        (rng.next_u64() % 7) as i32 - 3
                    } else {
                        0
                    }
                })
            })
            .collect();
        let moved: Vec<[f64; 3]> = pos
            .iter()
            .zip(&moves)
            .map(|(r, n)| {
                let t = c.to_cartesian(n.map(f64::from));
                core::array::from_fn(|k| r[k] + t[k])
            })
            .collect();
        for algo in ALGORITHMS {
            let a = build(&pos, Some(&c), pbc, cutoff, algo).unwrap();
            let b = build(&moved, Some(&c), pbc, cutoff, algo).unwrap();
            let shifted: Vec<_> = triples(&a)
                .into_iter()
                .map(|(i, j, s)| {
                    let (ni, nj) = (moves[i as usize], moves[j as usize]);
                    (i, j, core::array::from_fn(|k| s[k] + ni[k] - nj[k]))
                })
                .collect();
            assert_eq!(triples(&b), shifted);
            assert_close!(b.distances, a.distances, 1e-14, 0.0);
        }
    }
}

#[test]
fn no_cell_matches_non_periodic_cell() {
    let mut rng = XorShift64::new(29);
    for _ in 0..50 {
        let (pos, c, _, cutoff) = random_structure(&mut rng);
        for algo in ALGORITHMS {
            let with_cell = build(&pos, Some(&c), NONE, cutoff, algo).unwrap();
            let without = build(&pos, None, NONE, cutoff, algo).unwrap();
            assert_eq!(with_cell, without);
        }
    }
}

#[test]
fn f32_matches_f64() {
    // For f32 inputs, the f32 list is the f64 list of the same values with
    // vectors and distances rounded to f32.
    let mut rng = XorShift64::new(31);
    for _ in 0..50 {
        let (pos, c, pbc, cutoff) = random_structure(&mut rng);
        let pos32: Vec<[f32; 3]> = pos.iter().map(|r| r.map(|x| x as f32)).collect();
        let c32 = Cell::from_rows(c.rows().map(|r| r.map(|x| x as f32))).unwrap();
        let pos64: Vec<[f64; 3]> = pos32.iter().map(|r| r.map(f64::from)).collect();
        let c64 = Cell::from_rows(c32.rows().map(|r| r.map(f64::from))).unwrap();
        let cutoff = cutoff as f32;
        for algo in ALGORITHMS {
            let a = build(&pos64, Some(&c64), pbc, f64::from(cutoff), algo).unwrap();
            let b = build(&pos32, Some(&c32), pbc, cutoff, algo).unwrap();
            assert_eq!(a.row_offsets, b.row_offsets);
            assert_eq!(a.senders, b.senders);
            assert_eq!(a.shifts, b.shifts);
            let vectors: Vec<[f32; 3]> = a.vectors.iter().map(|v| v.map(|x| x as f32)).collect();
            assert_eq!(b.vectors, vectors);
            let distances: Vec<f32> = a.distances.iter().map(|&d| d as f32).collect();
            assert_eq!(b.distances, distances);
        }
    }
}

#[test]
fn receivers_and_sender_permutation() {
    let mut rng = XorShift64::new(43);
    for _ in 0..50 {
        let (pos, c, pbc, cutoff) = random_structure(&mut rng);
        let nl = build(&pos, Some(&c), pbc, cutoff, Algorithm::CellList).unwrap();
        let receivers: Vec<u32> = nl.receivers().collect();
        let expected: Vec<u32> = triples(&nl).iter().map(|t| t.0).collect();
        assert_eq!(receivers, expected);
        let perm = nl.sender_permutation();
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..nl.n_edges()).collect::<Vec<_>>());
        for j in 0..nl.n_atoms() {
            let edges = &perm[nl.edges_of(j)];
            assert!(edges.iter().all(|&e| nl.senders[e] == j as u32));
            assert!(edges.windows(2).all(|w| w[0] < w[1]));
        }
    }
}

#[test]
fn rejects_invalid_input() {
    let pos = [[0.0, 0.0, 0.0], [1.0, f64::NAN, 0.0]];
    let c = Cell::cubic(3.0);
    for algo in ALGORITHMS {
        for cutoff in [0.0, -1.0, f64::NAN, f64::INFINITY, 0.99e-8, 1.01e8] {
            let r = build(&pos[..1], Some(&c), ALL, cutoff, algo);
            assert_eq!(r, Err(Error::InvalidCutoff), "{cutoff}");
        }
        // ceil(150/3 + 0.01) = 51 images per axis: 103³ > 2²⁰. With 147,
        // 101³ ≤ 2²⁰.
        let r = build(&pos[..1], Some(&c), ALL, 150.0, algo);
        assert_eq!(r, Err(Error::InvalidCutoff));
        assert!(build(&pos[..1], Some(&c), ALL, 147.0, algo).is_ok());
        assert!(build(&pos[..1], Some(&c), WIRE, 150.0, algo).is_ok());
        assert_eq!(
            build(&pos, Some(&c), ALL, 1.0, algo),
            Err(Error::InvalidPosition(1))
        );
        // Fractional coordinate 2¹⁶ + 1, along a periodic or a non-periodic
        // axis; 2¹⁶ is accepted.
        let far = [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0 * ((1 << 16) + 1) as f64]];
        for pbc in [ALL, SLAB] {
            let r = build(&far, Some(&c), pbc, 1.0, algo);
            assert_eq!(r, Err(Error::InvalidPosition(1)), "{pbc:?}");
        }
        let edge = [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0 * (1 << 16) as f64]];
        assert!(build(&edge, Some(&c), ALL, 1.0, algo).is_ok());
        // Without a cell, positions are not bounded.
        assert!(build(&far, None, NONE, 1.0, algo).is_ok());
        // The fractional coordinate of 1e38 in a cell of edge 1e-8.
        let tiny = Cell::<f32>::from_rows([[1e-8, 0.0, 0.0], [1e-8, 1e-8, 0.0], [0.0, 0.0, 1e-8]]);
        assert_eq!(
            build(
                &[[1e38f32, 5e37, 0.0]],
                Some(&tiny.unwrap()),
                WIRE,
                2e-8,
                algo
            ),
            Err(Error::InvalidPosition(0))
        );
        assert_eq!(
            build(&pos[..1], None, WIRE, 1.0, algo),
            Err(Error::MissingCell)
        );
    }
}

#[test]
fn naive_rejects_too_much_work() {
    // 100 atoms with ceil(147/3 + 0.01) = 50 images per axis: 10⁴·101³ > 2²⁷.
    let pos = [[0.0; 3]; 100];
    let r = build(&pos, Some(&Cell::cubic(3.0)), ALL, 147.0, Algorithm::Naive);
    assert_eq!(r, Err(Error::TooMuchWork));
    // One image per axis: 2230²·27 > 2²⁷ ≥ 2229²·27.
    let pos = [[0.0; 3]; 2230];
    let r = build(&pos, Some(&Cell::cubic(10.0)), ALL, 3.0, Algorithm::Naive);
    assert_eq!(r, Err(Error::TooMuchWork));
    // No images: 11586² > 2²⁷ ≥ 11585².
    let pos = vec![[0.0; 3]; 11586];
    let r = build(&pos, None, NONE, 3.0, Algorithm::Naive);
    assert_eq!(r, Err(Error::TooMuchWork));
}

#[test]
fn cell_list_rejects_too_much_work() {
    // 100 coincident atoms in one bin searched over 101³ images: 10⁴·101³
    // pairs exceed 2²⁷.
    let pos = [[0.0; 3]; 100];
    let r = build(
        &pos,
        Some(&Cell::cubic(3.0)),
        ALL,
        147.0,
        Algorithm::CellList,
    );
    assert_eq!(r, Err(Error::TooMuchWork));
    // Coincident atoms share one bin: 11586² pairs exceed
    // max(2²⁷, 2¹²·11586) = 2²⁷.
    let pos = vec![[0.0; 3]; 11586];
    let r = build(&pos, None, NONE, 3.0, Algorithm::CellList);
    assert_eq!(r, Err(Error::TooMuchWork));
}

#[test]
fn cell_list_large_sparse() {
    // A simple cubic lattice of 16³ atoms with spacing 2, beyond the naive
    // work bound. Within 3 each atom has 6 neighbors at 2 and 12 at √8.
    let pos: Vec<[f64; 3]> = (0..4096)
        .map(|i| [i % 16, i / 16 % 16, i / 256].map(|x| 2.0 * x as f64))
        .collect();
    let c = Cell::cubic(32.0);
    let r = build(&pos, Some(&c), ALL, 3.0, Algorithm::Naive);
    assert_eq!(r, Err(Error::TooMuchWork));
    let nl = build(&pos, Some(&c), ALL, 3.0, Algorithm::CellList).unwrap();
    assert!((0..4096).all(|i| nl.edges_of(i).len() == 18));
    let (near, diagonal) = (2.0, 8f64.sqrt());
    assert!(nl.distances.iter().all(|&d| d == near || d == diagonal));
    check_vectors(&nl, &pos, Some(&c));
}

#[test]
fn cell_list_sparse_diagonal() {
    // 10⁵ atoms 4√3 apart on the diagonal of a bounding box 4·10⁵ wide: the
    // occupied bins are a vanishing fraction of the box. Each atom neighbors
    // the next and the previous one.
    let n = 100_000;
    let pos: Vec<[f64; 3]> = (0..n).map(|i| [4.0 * i as f64; 3]).collect();
    let nl = build(&pos, None, NONE, 7.0, Algorithm::CellList).unwrap();
    assert!((0..n).all(|i| nl.edges_of(i).len() == if i == 0 || i == n - 1 { 1 } else { 2 }));
    check_vectors(&nl, &pos, None);
}
