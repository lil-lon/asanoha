use asanoha_core::Real;

use crate::naive::MAX_CANDIDATES;
use crate::neighbor_list::{Edge, Prepared};
use crate::{Error, NeighborList};

/// Bins of the fractional coordinate along one axis.
struct Axis {
    periodic: bool,
    /// Number of bins.
    n: usize,
    /// Lower end of the binned range.
    lo: f64,
    /// Bins per unit of the coordinate.
    scale: f64,
    /// Bins searched on each side of an atom's bin.
    search: i64,
}

impl Axis {
    /// Bins `n` over `[lo, lo + span]`; `[0, 1)` on a periodic axis.
    fn new(periodic: bool, lo: f64, span: f64, n: usize, reach: f64) -> Self {
        let scale = if periodic {
            n as f64
        } else if n == 1 {
            0.0
        } else {
            n as f64 / span
        };
        // Coordinates at most `reach` apart are at most this many bins apart.
        let search = (reach * scale) as i64 + 1;
        Self {
            periodic,
            n,
            lo,
            scale,
            search: if periodic {
                search
            } else {
                search.min(n as i64 - 1)
            },
        }
    }

    /// The bin of coordinate `f`.
    fn bin(&self, f: f64) -> usize {
        // The cast saturates below 0.
        (((f - self.lo) * self.scale) as usize).min(self.n - 1)
    }

    /// The bins searched from bin `b`, each with the shift `S` of its image.
    fn targets(&self, b: usize) -> impl Iterator<Item = (usize, i32)> + '_ {
        let (n, b) = (self.n as i64, b as i64);
        (b - self.search..=b + self.search).filter_map(move |raw| {
            if self.periodic {
                Some((raw.rem_euclid(n) as usize, raw.div_euclid(n) as i32))
            } else {
                (0..n).contains(&raw).then_some((raw as usize, 0))
            }
        })
    }
}

/// The atoms sorted into bins; only occupied bins are stored.
struct Bins {
    axes: [Axis; 3],
    /// Bin of each atom along each axis.
    coords: Vec<[usize; 3]>,
    /// The occupied bins in increasing order.
    keys: Vec<[usize; 3]>,
    /// The atoms of bin `keys[k]` are `atoms[offsets[k]..offsets[k + 1]]`, in
    /// increasing order.
    offsets: Vec<usize>,
    atoms: Vec<usize>,
}

impl Bins {
    /// Bins at least `reach` wide, so that the search spans one bin on each
    /// side of an axis longer than the reach.
    fn new(p: &Prepared) -> Self {
        let f = &p.fractional;
        // `(lo, span)`: non-periodic axes are binned over the atoms' extent.
        let range: [(f64, f64); 3] = core::array::from_fn(|a| {
            if p.pbc[a] {
                (0.0, 1.0)
            } else {
                let lo = f.iter().fold(f64::INFINITY, |m, x| x[a].min(m));
                let hi = f.iter().fold(f64::NEG_INFINITY, |m, x| x[a].max(m));
                (lo, hi - lo)
            }
        });
        let axes: [Axis; 3] = core::array::from_fn(|a| {
            // The cast saturates at u32::MAX bins, beyond any physical extent,
            // so that bin indices fit in i64 even if the span overflows.
            let n = ((range[a].1 / p.reach[a]) as u32).max(1) as usize;
            Axis::new(p.pbc[a], range[a].0, range[a].1, n, p.reach[a])
        });
        let coords: Vec<[usize; 3]> = f
            .iter()
            .map(|x| core::array::from_fn(|a| axes[a].bin(x[a])))
            .collect();
        // The sort is stable, so the atoms of a bin stay in increasing order.
        let mut atoms: Vec<usize> = (0..f.len()).collect();
        atoms.sort_by_key(|&i| coords[i]);
        let (mut keys, mut offsets) = (Vec::new(), Vec::new());
        for (k, &i) in atoms.iter().enumerate() {
            if keys.last() != Some(&coords[i]) {
                keys.push(coords[i]);
                offsets.push(k);
            }
        }
        offsets.push(atoms.len());
        Self {
            axes,
            coords,
            keys,
            offsets,
            atoms,
        }
    }

    fn atoms_in(&self, c: [usize; 3]) -> &[usize] {
        match self.keys.binary_search(&c) {
            Ok(k) => &self.atoms[self.offsets[k]..self.offsets[k + 1]],
            Err(_) => &[],
        }
    }

    /// The bins searched from bin `c`, each with the shift `S` of its image.
    fn neighbors(&self, c: [usize; 3]) -> impl Iterator<Item = ([usize; 3], [i32; 3])> + '_ {
        let [ax, ay, az] = &self.axes;
        az.targets(c[2]).flat_map(move |(z, sz)| {
            ay.targets(c[1]).flat_map(move |(y, sy)| {
                ax.targets(c[0])
                    .map(move |(x, sx)| ([x, y, z], [sx, sy, sz]))
            })
        })
    }

    /// Whether the search checks more than `limit` pairs `(i, j, S)`; stops
    /// counting once it does.
    fn exceeds(&self, limit: f64) -> bool {
        let mut total = 0.0;
        for (k, &c) in self.keys.iter().enumerate() {
            let here = (self.offsets[k + 1] - self.offsets[k]) as f64;
            for (t, _) in self.neighbors(c) {
                total += here * self.atoms_in(t).len() as f64;
                if total > limit {
                    return true;
                }
            }
        }
        false
    }
}

/// Bound on the pairs checked per atom, beyond [`MAX_CANDIDATES`] in total.
/// Dense structures with bins as wide as the cutoff check a few hundred.
const CANDIDATES_PER_ATOM: f64 = (1 << 12) as f64;

/// Checks each atom against the atoms in the bins within reach, then sorts
/// its row into the canonical order `(j, S)`. Fails with
/// [`Error::TooMuchWork`] beyond `max(MAX_CANDIDATES, CANDIDATES_PER_ATOM·N)`
/// pairs checked.
pub(crate) fn search<T: Real>(p: &Prepared, nl: &mut NeighborList<T>) -> Result<(), Error> {
    let bins = Bins::new(p);
    if bins.exceeds(MAX_CANDIDATES.max(CANDIDATES_PER_ATOM * p.n_atoms() as f64)) {
        return Err(Error::TooMuchWork);
    }
    let mut row: Vec<(usize, Edge)> = Vec::new();
    for (i, &c) in bins.coords.iter().enumerate() {
        for (t, s) in bins.neighbors(c) {
            for &j in bins.atoms_in(t) {
                if i == j && s == [0; 3] {
                    continue;
                }
                if let Some(e) = p.edge(i, j, s) {
                    row.push((j, e));
                }
            }
        }
        row.sort_unstable_by_key(|(j, e)| (*j, e.shift));
        for (j, e) in row.drain(..) {
            nl.push(j, e);
        }
        nl.end_row();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cell, Structure};

    fn bin_counts(
        positions: &[[f64; 3]],
        cell: Option<&Cell<f64>>,
        pbc: [bool; 3],
        cutoff: f64,
    ) -> [usize; 3] {
        let s = Structure {
            positions,
            cell,
            pbc,
        };
        Bins::new(&Prepared::new(&s, cutoff).unwrap())
            .axes
            .map(|a| a.n)
    }

    #[test]
    fn bin_counts_follow_heights() {
        let pos = [[0.0; 3]; 100];
        // Heights 10, 20, 7 give floor(height / 3) bins; a height of exactly
        // 9 gives 2, as the rounding margin widens the reach.
        let rows = [[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 7.0]];
        let c = Cell::from_rows(rows).unwrap();
        assert_eq!(bin_counts(&pos, Some(&c), [true; 3], 3.0), [3, 6, 2]);
        assert_eq!(
            bin_counts(&pos, Some(&Cell::cubic(9.0)), [true; 3], 3.0),
            [2; 3]
        );
        // Triclinic: heights 3, 3, 5 while every edge has length 5.
        let rows = [[5.0, 0.0, 0.0], [4.0, 3.0, 0.0], [0.0, 0.0, 5.0]];
        let c = Cell::from_rows(rows).unwrap();
        assert_eq!(bin_counts(&pos, Some(&c), [true; 3], 1.4), [2, 2, 3]);
    }

    #[test]
    fn bin_counts_follow_extent_on_non_periodic_axes() {
        // Extent 9.5 along x and 0 along y and z.
        let line: Vec<[f64; 3]> = (0..20).map(|i| [0.5 * i as f64, 1.0, 1.0]).collect();
        let c = Cell::cubic(100.0);
        assert_eq!(bin_counts(&line, Some(&c), [false; 3], 3.0), [3, 1, 1]);
        assert_eq!(bin_counts(&line, None, [false; 3], 3.0), [3, 1, 1]);
        // A large offset along x leaves the bins along y as wide as the
        // cutoff: extent 76, less one bin as the rounding margin widens the
        // reach.
        let pos: Vec<[f64; 3]> = (0..20).map(|i| [1e20, 4.0 * i as f64, 0.0]).collect();
        assert_eq!(bin_counts(&pos, None, [false; 3], 1.0), [1, 75, 1]);
        // Periodic y and z have 33 bins each, more in total than atoms.
        assert_eq!(
            bin_counts(&line, Some(&c), [false, true, true], 3.0),
            [3, 33, 33]
        );
    }

    #[test]
    fn only_occupied_bins_are_stored() {
        let pos = [[0.0; 3], [1.0; 3], [20.0; 3]];
        let s = Structure {
            positions: &pos,
            cell: Some(&Cell::cubic(30.0)),
            pbc: [true; 3],
        };
        let bins = Bins::new(&Prepared::new(&s, 3.0).unwrap());
        assert_eq!(bins.axes.each_ref().map(|a| a.n), [9; 3]);
        assert_eq!(bins.keys, [[0; 3], [6; 3]]);
        assert_eq!(bins.atoms_in([0; 3]), [0, 1]);
        assert_eq!(bins.atoms_in([6; 3]), [2]);
        assert_eq!(bins.atoms_in([1; 3]), []);
        assert_eq!(bin_counts(&[], None, [false; 3], 3.0), [1, 1, 1]);
    }
}
