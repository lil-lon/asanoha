use core::ops::{Range, RangeInclusive};

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
        self.runs(b).flat_map(|(bins, s)| bins.map(move |c| (c, s)))
    }

    /// [`Axis::targets`] as runs of consecutive bins with the same shift.
    fn runs(&self, b: usize) -> impl Iterator<Item = (RangeInclusive<usize>, i32)> {
        let (n, b) = (self.n as i64, b as i64);
        let (lo, hi) = if self.periodic {
            (b - self.search, b + self.search)
        } else {
            ((b - self.search).max(0), (b + self.search).min(n - 1))
        };
        (lo.div_euclid(n)..=hi.div_euclid(n)).map(move |s| {
            let start = lo.max(s * n) - s * n;
            let end = hi.min(s * n + n - 1) - s * n;
            (start as usize..=end as usize, s as i32)
        })
    }
}

/// Bins at least `reach` wide, so that the search spans one bin on each side
/// of an axis longer than the reach.
fn axes(p: &Prepared) -> [Axis; 3] {
    let f = &p.fractional;
    core::array::from_fn(|a| {
        // `(lo, span)`: non-periodic axes are binned over the atoms' extent.
        let (lo, span) = if p.pbc[a] {
            (0.0, 1.0)
        } else {
            let lo = f.iter().fold(f64::INFINITY, |m, x| x[a].min(m));
            let hi = f.iter().fold(f64::NEG_INFINITY, |m, x| x[a].max(m));
            (lo, hi - lo)
        };
        // The cast saturates at u32::MAX bins, beyond any physical extent, so
        // that bin indices fit in i64 even if the span overflows.
        let n = ((span / p.reach[a]) as u32).max(1) as usize;
        Axis::new(p.pbc[a], lo, span, n, p.reach[a])
    })
}

/// Consecutive bins along `c` with the same shift, searched from another bin
/// and holding at least one atom.
struct Target {
    /// Its atoms, as a range of [`Bins::atoms`].
    atoms: Range<usize>,
    /// The shift `S` of its image, relative to the wrapped positions.
    shift: [i32; 3],
    /// `S·H`.
    translation: [f64; 3],
}

/// The atoms sorted into bins, and for each occupied bin the bins it
/// searches; empty bins are not stored. Each [`Target`] adds at least one
/// pair checked, so there are no more targets than pairs checked.
struct Bins {
    /// The atoms in bin order; the atoms of a bin are consecutive and in
    /// increasing order.
    atoms: Vec<usize>,
    /// [`Prepared::wrapped`] of `atoms`, in the same order.
    wrapped: Vec<[f64; 3]>,
    /// Occupied bin of each atom, numbered in increasing bin order.
    bin_of: Vec<usize>,
    /// The bins searched from occupied bin `k`, in runs along `c`, are
    /// `targets[target_offsets[k]..target_offsets[k + 1]]`.
    target_offsets: Vec<usize>,
    targets: Vec<Target>,
}

impl Bins {
    /// Fails with [`Error::TooMuchWork`] if the search checks more than
    /// `limit` pairs `(i, j, S)`.
    fn new(p: &Prepared, limit: f64) -> Result<Self, Error> {
        let axes = axes(p);
        let coords: Vec<[usize; 3]> = p
            .fractional
            .iter()
            .map(|x| core::array::from_fn(|a| axes[a].bin(x[a])))
            .collect();
        // The sort is stable, so the atoms of a bin stay in increasing order.
        let mut atoms: Vec<usize> = (0..coords.len()).collect();
        atoms.sort_by_key(|&i| coords[i]);
        let (mut keys, mut offsets) = (Vec::new(), Vec::new());
        let mut bin_of = vec![0; atoms.len()];
        for (m, &i) in atoms.iter().enumerate() {
            if keys.last() != Some(&coords[i]) {
                keys.push(coords[i]);
                offsets.push(m);
            }
            bin_of[i] = keys.len() - 1;
        }
        offsets.push(atoms.len());
        // The occupied columns [x, y], each with its bins as a range of `keys`.
        let mut columns: Vec<([usize; 2], Range<usize>)> = Vec::new();
        for (k, c) in keys.iter().enumerate() {
            match columns.last_mut() {
                Some((xy, bins)) if *xy == [c[0], c[1]] => bins.end = k + 1,
                _ => columns.push(([c[0], c[1]], k..k + 1)),
            }
        }
        let (mut target_offsets, mut targets) = (vec![0], Vec::new());
        let mut total = 0.0;
        let [ax, ay, az] = &axes;
        let mut near = Vec::new();
        for (xy, bins) in &columns {
            // The occupied columns searched from this one, with their shifts.
            near.clear();
            for (x, sx) in ax.targets(xy[0]) {
                for (y, sy) in ay.targets(xy[1]) {
                    if let Ok(m) = columns.binary_search_by_key(&[x, y], |c| c.0) {
                        near.push((&keys[columns[m].1.clone()], columns[m].1.start, sx, sy));
                    }
                }
            }
            for k in bins.clone() {
                let here = (offsets[k + 1] - offsets[k]) as f64;
                for &(column, start, sx, sy) in &near {
                    for (z, sz) in az.runs(keys[k][2]) {
                        let first = start + column.partition_point(|t| t[2] < *z.start());
                        let last = start + column.partition_point(|t| t[2] <= *z.end());
                        let atoms = offsets[first]..offsets[last];
                        if atoms.is_empty() {
                            continue;
                        }
                        total += here * atoms.len() as f64;
                        if total > limit {
                            return Err(Error::TooMuchWork);
                        }
                        let shift = [sx, sy, sz];
                        targets.push(Target {
                            atoms,
                            shift,
                            translation: p.translation(shift),
                        });
                    }
                }
                target_offsets.push(targets.len());
            }
        }
        Ok(Self {
            wrapped: atoms.iter().map(|&i| p.wrapped(i)).collect(),
            atoms,
            bin_of,
            target_offsets,
            targets,
        })
    }

    /// The bins searched from the bin of atom `i`.
    fn targets_of(&self, i: usize) -> &[Target] {
        let k = self.bin_of[i];
        &self.targets[self.target_offsets[k]..self.target_offsets[k + 1]]
    }
}

/// Bound on the pairs checked per atom, beyond [`MAX_CANDIDATES`] in total.
/// Dense structures with bins as wide as the cutoff check a few hundred.
const CANDIDATES_PER_ATOM: f64 = (1 << 12) as f64;

/// Sort key of an edge in the canonical row order `(j, S)`.
fn row_key(j: usize, s: [i32; 3]) -> u128 {
    // Flipping the sign bit maps the order of i32 onto that of u32.
    s.iter().fold(j as u128, |k, &x| {
        k << 32 | u128::from(x as u32 ^ 0x8000_0000)
    })
}

/// The sender `j` of a row key.
fn sender(key: u128) -> usize {
    (key >> 96) as usize
}

/// Computes each pair once, from its lower atom, and writes it and its
/// reverse; for the self-images of an atom, `S` and `−S`, it computes the
/// lexicographically positive one. A pair goes through the exact
/// [`Prepared::edge`] only if its vector from the wrapped positions passes
/// [`Prepared::near_sq`]. Fails with [`Error::TooMuchWork`] beyond
/// `max(MAX_CANDIDATES, CANDIDATES_PER_ATOM·N)` pairs checked. Writes into
/// the empty list `nl`.
pub(crate) fn search<T: Real>(p: &Prepared, nl: &mut NeighborList<T>) -> Result<(), Error> {
    let n = p.n_atoms();
    let bins = Bins::new(p, MAX_CANDIDATES.max(CANDIDATES_PER_ATOM * n as f64))?;
    let near_sq = p.near_sq();
    // The edges of atom i to atoms j ≥ i are half[half_offsets[i]..half_offsets[i + 1]],
    // sorted by (j, S).
    let mut half: Vec<(u128, Edge)> = Vec::new();
    let mut half_offsets = Vec::with_capacity(n + 1);
    half_offsets.push(0);
    let mut degree = vec![0; n];
    for i in 0..n {
        let start = half.len();
        let ri = p.wrapped(i);
        for t in bins.targets_of(i) {
            let r = t.atoms.clone();
            for (&j, rj) in bins.atoms[r.clone()].iter().zip(&bins.wrapped[r]) {
                if j < i {
                    continue;
                }
                let v: [f64; 3] = core::array::from_fn(|k| (rj[k] + t.translation[k]) - ri[k]);
                if v[0] * v[0] + v[1] * v[1] + v[2] * v[2] >= near_sq
                    || (j == i && t.shift <= [0; 3])
                {
                    continue;
                }
                if let Some(e) = p.edge(i, j, t.shift) {
                    half.push((row_key(j, e.shift), e));
                }
            }
        }
        half[start..].sort_unstable_by_key(|&(k, _)| k);
        for &(k, _) in &half[start..] {
            let j = sender(k);
            degree[i] += 1;
            degree[j] += 1;
        }
        half_offsets.push(half.len());
    }
    for d in &degree {
        nl.row_offsets
            .push(nl.row_offsets[nl.row_offsets.len() - 1] + d);
    }
    let mut next = nl.row_offsets[..n].to_vec();
    nl.size_edges();
    // Row i receives its edges from atoms j < i while those atoms are
    // processed, in increasing order of j; then its self-images, −S before S;
    // then its edges to atoms j > i.
    for i in 0..n {
        let row = &half[half_offsets[i]..half_offsets[i + 1]];
        let (own, others) = row.split_at(row.partition_point(|&(k, _)| sender(k) == i));
        for (_, e) in own.iter().rev() {
            nl.set(next[i], i, &e.reversed());
            next[i] += 1;
        }
        for (k, e) in own.iter().chain(others) {
            nl.set(next[i], sender(*k), e);
            next[i] += 1;
        }
        // Row j needs the edges from i in increasing −S, the reverse of
        // their order here.
        for run in others.chunk_by(|a, b| sender(a.0) == sender(b.0)) {
            let j = sender(run[0].0);
            for (_, e) in run.iter().rev() {
                nl.set(next[j], i, &e.reversed());
                next[j] += 1;
            }
        }
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
        axes(&Prepared::new(&s, cutoff).unwrap()).map(|a| a.n)
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
        let p = Prepared::new(&s, 3.0).unwrap();
        assert_eq!(axes(&p).map(|a| a.n), [9; 3]);
        // Bins [0; 3] and [6; 3] are occupied; each searches only itself, as
        // the other is 6 bins away.
        let bins = Bins::new(&p, f64::INFINITY).unwrap();
        assert_eq!(bins.atoms, [0, 1, 2]);
        assert_eq!(bins.bin_of, [0, 0, 1]);
        assert_eq!(bins.target_offsets, [0, 1, 2]);
        let searched = |i| -> Vec<&[usize]> {
            bins.targets_of(i)
                .iter()
                .map(|t| &bins.atoms[t.atoms.clone()])
                .collect()
        };
        assert_eq!(searched(0), [[0, 1]]);
        assert_eq!(searched(2), [[2]]);
        assert_eq!(bin_counts(&[], None, [false; 3], 3.0), [1, 1, 1]);
    }
}
