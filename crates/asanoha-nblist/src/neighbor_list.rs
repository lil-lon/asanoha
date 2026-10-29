use core::ops::{Range, RangeInclusive};

use asanoha_core::Real;

use crate::{Cell, Error, cell_list, naive};

/// Atoms whose neighbors are searched.
#[derive(Debug, Clone, Copy)]
pub struct Structure<'a, T: Real> {
    /// Cartesian positions; atoms may lie outside the cell.
    pub positions: &'a [[T; 3]],
    /// The cell; `None` is allowed only when no axis is periodic.
    pub cell: Option<&'a Cell<T>>,
    /// Periodicity along `a, b, c`.
    pub pbc: [bool; 3],
}

/// Search algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// All pairs times all images within reach of the cutoff; O(N²). Fails
    /// with [`Error::TooMuchWork`] beyond 2²⁷ pair images.
    Naive,
    /// Bins the atoms by fractional coordinates into bins at least as wide
    /// as the cutoff, stores only the occupied bins, and checks only pairs in
    /// nearby bins; O(N log N + E) at bounded density. The output equals
    /// [`Algorithm::Naive`]'s. Fails with [`Error::TooMuchWork`] beyond
    /// `max(2²⁷, 2¹²·N)` pairs checked.
    CellList,
}

/// Directed full pair list `|r_e|² < r_c²` in receiver-sorted CSR form.
///
/// Edge `e` in `row_offsets[i]..row_offsets[i + 1]` has receiver `i`, sender
/// `j = senders[e]` and shift `S = shifts[e]`, and its vector is
/// `r_e = (positions[j] − positions[i]) + S·H` for the positions as given.
/// Both `(i, j, S)` and `(j, i, −S)` are present, and each row is sorted by
/// `(j, S)`. Every step, including the cutoff test, is evaluated in f64; only
/// `vectors` and `distances` are rounded to `T`.
#[derive(Debug, Clone, PartialEq)]
pub struct NeighborList<T: Real> {
    /// Edge offsets per receiver; length `n_atoms + 1`.
    pub row_offsets: Vec<usize>,
    /// Sender `j` per edge.
    pub senders: Vec<u32>,
    /// Shift `S` per edge, relative to the input positions.
    pub shifts: Vec<[i32; 3]>,
    /// `r_e` per edge.
    pub vectors: Vec<[T; 3]>,
    /// `|r_e|` per edge.
    pub distances: Vec<T>,
}

impl<T: Real> NeighborList<T> {
    /// Builds the list of pairs closer than `cutoff`.
    pub fn build(s: &Structure<'_, T>, cutoff: T, algo: Algorithm) -> Result<Self, Error> {
        let mut nl = Self::empty();
        nl.build_into(s, cutoff, algo)?;
        Ok(nl)
    }

    /// [`NeighborList::build`] into `self`, reusing the capacity of its
    /// vectors. On error `self` is left empty, with no atoms.
    pub fn build_into(
        &mut self,
        s: &Structure<'_, T>,
        cutoff: T,
        algo: Algorithm,
    ) -> Result<(), Error> {
        let result = self.fill(s, cutoff, algo);
        if result.is_err() {
            *self = Self::empty();
        }
        result
    }

    /// A list with no atoms.
    fn empty() -> Self {
        Self {
            row_offsets: vec![0],
            senders: Vec::new(),
            shifts: Vec::new(),
            vectors: Vec::new(),
            distances: Vec::new(),
        }
    }

    /// [`NeighborList::build_into`] before the error handling. The edge
    /// vectors keep their contents until the search sizes or clears them.
    fn fill(&mut self, s: &Structure<'_, T>, cutoff: T, algo: Algorithm) -> Result<(), Error> {
        self.row_offsets.clear();
        self.row_offsets.push(0);
        // The atom count alone may exceed the work bound; check it before
        // allocating per-atom buffers.
        if algo == Algorithm::Naive {
            naive::check_work(s.positions.len(), [0; 3])?;
        }
        let p = Prepared::new(s, cutoff)?;
        match algo {
            Algorithm::Naive => {
                self.truncate_edges(0);
                naive::search(&p, self)
            }
            Algorithm::CellList => cell_list::search(&p, self),
        }
    }

    /// Keeps the first `n` entries of each edge vector.
    fn truncate_edges(&mut self, n: usize) {
        self.senders.truncate(n);
        self.shifts.truncate(n);
        self.vectors.truncate(n);
        self.distances.truncate(n);
    }

    /// Sizes the edge vectors to `row_offsets`, after which every edge is to
    /// be written once with [`NeighborList::set`]. Entries kept from the
    /// previous contents are not cleared.
    pub(crate) fn size_edges(&mut self) {
        let n = self.row_offsets[self.row_offsets.len() - 1];
        self.truncate_edges(n);
        self.senders.resize(n, 0);
        self.shifts.resize(n, [0; 3]);
        self.vectors.resize(n, [T::ZERO; 3]);
        self.distances.resize(n, T::ZERO);
    }

    /// Sets edge `k` to `e` from `j`.
    pub(crate) fn set(&mut self, k: usize, j: usize, e: &Edge) {
        self.senders[k] = j as u32;
        self.shifts[k] = e.shift;
        self.vectors[k] = e.vector.map(T::from_f64);
        self.distances[k] = T::from_f64(e.distance);
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.row_offsets.len() - 1
    }

    /// Number of directed edges.
    pub fn n_edges(&self) -> usize {
        self.senders.len()
    }

    /// Edges whose receiver is atom `i`.
    pub fn edges_of(&self, i: usize) -> Range<usize> {
        self.row_offsets[i]..self.row_offsets[i + 1]
    }

    /// Receiver `i` per edge.
    pub fn receivers(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.n_atoms()).flat_map(|i| core::iter::repeat_n(i as u32, self.edges_of(i).len()))
    }

    /// Edge ids grouped by sender: the edges whose sender is `j` are
    /// `perm[row_offsets[j]..row_offsets[j + 1]]`, in increasing order. Each
    /// edge's reverse is in the list, so atom `j` sends as many edges as it
    /// receives.
    pub fn sender_permutation(&self) -> Vec<usize> {
        let mut next = self.row_offsets[..self.n_atoms()].to_vec();
        let mut perm = vec![0; self.n_edges()];
        for (e, &j) in self.senders.iter().enumerate() {
            perm[next[j as usize]] = e;
            next[j as usize] += 1;
        }
        perm
    }

    /// Appends an edge from `j` to the current row.
    pub(crate) fn push(&mut self, j: usize, e: Edge) {
        self.senders.push(j as u32);
        self.shifts.push(e.shift);
        self.vectors.push(e.vector.map(T::from_f64));
        self.distances.push(T::from_f64(e.distance));
    }

    /// Closes the current row.
    pub(crate) fn end_row(&mut self) {
        self.row_offsets.push(self.senders.len());
    }
}

/// An accepted edge.
pub(crate) struct Edge {
    pub(crate) shift: [i32; 3],
    vector: [f64; 3],
    distance: f64,
}

impl Edge {
    /// The edge in the opposite direction, as [`Prepared::edge`] computes it:
    /// negating the shift negates `S·H` exactly, and `r_i − r_j` is exactly
    /// `−(r_j − r_i)`, so the vector is exactly negated.
    pub(crate) fn reversed(&self) -> Self {
        Self {
            shift: self.shift.map(|x| -x),
            vector: self.vector.map(|x| -x),
            distance: self.distance,
        }
    }
}

/// Accepted cutoffs, the range of accepted cell lengths. Every distance that
/// passes the cutoff test is then representable in f32.
const CUTOFF_RANGE: RangeInclusive<f64> = 1e-8..=1e8;

/// Bound on the number of images searched, `Π(2·images[a] + 1)`.
const MAX_IMAGES: f64 = (1 << 20) as f64;

/// Bound on `floor(r·H⁻¹)` along every axis of the cell. An atom within this
/// many cells of the origin has `|r| < 2¹⁸·L`; with `|H⁻¹| < 1e6/L` (see
/// [`Cell::from_rows`]) its computed `r·H⁻¹` is off by about
/// `10·2¹⁸·1e6·ε ≈ 6e-4` at most.
const MAX_WRAP: f64 = (1 << 16) as f64;

/// Added to `r_c / height` before rounding up the image radius, to cover the
/// error in the fractional coordinates of both atoms of a pair.
const IMAGE_MARGIN: f64 = 1e-2;

/// Lower bound on `r_c / L` with a cell. Edge vectors sum terms up to about
/// `2²⁰·L` (positions within [`MAX_WRAP`] cells, shifts within
/// [`MAX_IMAGES`]), so they are off by about `ε·2²⁰·L ≈ 2e-10·L`; with this
/// bound that is below `1e-5·r_c`, and only pairs that close to the cutoff can
/// be misclassified.
const MIN_CUTOFF_PER_LENGTH: f64 = 1e-4;

/// Relative bound on the rounding error of edge vectors and cell heights;
/// see [`MIN_CUTOFF_PER_LENGTH`].
const VECTOR_ERROR: f64 = 1e-5;

/// Relative widening of the cutoff in [`Prepared::near_sq`]. Positions within
/// [`MAX_WRAP`] cells and shifts within [`MAX_IMAGES`] keep every term of
/// `(r'_j + S·H) − r'_i` with `r' = r − w·H` below about `2²¹·L`, so it and
/// the vector of [`Prepared::edge`] are within about `10·ε·2²¹·L ≈ 5e-9·L` of
/// the true vector, below `1e-4·r_c` by [`MIN_CUTOFF_PER_LENGTH`].
const NEAR_MARGIN: f64 = 1e-3;

/// Bound on the rounding error of a computed fractional coordinate, per unit
/// of `κ·(F + 1)`, where `κ = L³/V` bounds `L·|H⁻¹|` and `F` bounds `|r·H⁻¹|`.
/// The determinant and the cofactors of `H`, and the product `r·H⁻¹`, each
/// contribute at most about `30εκF`.
const FRACTIONAL_ERROR: f64 = 128.0 * f64::EPSILON;

/// Validated input in f64 and the cell images each atom wraps by.
pub(crate) struct Prepared {
    positions: Vec<[f64; 3]>,
    rows: [[f64; 3]; 3],
    /// `w = floor(r·H⁻¹)` on periodic axes, 0 otherwise; `r − w·H` is the
    /// atom wrapped into the cell.
    wraps: Vec<[i32; 3]>,
    /// `f = r·H⁻¹ − w`, in `[0, 1)` on periodic axes; without a cell, the
    /// positions.
    pub(crate) fractional: Vec<[f64; 3]>,
    pub(crate) pbc: [bool; 3],
    /// `ceil(r_c / height + IMAGE_MARGIN)` on periodic axes, 0 otherwise. Any
    /// image within the cutoff of a wrapped atom has `|S[a]| ≤ images[a]`.
    pub(crate) images: [i32; 3],
    /// Every edge `(i, j, S)` closer than the cutoff, with `S` relative to the
    /// wrapped positions, has `|f_j + S − f_i| ≤ reach` along each axis for
    /// the computed `f`: `r_c / height` widened by the rounding errors.
    pub(crate) reach: [f64; 3],
    cutoff_sq: f64,
}

impl Prepared {
    pub(crate) fn new<T: Real>(s: &Structure<'_, T>, cutoff: T) -> Result<Self, Error> {
        let cutoff = cutoff.to_f64();
        if !CUTOFF_RANGE.contains(&cutoff) {
            return Err(Error::InvalidCutoff);
        }
        if s.positions.len() > u32::MAX as usize {
            return Err(Error::TooManyAtoms);
        }
        let positions: Vec<[f64; 3]> = s.positions.iter().map(|r| r.map(T::to_f64)).collect();
        if let Some(i) = positions
            .iter()
            .position(|r| !r.iter().all(|x| x.is_finite()))
        {
            return Err(Error::InvalidPosition(i));
        }
        let Some(cell) = s.cell else {
            if s.pbc != [false; 3] {
                return Err(Error::MissingCell);
            }
            // The positions serve as fractional coordinates of a unit cell,
            // exact but rounded in binning by about ε times the coordinates
            // along the same axis.
            let max: [f64; 3] =
                core::array::from_fn(|a| positions.iter().fold(0.0, |m, r| r[a].abs().max(m)));
            return Ok(Self {
                wraps: vec![[0; 3]; positions.len()],
                fractional: positions.clone(),
                positions,
                rows: [[0.0; 3]; 3],
                pbc: s.pbc,
                images: [0; 3],
                reach: max
                    .map(|m| cutoff * (1.0 + VECTOR_ERROR) + 2.0 * FRACTIONAL_ERROR * (m + 1.0)),
                cutoff_sq: cutoff * cutoff,
            });
        };
        // Cell<T> made the same f64 computation, so this cell is accepted too.
        let cell = Cell::from_rows(cell.rows().map(|r| r.map(T::to_f64)))
            .expect("a cell accepted in T is accepted in f64");
        let length = cell
            .rows()
            .iter()
            .map(|r| (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt())
            .fold(0.0, f64::max);
        if cutoff < MIN_CUTOFF_PER_LENGTH * length {
            return Err(Error::InvalidCutoff);
        }
        let heights = cell.heights();
        let images: [f64; 3] = core::array::from_fn(|a| {
            if s.pbc[a] {
                (cutoff / heights[a] + IMAGE_MARGIN).ceil()
            } else {
                0.0
            }
        });
        if images.iter().map(|n| 2.0 * n + 1.0).product::<f64>() > MAX_IMAGES {
            return Err(Error::InvalidCutoff);
        }
        let mut wraps = Vec::with_capacity(positions.len());
        let mut fractional = Vec::with_capacity(positions.len());
        let mut max = 0.0;
        for (i, &r) in positions.iter().enumerate() {
            let f = cell.to_fractional(r);
            if !f
                .iter()
                .all(|x| (-MAX_WRAP..=MAX_WRAP).contains(&x.floor()))
            {
                return Err(Error::InvalidPosition(i));
            }
            let w: [i32; 3] =
                core::array::from_fn(|a| if s.pbc[a] { f[a].floor() as i32 } else { 0 });
            wraps.push(w);
            fractional.push(core::array::from_fn(|a| f[a] - f64::from(w[a])));
            max = f.iter().fold(max, |m: f64, x| x.abs().max(m));
        }
        let delta = FRACTIONAL_ERROR * length.powi(3) / cell.volume() * (max + 1.0);
        Ok(Self {
            positions,
            rows: *cell.rows(),
            wraps,
            fractional,
            pbc: s.pbc,
            images: images.map(|n| n as i32),
            reach: heights.map(|h| cutoff * (1.0 + VECTOR_ERROR) / h + 2.0 * delta),
            cutoff_sq: cutoff * cutoff,
        })
    }

    pub(crate) fn n_atoms(&self) -> usize {
        self.positions.len()
    }

    /// `r − w·H`: atom `i` wrapped into the cell.
    pub(crate) fn wrapped(&self, i: usize) -> [f64; 3] {
        let t = self.translation(self.wraps[i]);
        core::array::from_fn(|k| self.positions[i][k] - t[k])
    }

    /// `S·H`.
    pub(crate) fn translation(&self, s: [i32; 3]) -> [f64; 3] {
        let h = &self.rows;
        let s = s.map(f64::from);
        core::array::from_fn(|k| s[0] * h[0][k] + s[1] * h[1][k] + s[2] * h[2][k])
    }

    /// Square of the cutoff widened by [`NEAR_MARGIN`]: every pair that
    /// [`Prepared::edge`] accepts has `|(r'_j + S·H) − r'_i|²` below it, for
    /// [`Prepared::wrapped`] `r'` and [`Prepared::translation`] `S·H`.
    /// Without a cell both vectors are `r_j − r_i`.
    pub(crate) fn near_sq(&self) -> f64 {
        self.cutoff_sq * (1.0 + NEAR_MARGIN) * (1.0 + NEAR_MARGIN)
    }

    /// The edge from `j` shifted by `S` to `i`, if closer than the cutoff.
    /// `S` is relative to the wrapped positions.
    pub(crate) fn edge(&self, i: usize, j: usize, s: [i32; 3]) -> Option<Edge> {
        let shift: [i32; 3] = core::array::from_fn(|k| s[k] + self.wraps[i][k] - self.wraps[j][k]);
        // From the input positions, so that the difference of nearby atoms is
        // exact. Negating the shift negates the translation exactly, so the
        // reverse edge gets exactly −r.
        let t = self.translation(shift);
        let (ri, rj) = (self.positions[i], self.positions[j]);
        let v: [f64; 3] = core::array::from_fn(|k| (rj[k] - ri[k]) + t[k]);
        let d2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        (d2 < self.cutoff_sq).then(|| Edge {
            shift,
            vector: v,
            distance: d2.sqrt(),
        })
    }
}
