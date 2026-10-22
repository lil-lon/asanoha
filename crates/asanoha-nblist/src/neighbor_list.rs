use core::ops::{Range, RangeInclusive};

use asanoha_core::Real;

use crate::{Cell, Error, naive};

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
    /// All pairs times all images within reach of the cutoff; O(N²).
    Naive,
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
        let p = Prepared::new(s, cutoff)?;
        let mut nl = Self {
            row_offsets: vec![0],
            senders: Vec::new(),
            shifts: Vec::new(),
            vectors: Vec::new(),
            distances: Vec::new(),
        };
        match algo {
            Algorithm::Naive => naive::search(&p, &mut nl),
        }
        Ok(nl)
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
    shift: [i32; 3],
    vector: [f64; 3],
    distance: f64,
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

/// Validated input in f64 and the cell images each atom wraps by.
pub(crate) struct Prepared {
    positions: Vec<[f64; 3]>,
    rows: [[f64; 3]; 3],
    /// `w = floor(r·H⁻¹)` on periodic axes, 0 otherwise; `r − w·H` is the
    /// atom wrapped into the cell.
    wraps: Vec<[i32; 3]>,
    /// `ceil(r_c / height + IMAGE_MARGIN)` on periodic axes, 0 otherwise. Any
    /// image within the cutoff of a wrapped atom has `|S[a]| ≤ images[a]`.
    pub(crate) images: [i32; 3],
    cutoff_sq: f64,
}

impl Prepared {
    fn new<T: Real>(s: &Structure<'_, T>, cutoff: T) -> Result<Self, Error> {
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
            return Ok(Self {
                wraps: vec![[0; 3]; positions.len()],
                positions,
                rows: [[0.0; 3]; 3],
                images: [0; 3],
                cutoff_sq: cutoff * cutoff,
            });
        };
        // Cell<T> made the same f64 computation, so this cell is accepted too.
        let cell = Cell::from_rows(cell.rows().map(|r| r.map(T::to_f64)))
            .expect("a cell accepted in T is accepted in f64");
        let heights = cell.heights();
        let reach: [f64; 3] = core::array::from_fn(|a| {
            if s.pbc[a] {
                (cutoff / heights[a] + IMAGE_MARGIN).ceil()
            } else {
                0.0
            }
        });
        if reach.iter().map(|n| 2.0 * n + 1.0).product::<f64>() > MAX_IMAGES {
            return Err(Error::InvalidCutoff);
        }
        let mut wraps = Vec::with_capacity(positions.len());
        for (i, &r) in positions.iter().enumerate() {
            let f = cell.to_fractional(r).map(f64::floor);
            if !f.iter().all(|x| (-MAX_WRAP..=MAX_WRAP).contains(x)) {
                return Err(Error::InvalidPosition(i));
            }
            wraps.push(core::array::from_fn(
                |a| if s.pbc[a] { f[a] as i32 } else { 0 },
            ));
        }
        Ok(Self {
            positions,
            rows: *cell.rows(),
            wraps,
            images: reach.map(|n| n as i32),
            cutoff_sq: cutoff * cutoff,
        })
    }

    pub(crate) fn n_atoms(&self) -> usize {
        self.positions.len()
    }

    /// The edge from `j` shifted by `S` to `i`, if closer than the cutoff.
    /// `S` is relative to the wrapped positions.
    pub(crate) fn edge(&self, i: usize, j: usize, s: [i32; 3]) -> Option<Edge> {
        let shift: [i32; 3] = core::array::from_fn(|a| s[a] + self.wraps[i][a] - self.wraps[j][a]);
        // From the input positions, so that the difference of nearby atoms is
        // exact. Negating the shift negates the translation exactly, so the
        // reverse edge gets exactly −r.
        let t = shift.map(f64::from);
        let h = &self.rows;
        let (ri, rj) = (self.positions[i], self.positions[j]);
        let v: [f64; 3] = core::array::from_fn(|k| {
            (rj[k] - ri[k]) + (t[0] * h[0][k] + t[1] * h[1][k] + t[2] * h[2][k])
        });
        let d2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        (d2 < self.cutoff_sq).then(|| Edge {
            shift,
            vector: v,
            distance: d2.sqrt(),
        })
    }
}
