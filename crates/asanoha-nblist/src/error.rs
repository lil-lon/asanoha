use core::fmt;

/// Errors returned by the neighbor list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The cell matrix is singular, not finite, or its lattice vectors are
    /// outside the supported length range.
    InvalidCell,
    /// The cutoff is outside `[1e-8, 1e8]`, or reaches more than 2²⁰ cell
    /// images.
    InvalidCutoff,
    /// The position of this atom is not finite, or lies more than 2¹⁶ cells
    /// from the origin along an axis of the cell.
    InvalidPosition(usize),
    /// An axis is periodic but no cell is given.
    MissingCell,
    /// More atoms than `u32` indices can address.
    TooManyAtoms,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidCell => f.write_str("invalid cell"),
            Error::InvalidCutoff => f.write_str("invalid cutoff"),
            Error::InvalidPosition(i) => write!(f, "invalid position of atom {i}"),
            Error::MissingCell => f.write_str("periodic axis without a cell"),
            Error::TooManyAtoms => f.write_str("too many atoms"),
        }
    }
}

impl std::error::Error for Error {}
