use core::fmt;

/// Errors returned by the neighbor list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The cell matrix is singular, not finite, or its lattice vectors are
    /// outside the supported length range.
    InvalidCell,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidCell => f.write_str("invalid cell"),
        }
    }
}

impl std::error::Error for Error {}
