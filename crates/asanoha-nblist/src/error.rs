use core::fmt;

/// Errors returned by the neighbor list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The cell matrix is singular or not finite.
    SingularCell,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::SingularCell => f.write_str("singular cell"),
        }
    }
}

impl std::error::Error for Error {}
