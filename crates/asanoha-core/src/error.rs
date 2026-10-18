use core::fmt;

/// Errors shared by the asanoha crates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Malformed irreps string or irreps mismatch.
    Irreps(String),
    /// Malformed JSON.
    Json(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Irreps(msg) => write!(f, "irreps error: {msg}"),
            Error::Json(msg) => write!(f, "JSON error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
