use core::fmt;

/// Errors shared by the asanoha crates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Malformed irreps string or irreps mismatch.
    Irreps(String),
    /// Malformed JSON.
    Json(String),
    /// Malformed file contents.
    Format(String),
    /// A tensor dtype the reader or the requested conversion does not handle.
    UnsupportedDtype(String),
    /// No tensor with this name.
    MissingTensor(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Irreps(msg) => write!(f, "irreps error: {msg}"),
            Error::Json(msg) => write!(f, "JSON error: {msg}"),
            Error::Format(msg) => write!(f, "format error: {msg}"),
            Error::UnsupportedDtype(msg) => write!(f, "unsupported dtype: {msg}"),
            Error::MissingTensor(name) => write!(f, "missing tensor {name:?}"),
        }
    }
}

impl std::error::Error for Error {}
