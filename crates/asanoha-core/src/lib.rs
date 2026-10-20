//! Numeric trait, irreps bookkeeping, feature layouts, and JSON and
//! safetensors readers shared by the asanoha crates. Nothing here is
//! model-specific.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod features;
mod irreps;
pub mod json;
mod real;
pub mod safetensors;
#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use error::Error;
pub use features::{Features, Layout};
pub use irreps::{Irrep, Irreps, MulIrrep, Parity};
pub use real::Real;
