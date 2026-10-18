//! Numeric trait, irreps bookkeeping and feature layouts shared by the asanoha
//! crates. Nothing here is model-specific.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod features;
mod irreps;
mod real;
#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use error::Error;
pub use features::{Features, Layout};
pub use irreps::{Irrep, Irreps, MulIrrep, Parity};
pub use real::Real;
