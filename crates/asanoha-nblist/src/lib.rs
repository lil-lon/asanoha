//! Neighbor list for the asanoha crates: periodic cells and, later, the
//! directed full pair list in receiver-sorted CSR form.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod cell;
mod error;

pub use cell::Cell;
pub use error::Error;
