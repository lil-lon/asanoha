//! Neighbor list for the asanoha crates: periodic cells and the directed full
//! pair list in receiver-sorted CSR form.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod cell;
mod cell_list;
mod error;
mod naive;
mod neighbor_list;

pub use cell::Cell;
pub use error::Error;
pub use neighbor_list::{Algorithm, NeighborList, Structure};
