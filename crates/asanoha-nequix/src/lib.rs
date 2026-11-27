//! nequix's interatomic potential assembled from the asanoha kernels: its
//! interaction layer `NequixConvolution` with the gradient of the layer with
//! respect to its inputs, and the model `Nequix` with the energy, forces and
//! virial of a structure.
//!
//! [`Layer`] follows `NequixConvolution.__call__` and [`Model`] follows
//! `Nequix` in nequix (MIT) `nequix/model.py`. Its license is reproduced in
//! `THIRD_PARTY_LICENSES/nequix-LICENSE` for attribution.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod layer;
mod model;

pub use layer::{Activations, Layer};
pub use model::{Evaluation, Model};
