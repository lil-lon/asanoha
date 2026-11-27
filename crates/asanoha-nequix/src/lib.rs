//! nequix's interatomic potential assembled from the asanoha kernels: its
//! interaction layer `NequixConvolution` and the gradient of the layer with
//! respect to its inputs.
//!
//! [`Layer`] follows `NequixConvolution.__call__` in nequix (MIT)
//! `nequix/model.py`. Its license is reproduced in
//! `THIRD_PARTY_LICENSES/nequix-LICENSE` for attribution.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod layer;

pub use layer::{Activations, Layer};
