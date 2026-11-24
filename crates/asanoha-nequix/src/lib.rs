//! nequix's interatomic potential assembled from the asanoha kernels: its
//! interaction layer `NequixConvolution` and the gradient of the layer with
//! respect to its inputs.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod layer;

pub use layer::{Activations, Layer};
