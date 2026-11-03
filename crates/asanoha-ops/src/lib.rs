//! Node-wise operations of nequix's `NequixConvolution`: e3nn-jax's irreps
//! `Linear`, the gate activation and nequix's RMS layer norm, each with the
//! gradient with respect to its input.
//!
//! All features are in [`Layout::IrMul`](asanoha_core::Layout::IrMul).
//! Weights and normalization constants are data from the exporter.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod gate;
mod linear;
mod norm;

pub use gate::Gate;
pub use linear::{Linear, LinearPath};
pub use norm::RmsNorm;
