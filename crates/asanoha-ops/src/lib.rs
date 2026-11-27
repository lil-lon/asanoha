//! Node-wise operations of nequix's `NequixConvolution`: e3nn-jax's irreps
//! `Linear`, the gate activation and nequix's RMS layer norm, each with the
//! gradient with respect to its input.
//!
//! All features are in [`Layout::IrMul`](asanoha_core::Layout::IrMul).
//! Weights and normalization constants are data from the exporter.
//!
//! `Linear` and `Gate` follow `FunctionalLinear` and `gate` in e3nn/e3nn-jax
//! (Apache-2.0) `e3nn_jax/_src/linear.py` and `e3nn_jax/_src/gate.py`, and
//! `RmsNorm` is transcribed from `RMSLayerNorm` in nequix (MIT)
//! `nequix/layer_norm.py`. Their licenses are reproduced in
//! `THIRD_PARTY_LICENSES/` for attribution.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod gate;
mod linear;
mod norm;

pub use gate::Gate;
pub use linear::{Linear, LinearPath};
pub use norm::RmsNorm;
