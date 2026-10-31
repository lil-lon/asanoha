//! Radial part of the edge features: the Bessel basis times the polynomial
//! cutoff, and the radial MLP that maps it to tensor-product weights.
//!
//! The formulas are transcribed from `bessel_basis`, `polynomial_cutoff` and
//! `MLP` in nequix (MIT) `nequix/model.py`. Its license is reproduced in
//! `THIRD_PARTY_LICENSES/nequix-LICENSE` for attribution.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod basis;
mod mlp;

pub use basis::RadialBasis;
pub use mlp::{Mlp, MlpWorkspace};
