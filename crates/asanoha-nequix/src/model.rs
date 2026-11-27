use asanoha_core::{Features, Layout, Real};
use asanoha_nblist::NeighborList;
use asanoha_ops::Linear;
use asanoha_radial::RadialBasis;
use asanoha_sph::{DIM_LMAX3, sh_lmax3, sh_lmax3_with_grad};

use crate::Layer;

/// nequix's `Nequix` with species-independent skip connections:
///
/// ```text
/// h⁰   = one-hot(species)
/// hᵏ⁺¹ = layers[k](hᵏ, Y(r_e), R(|r_e|))
/// ε_i  = readout(h)_i · scale + shift + atom_energies[species_i]
/// E    = Σ_i ε_i
/// ```
///
/// The first layer's input irreps must be one `0e` channel per species, the
/// last layer's output irreps the input irreps of `readout`, and the output
/// irreps of `readout` a single `0e`. The tensor products take the spherical
/// harmonics `l ≤ 3`, and the radial MLPs take `radial.n_basis` inputs.
#[derive(Debug, Clone)]
pub struct Model<T> {
    /// Radial basis of the edge lengths.
    pub radial: RadialBasis<T>,
    /// Interaction layers.
    pub layers: Vec<Layer<T>>,
    /// Linear map of the last layer's output to the atomic energies.
    pub readout: Linear<T>,
    /// Scale of the atomic energies.
    pub scale: T,
    /// Shift of the atomic energies.
    pub shift: T,
    /// Energy of an isolated atom of each species.
    pub atom_energies: Vec<T>,
}

/// The energy and its derivatives from [`Model::evaluate`].
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation<T> {
    /// `E`.
    pub energy: T,
    /// `F_i = −∂E/∂r_i` of every atom.
    pub forces: Vec<[T; 3]>,
    /// `∂E/∂ε` for a symmetric strain `ε` of the positions and the cell,
    /// `sym(Σ_e r_e ⊗ ∂E/∂r_e)`; the stress is the virial over the volume.
    pub virial: [[T; 3]; 3],
}

impl<T: Real> Model<T> {
    /// The energy, forces and virial of atoms with the species indices
    /// `species` and the neighbor list `nl`.
    pub fn evaluate(&self, species: &[usize], nl: &NeighborList<T>) -> Evaluation<T> {
        let n = species.len();
        assert_eq!(nl.n_atoms(), n);
        let n_basis = self.radial.n_basis;
        let sh: Vec<T> = nl.vectors.iter().flat_map(|&r| sh_lmax3(r)).collect();
        let mut radial = vec![T::ZERO; nl.n_edges() * n_basis];
        self.radial.eval_batch(&nl.distances, &mut radial);

        let irreps_in = self.layers[0].linear_1.irreps_in();
        let mut x = Features::zeros(irreps_in.clone(), Layout::IrMul, n);
        for (row, &s) in x.data.chunks_exact_mut(irreps_in.dim()).zip(species) {
            row[s] = T::ONE;
        }
        let mut acts = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let mut out = Features::zeros(layer.gate.irreps_out().clone(), Layout::IrMul, n);
            acts.push(layer.forward(&nl.row_offsets, &nl.senders, &x, &sh, &radial, &mut out));
            x = out;
        }
        let mut node = Features::zeros(self.readout.irreps_out().clone(), Layout::IrMul, n);
        self.readout.forward(&x, &mut node);
        let energy = node
            .data
            .iter()
            .zip(species)
            .fold(T::ZERO, |sum, (&e, &s)| {
                sum + e * self.scale + self.shift + self.atom_energies[s]
            });

        // `∂E/∂ε_i = 1`, so the readout's output gradient is `scale` at every
        // atom.
        node.data.fill(self.scale);
        let mut d_x = Features::zeros(x.irreps.clone(), Layout::IrMul, n);
        self.readout.backward(&node, &mut d_x);
        let mut d_sh = vec![T::ZERO; sh.len()];
        let mut d_radial = vec![T::ZERO; radial.len()];
        for (layer, act) in self.layers.iter().zip(&acts).rev() {
            let mut d_in = Features::zeros(layer.linear_1.irreps_in().clone(), Layout::IrMul, n);
            layer.backward(
                &nl.row_offsets,
                &nl.senders,
                &sh,
                act,
                &d_x,
                &mut d_in,
                &mut d_sh,
                &mut d_radial,
            );
            d_x = d_in;
        }

        // `r_e = r_j − r_i + S·H` for receiver `i` and sender `j`, so `∂E/∂r_e`
        // adds to the force on `i` and subtracts from the force on `j`.
        let mut forces = vec![[T::ZERO; 3]; n];
        let mut w = [[T::ZERO; 3]; 3];
        let (mut basis, mut d_basis) = (vec![T::ZERO; n_basis], vec![T::ZERO; n_basis]);
        let grads = d_sh
            .chunks_exact(DIM_LMAX3)
            .zip(d_radial.chunks_exact(n_basis));
        for (e, (i, (d_y, d_r))) in nl.receivers().zip(grads).enumerate() {
            let (r, dist) = (nl.vectors[e], nl.distances[e]);
            let (_, y_grad) = sh_lmax3_with_grad(r);
            self.radial.eval_with_grad(dist, &mut basis, &mut d_basis);
            let d_dist = d_r
                .iter()
                .zip(&d_basis)
                .fold(T::ZERO, |s, (&a, &b)| s + a * b);
            let mut g = r.map(|c| d_dist * c / dist);
            for (&a, yg) in d_y.iter().zip(&y_grad) {
                for (gk, &ygk) in g.iter_mut().zip(yg) {
                    *gk += a * ygk;
                }
            }
            let j = nl.senders[e] as usize;
            for k in 0..3 {
                forces[i as usize][k] += g[k];
                forces[j][k] -= g[k];
                for (wk, &gm) in w[k].iter_mut().zip(&g) {
                    *wk += r[k] * gm;
                }
            }
        }
        let virial =
            core::array::from_fn(|a| core::array::from_fn(|b| (w[a][b] + w[b][a]) * T::HALF));
        Evaluation {
            energy,
            forces,
            virial,
        }
    }
}
