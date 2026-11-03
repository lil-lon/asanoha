use asanoha_core::{Error, Features, Irrep, Irreps, Layout, MulIrrep, Parity, Real};

/// `d silu(z) / dz = σ(z) · (1 + z · (1 − σ(z)))`.
fn silu_grad<T: Real>(z: T) -> T {
    let s = z.sigmoid();
    s * (T::ONE + z * (T::ONE - s))
}

/// e3nn-jax's `gate(x, even_act=silu, even_gate_act=silu)` with normalized
/// activations, for input irreps `Kx0e` followed by non-scalar blocks with
/// `G ≤ K` channels in total. The first `K − G` scalars become `silu(s) / c`;
/// the last `G` become gates `silu(s) / c` that multiply the non-scalar
/// channels in order and are dropped. The output irreps are `(K − G)x0e`,
/// omitted when `K = G`, followed by the non-scalar blocks.
#[derive(Debug, Clone)]
pub struct Gate<T> {
    irreps_in: Irreps,
    irreps_out: Irreps,
    /// `K`.
    n_scalars: usize,
    /// `K − G`.
    n_extra: usize,
    /// `(mul, 2l + 1)` of each non-empty non-scalar block.
    blocks: Vec<(usize, usize)>,
    /// `c = √mean(silu(x)²)` over e3nn-jax's `normalspace(1_000_001)`.
    c: T,
}

impl<T: Real> Gate<T> {
    /// Checks the input irreps; `c_silu` is the constant e3nn-jax's
    /// `normalize_function` divides `silu` by.
    pub fn new(irreps_in: Irreps, c_silu: f64) -> Result<Self, Error> {
        let err = |msg: &str| Err(Error::Irreps(format!("gate input {irreps_in}: {msg}")));
        let Some((first, rest)) = irreps_in.blocks().split_first() else {
            return err("no scalars");
        };
        if first.ir != Irrep::new(0, Parity::Even) {
            return err("the first block is not 0e");
        }
        if rest.iter().any(|b| b.ir.l == 0) {
            return err("scalars after the first block");
        }
        let n_scalars = first.mul as usize;
        let n_gates = rest.iter().map(|b| b.mul as usize).sum::<usize>();
        if n_gates > n_scalars {
            return err("fewer scalars than non-scalar channels");
        }
        let n_extra = n_scalars - n_gates;
        let extra = MulIrrep {
            mul: n_extra as u32,
            ir: first.ir,
        };
        let blocks_out = (n_extra > 0)
            .then_some(extra)
            .into_iter()
            .chain(rest.iter().copied())
            .collect();
        let blocks = rest
            .iter()
            .filter(|b| b.mul > 0)
            .map(|b| (b.mul as usize, b.ir.dim()))
            .collect();
        Ok(Self {
            irreps_out: Irreps::new(blocks_out),
            irreps_in,
            n_scalars,
            n_extra,
            blocks,
            c: T::from_f64(c_silu),
        })
    }

    /// Irreps of the output.
    pub fn irreps_out(&self) -> &Irreps {
        &self.irreps_out
    }

    fn check(&self, x: &Features<T>, out: &Features<T>) {
        assert_eq!((&x.irreps, x.layout), (&self.irreps_in, Layout::IrMul));
        assert_eq!((&out.irreps, out.layout), (&self.irreps_out, Layout::IrMul));
        assert_eq!(x.n, out.n);
    }

    /// Writes the gated features of every node into `out`.
    pub fn forward(&self, x: &Features<T>, out: &mut Features<T>) {
        self.check(x, out);
        let (dim_in, dim_out) = (self.irreps_in.dim(), self.irreps_out.dim());
        let mut gates = vec![T::ZERO; self.n_scalars - self.n_extra];
        for (xr, or) in x
            .data
            .chunks_exact(dim_in)
            .zip(out.data.chunks_exact_mut(dim_out))
        {
            let (s, v) = xr.split_at(self.n_scalars);
            let (s_extra, s_gate) = s.split_at(self.n_extra);
            let (o_extra, o_v) = or.split_at_mut(self.n_extra);
            for (o, &z) in o_extra.iter_mut().zip(s_extra) {
                *o = z.silu() / self.c;
            }
            for (g, &z) in gates.iter_mut().zip(s_gate) {
                *g = z.silu() / self.c;
            }
            let (mut start, mut channel) = (0, 0);
            for &(mul, dim) in &self.blocks {
                let g = &gates[channel..][..mul];
                let o_block = o_v[start..][..mul * dim].chunks_exact_mut(mul);
                for (om, vm) in o_block.zip(v[start..].chunks_exact(mul)) {
                    for ((o, &vu), &gu) in om.iter_mut().zip(vm).zip(g) {
                        *o = gu * vu;
                    }
                }
                start += mul * dim;
                channel += mul;
            }
        }
    }

    /// Writes `∂L/∂x` into `d_x` from `∂L/∂out` in `d_out`, at the input `x`.
    pub fn backward(&self, x: &Features<T>, d_out: &Features<T>, d_x: &mut Features<T>) {
        self.check(x, d_out);
        assert_eq!((&d_x.irreps, d_x.layout, d_x.n), (&x.irreps, x.layout, x.n));
        let (dim_in, dim_out) = (self.irreps_in.dim(), self.irreps_out.dim());
        let mut gates = vec![T::ZERO; self.n_scalars - self.n_extra];
        for ((xr, dr), gr) in x
            .data
            .chunks_exact(dim_in)
            .zip(d_out.data.chunks_exact(dim_out))
            .zip(d_x.data.chunks_exact_mut(dim_in))
        {
            let (s, v) = xr.split_at(self.n_scalars);
            let (s_extra, s_gate) = s.split_at(self.n_extra);
            let (d_extra, d_v) = dr.split_at(self.n_extra);
            let (g_s, g_v) = gr.split_at_mut(self.n_scalars);
            let (g_extra, g_gate) = g_s.split_at_mut(self.n_extra);
            for ((g, &z), &d) in g_extra.iter_mut().zip(s_extra).zip(d_extra) {
                *g = d * silu_grad(z) / self.c;
            }
            for (g, &z) in gates.iter_mut().zip(s_gate) {
                *g = z.silu() / self.c;
            }
            // `g_gate` collects `Σ_m ∂L/∂out[m][u] · x[m][u]` per gated channel.
            g_gate.fill(T::ZERO);
            let (mut start, mut channel) = (0, 0);
            for &(mul, dim) in &self.blocks {
                let span = start..start + mul * dim;
                let rows = g_v[span.clone()]
                    .chunks_exact_mut(mul)
                    .zip(d_v[span.clone()].chunks_exact(mul))
                    .zip(v[span].chunks_exact(mul));
                for ((gm, dm), vm) in rows {
                    for u in 0..mul {
                        gm[u] = dm[u] * gates[channel + u];
                        g_gate[channel + u] += dm[u] * vm[u];
                    }
                }
                start += mul * dim;
                channel += mul;
            }
            for (g, &z) in g_gate.iter_mut().zip(s_gate) {
                *g *= silu_grad(z) / self.c;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asanoha_core::assert_close;
    use asanoha_core::testing::{XorShift64, finite_difference};

    fn features(irreps: &Irreps, n: usize, data: Vec<f64>) -> Features<f64> {
        Features {
            irreps: irreps.clone(),
            layout: Layout::IrMul,
            n,
            data,
        }
    }

    fn forward(gate: &Gate<f64>, n: usize, x: &[f64]) -> Vec<f64> {
        let x = features(&gate.irreps_in, n, x.to_vec());
        let mut out = Features::zeros(gate.irreps_out.clone(), Layout::IrMul, n);
        gate.forward(&x, &mut out);
        out.data
    }

    #[test]
    fn known_values() {
        let gate = Gate::new("3x0e+1x1o".parse().unwrap(), 0.5).unwrap();
        assert_eq!(gate.irreps_out.to_string(), "2x0e+1x1o");
        let out = forward(&gate, 1, &[1.0, -1.0, 2.0, 1.0, 2.0, 3.0]);
        let g = 2.0 * 2f64.silu();
        let expected = [2.0 * 1f64.silu(), 2.0 * (-1f64).silu(), g, 2.0 * g, 3.0 * g];
        assert_close!(out, expected, 1e-15, 0.0);

        // Without non-scalars every scalar is activated.
        let gate = Gate::new("2x0e".parse().unwrap(), 0.5).unwrap();
        assert_eq!(gate.irreps_out.to_string(), "2x0e");
        let out = forward(&gate, 2, &[1.0, -1.0, 0.5, 3.0]);
        let expected = [1.0, -1.0, 0.5, 3.0].map(|z: f64| 2.0 * z.silu());
        assert_close!(out, expected, 1e-15, 0.0);

        // When every scalar is a gate, no scalars remain in the output.
        let gate = Gate::new("1x0e+1x1o".parse().unwrap(), 0.5).unwrap();
        assert_eq!(gate.irreps_out.to_string(), "1x1o");
        let out = forward(&gate, 1, &[1.0, 1.0, 2.0, 3.0]);
        let g = 2.0 * 1f64.silu();
        assert_close!(out, [g, 2.0 * g, 3.0 * g], 1e-15, 0.0);

        // Empty non-scalar blocks are kept and take no gates.
        let gate = Gate::new("2x0e+0x2e+1x1o".parse().unwrap(), 0.5).unwrap();
        assert_eq!(gate.irreps_out.to_string(), "1x0e+0x2e+1x1o");
        let out = forward(&gate, 1, &[1.0, 1.0, 1.0, 2.0, 3.0]);
        assert_close!(out, [g, g, 2.0 * g, 3.0 * g], 1e-15, 0.0);
    }

    #[test]
    fn backward_matches_finite_differences() {
        let mut rng = XorShift64::new(1);
        for irreps in ["9x0e+2x1o+1x2e+3x3o", "4x0e"] {
            let gate = Gate::new(irreps.parse().unwrap(), 0.6).unwrap();
            let n = 3;
            let mut normal = |len| (0..len).map(|_| rng.normal()).collect::<Vec<_>>();
            let x = normal(n * gate.irreps_in.dim());
            let d_out = normal(n * gate.irreps_out.dim());
            let mut d_x = Features::zeros(gate.irreps_in.clone(), Layout::IrMul, n);
            gate.backward(
                &features(&gate.irreps_in, n, x.clone()),
                &features(&gate.irreps_out, n, d_out.clone()),
                &mut d_x,
            );
            let loss = |v: &[f64]| {
                forward(&gate, n, v)
                    .iter()
                    .zip(&d_out)
                    .map(|(y, d)| y * d)
                    .sum()
            };
            assert_close!(finite_difference(loss, &x, 1e-6), d_x.data, 1e-8, 0.0);
        }
    }

    #[test]
    fn rejects_ungateable_irreps() {
        assert!(Gate::<f64>::new("3x0e+1x1o+2x2e".parse().unwrap(), 1.0).is_ok());
        for bad in ["", "1x1o", "2x0o+1x1o", "3x0e+1x1o+1x0e", "2x0e+3x1o"] {
            assert!(matches!(
                Gate::<f64>::new(bad.parse().unwrap(), 1.0),
                Err(Error::Irreps(_))
            ));
        }
    }
}
