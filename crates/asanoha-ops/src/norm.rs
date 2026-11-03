use asanoha_core::{Error, Features, Irreps, Layout, Real};

/// nequix's `RMSLayerNorm(centering=False, std_balance_degrees=True)` with
/// affine weights: `out_b[m][u] = x_b[m][u] · γ_b[u] / √(s + ε)` for every
/// block `b`, with `s = Σ_b Σ_{m,u} x_b[m][u]² / (n_blocks · d_b²)` and
/// `d_b = 2l_b + 1`.
#[derive(Debug, Clone)]
pub struct RmsNorm<T> {
    irreps: Irreps,
    /// `γ_b[u]` of each component of a feature row.
    gamma: Vec<T>,
    /// `1 / (n_blocks · d_b²)` of each component of a feature row.
    weight: Vec<T>,
    eps: T,
}

impl<T: Real> RmsNorm<T> {
    /// Checks that `gamma` has one weight per channel of each block.
    pub fn new(irreps: Irreps, gamma: Vec<Vec<f64>>, eps: f64) -> Result<Self, Error> {
        let blocks = irreps.blocks();
        if gamma.len() != blocks.len()
            || blocks
                .iter()
                .zip(&gamma)
                .any(|(b, g)| g.len() != b.mul as usize)
        {
            return Err(Error::Irreps(format!("norm weights do not match {irreps}")));
        }
        let (mut gamma_c, mut weight_c) = (Vec::new(), Vec::new());
        for (b, g) in blocks.iter().zip(&gamma) {
            let d = b.ir.dim();
            let w = T::from_f64(1.0 / (blocks.len() * d * d) as f64);
            for _ in 0..d {
                gamma_c.extend(g.iter().map(|&v| T::from_f64(v)));
                weight_c.extend((0..b.mul).map(|_| w));
            }
        }
        Ok(Self {
            irreps,
            gamma: gamma_c,
            weight: weight_c,
            eps: T::from_f64(eps),
        })
    }

    fn check(&self, x: &Features<T>, out: &Features<T>) {
        assert_eq!((&x.irreps, x.layout), (&self.irreps, Layout::IrMul));
        assert_eq!(
            (&out.irreps, out.layout, out.n),
            (&self.irreps, Layout::IrMul, x.n)
        );
    }

    /// `1 / √(s + ε)` of a feature row.
    fn inv_rms(&self, x: &[T]) -> T {
        let s = x
            .iter()
            .zip(&self.weight)
            .fold(T::ZERO, |s, (&v, &w)| s + w * v * v);
        T::ONE / (s + self.eps).sqrt()
    }

    /// Writes the normalized features of every node into `out`.
    pub fn forward(&self, x: &Features<T>, out: &mut Features<T>) {
        self.check(x, out);
        let dim = self.irreps.dim();
        for (xr, or) in x.data.chunks_exact(dim).zip(out.data.chunks_exact_mut(dim)) {
            let r = self.inv_rms(xr);
            for ((o, &v), &g) in or.iter_mut().zip(xr).zip(&self.gamma) {
                *o = v * g * r;
            }
        }
    }

    /// Writes `∂L/∂x` into `d_x` from `∂L/∂out` in `d_out`, at the input `x`:
    /// `∂L/∂x = r · γ · ∂L/∂out − r³ · w · x · Σ(∂L/∂out · γ · x)` per row,
    /// with `r = 1 / √(s + ε)` and `w` the weight of each component in `s`.
    pub fn backward(&self, x: &Features<T>, d_out: &Features<T>, d_x: &mut Features<T>) {
        self.check(x, d_out);
        self.check(x, d_x);
        let dim = self.irreps.dim();
        for ((xr, dr), gr) in x
            .data
            .chunks_exact(dim)
            .zip(d_out.data.chunks_exact(dim))
            .zip(d_x.data.chunks_exact_mut(dim))
        {
            let r = self.inv_rms(xr);
            let t = xr
                .iter()
                .zip(dr)
                .zip(&self.gamma)
                .fold(T::ZERO, |s, ((&v, &d), &g)| s + d * g * v);
            let r3t = r * r * r * t;
            for (k, g) in gr.iter_mut().enumerate() {
                *g = self.gamma[k] * r * dr[k] - r3t * self.weight[k] * xr[k];
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

    fn forward(norm: &RmsNorm<f64>, n: usize, x: &[f64]) -> Vec<f64> {
        let mut out = Features::zeros(norm.irreps.clone(), Layout::IrMul, n);
        norm.forward(&features(&norm.irreps, n, x.to_vec()), &mut out);
        out.data
    }

    #[test]
    fn known_values() {
        let irreps: Irreps = "2x0e+1x1o".parse().unwrap();
        let norm = RmsNorm::new(irreps, vec![vec![1.0, 2.0], vec![3.0]], 0.0).unwrap();
        // s = (1 + 4) / 2 + (1 + 0 + 4) / (2 · 9) = 25 / 9, so 1/√s = 3/5.
        let out = forward(&norm, 1, &[1.0, 2.0, 1.0, 0.0, 2.0]);
        assert_close!(out, [0.6, 2.4, 1.8, 0.0, 3.6], 1e-15, 0.0);
    }

    #[test]
    fn backward_matches_finite_differences() {
        let mut rng = XorShift64::new(1);
        for irreps in ["9x0e+2x1o+1x2e+3x3o", "4x0e"] {
            let irreps: Irreps = irreps.parse().unwrap();
            let gamma = irreps
                .blocks()
                .iter()
                .map(|b| (0..b.mul).map(|_| rng.normal()).collect())
                .collect();
            let norm = RmsNorm::new(irreps.clone(), gamma, 1e-12).unwrap();
            let n = 3;
            let mut normal = |len| (0..len).map(|_| rng.normal()).collect::<Vec<_>>();
            let (x, d_out) = (normal(n * irreps.dim()), normal(n * irreps.dim()));
            let mut d_x = Features::zeros(irreps.clone(), Layout::IrMul, n);
            norm.backward(
                &features(&irreps, n, x.clone()),
                &features(&irreps, n, d_out.clone()),
                &mut d_x,
            );
            let loss = |v: &[f64]| {
                forward(&norm, n, v)
                    .iter()
                    .zip(&d_out)
                    .map(|(y, d)| y * d)
                    .sum()
            };
            assert_close!(finite_difference(loss, &x, 1e-6), d_x.data, 1e-8, 0.0);
        }
    }

    #[test]
    fn rejects_mismatched_weights() {
        let new =
            |gamma: Vec<Vec<f64>>| RmsNorm::<f64>::new("2x0e+1x1o".parse().unwrap(), gamma, 0.0);
        assert!(new(vec![vec![1.0; 2], vec![1.0]]).is_ok());
        for bad in [
            vec![vec![1.0; 2]],
            vec![vec![1.0; 2], vec![1.0; 2]],
            vec![vec![1.0], vec![1.0]],
        ] {
            assert!(matches!(new(bad), Err(Error::Irreps(_))));
        }
    }
}
