use asanoha_core::{Error, Features, Irreps, Layout, Real};

/// Weights from an input block to an output block of the same irrep.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearPath {
    /// Block of the input irreps.
    pub in_block: usize,
    /// Block of the output irreps.
    pub out_block: usize,
    /// `[mul_in][mul_out]` row-major, with e3nn's path normalization `√α`
    /// folded in.
    pub weights: Vec<f64>,
}

/// A path with its offsets resolved.
#[derive(Debug, Clone)]
struct Kernel<T> {
    in_start: usize,
    out_start: usize,
    /// `2l + 1`.
    dim: usize,
    mul_in: usize,
    mul_out: usize,
    weights: Vec<T>,
}

/// e3nn-jax's `Linear` without biases: each output block is the sum over its
/// paths of the input block times the path's weights, the same for every
/// component `m`. Output blocks without paths are zero.
#[derive(Debug, Clone)]
pub struct Linear<T> {
    irreps_in: Irreps,
    irreps_out: Irreps,
    kernels: Vec<Kernel<T>>,
}

impl<T: Real> Linear<T> {
    /// Checks the paths against the irreps.
    pub fn new(
        irreps_in: Irreps,
        irreps_out: Irreps,
        paths: Vec<LinearPath>,
    ) -> Result<Self, Error> {
        let (slices_in, slices_out) = (irreps_in.slices(), irreps_out.slices());
        let mut kernels = Vec::with_capacity(paths.len());
        for (k, p) in paths.into_iter().enumerate() {
            let err = |msg: &str| Err(Error::Irreps(format!("path {k}: {msg}")));
            let (Some(b_in), Some(b_out)) = (
                irreps_in.blocks().get(p.in_block),
                irreps_out.blocks().get(p.out_block),
            ) else {
                return err("block out of range");
            };
            if b_in.ir != b_out.ir {
                return err("input and output irreps differ");
            }
            let (mul_in, mul_out) = (b_in.mul as usize, b_out.mul as usize);
            if mul_in.checked_mul(mul_out) != Some(p.weights.len()) {
                return err("weights are not [mul_in][mul_out]");
            }
            // A path from or to an empty block contributes nothing.
            if p.weights.is_empty() {
                continue;
            }
            kernels.push(Kernel {
                in_start: slices_in[p.in_block].start,
                out_start: slices_out[p.out_block].start,
                dim: b_in.ir.dim(),
                mul_in,
                mul_out,
                weights: p.weights.iter().map(|&w| T::from_f64(w)).collect(),
            });
        }
        Ok(Self {
            irreps_in,
            irreps_out,
            kernels,
        })
    }

    fn check(&self, x: &Features<T>, out: &Features<T>) {
        assert_eq!((&x.irreps, x.layout), (&self.irreps_in, Layout::IrMul));
        assert_eq!((&out.irreps, out.layout), (&self.irreps_out, Layout::IrMul));
        assert_eq!(x.n, out.n);
    }

    /// Writes `out[o][m][v] = Σ_{paths i → o} Σ_u x[i][m][u] · W[u][v]` for
    /// every node.
    pub fn forward(&self, x: &Features<T>, out: &mut Features<T>) {
        self.check(x, out);
        let (dim_in, dim_out) = (self.irreps_in.dim(), self.irreps_out.dim());
        for (xr, or) in x
            .data
            .chunks_exact(dim_in)
            .zip(out.data.chunks_exact_mut(dim_out))
        {
            or.fill(T::ZERO);
            for k in &self.kernels {
                for m in 0..k.dim {
                    let xm = &xr[k.in_start + m * k.mul_in..][..k.mul_in];
                    let om = &mut or[k.out_start + m * k.mul_out..][..k.mul_out];
                    for (&xu, wu) in xm.iter().zip(k.weights.chunks_exact(k.mul_out)) {
                        for (o, &w) in om.iter_mut().zip(wu) {
                            *o += xu * w;
                        }
                    }
                }
            }
        }
    }

    /// Adds `∂L/∂x` to `d_x` from `∂L/∂out` in `d_out`. The weights are
    /// constants.
    pub fn backward(&self, d_out: &Features<T>, d_x: &mut Features<T>) {
        self.check(d_x, d_out);
        let (dim_in, dim_out) = (self.irreps_in.dim(), self.irreps_out.dim());
        for (dr, gr) in d_out
            .data
            .chunks_exact(dim_out)
            .zip(d_x.data.chunks_exact_mut(dim_in))
        {
            for k in &self.kernels {
                for m in 0..k.dim {
                    let dm = &dr[k.out_start + m * k.mul_out..][..k.mul_out];
                    let gm = &mut gr[k.in_start + m * k.mul_in..][..k.mul_in];
                    for (g, wu) in gm.iter_mut().zip(k.weights.chunks_exact(k.mul_out)) {
                        *g += dm.iter().zip(wu).fold(T::ZERO, |s, (&d, &w)| s + d * w);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asanoha_core::assert_close;
    use asanoha_core::testing::XorShift64;

    fn features(irreps: &Irreps, n: usize, data: Vec<f64>) -> Features<f64> {
        Features {
            irreps: irreps.clone(),
            layout: Layout::IrMul,
            n,
            data,
        }
    }

    fn path(in_block: usize, out_block: usize, weights: &[f64]) -> LinearPath {
        LinearPath {
            in_block,
            out_block,
            weights: weights.to_vec(),
        }
    }

    #[test]
    fn known_values() {
        let irreps_in: Irreps = "2x0e+1x1o+1x0e".parse().unwrap();
        let irreps_out: Irreps = "1x0e+2x1o+1x2e+0x0e".parse().unwrap();
        let paths = vec![
            path(0, 0, &[2.0, 3.0]),
            path(1, 1, &[1.0, -1.0]),
            path(2, 0, &[4.0]),
            path(2, 3, &[]),
        ];
        let lin = Linear::new(irreps_in.clone(), irreps_out.clone(), paths).unwrap();
        let x = features(&irreps_in, 1, vec![1.0, 2.0, 1.0, 2.0, 3.0, 0.5]);
        let mut out = features(&irreps_out, 1, vec![7.0; 12]);
        lin.forward(&x, &mut out);
        // 0e: 2·1 + 3·2 + 4·0.5; 1o: (1, 2, 3) times 1 and −1, as [m][v];
        // 2e has no path, and 0x0e is empty.
        let expected = [
            10.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(out.data, expected);
    }

    /// All paths between blocks of equal irreps with random weights; the
    /// output `3o` has no path.
    fn random_linear(rng: &mut XorShift64) -> Linear<f64> {
        let irreps_in: Irreps = "3x0e+2x1o+2x0e+1x2e+4x1o".parse().unwrap();
        let irreps_out: Irreps = "2x0e+3x1o+1x3o+2x2e".parse().unwrap();
        let mut paths = Vec::new();
        for (i, b_in) in irreps_in.blocks().iter().enumerate() {
            for (o, b_out) in irreps_out.blocks().iter().enumerate() {
                if b_in.ir == b_out.ir {
                    let len = (b_in.mul * b_out.mul) as usize;
                    let weights: Vec<f64> = (0..len).map(|_| rng.normal()).collect();
                    paths.push(path(i, o, &weights));
                }
            }
        }
        Linear::new(irreps_in, irreps_out, paths).unwrap()
    }

    #[test]
    fn backward_is_the_transpose() {
        let mut rng = XorShift64::new(1);
        let lin = random_linear(&mut rng);
        let n = 3;
        let mut normal = |irreps: &Irreps| {
            let data = (0..n * irreps.dim()).map(|_| rng.normal()).collect();
            features(irreps, n, data)
        };
        let (x, d_out) = (normal(&lin.irreps_in), normal(&lin.irreps_out));
        let start = normal(&lin.irreps_in);

        let mut out = Features::zeros(lin.irreps_out.clone(), Layout::IrMul, n);
        lin.forward(&x, &mut out);
        let mut d_x = start.clone();
        lin.backward(&d_out, &mut d_x);
        for (g, s) in d_x.data.iter_mut().zip(&start.data) {
            *g -= s;
        }

        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>();
        assert_close!(
            dot(&out.data, &d_out.data),
            dot(&x.data, &d_x.data),
            1e-13,
            0.0
        );
        // The output 3o has no path, so its gradient reaches nothing.
        let mut d_3o = Features::zeros(lin.irreps_out.clone(), Layout::IrMul, n);
        let slice = lin.irreps_out.slices()[2].clone();
        for row in d_3o.data.chunks_exact_mut(lin.irreps_out.dim()) {
            row[slice.clone()].fill(1.0);
        }
        let mut d_x = Features::zeros(lin.irreps_in.clone(), Layout::IrMul, n);
        lin.backward(&d_3o, &mut d_x);
        assert!(d_x.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn rejects_inconsistent_paths() {
        let new = |p: LinearPath| {
            Linear::<f64>::new(
                "2x0e+1x1o".parse().unwrap(),
                "3x0e".parse().unwrap(),
                vec![p],
            )
        };
        assert!(new(path(0, 0, &[0.0; 6])).is_ok());
        for bad in [
            path(2, 0, &[0.0; 6]),
            path(0, 1, &[0.0; 6]),
            path(1, 0, &[0.0; 3]),
            path(0, 0, &[0.0; 5]),
        ] {
            assert!(matches!(new(bad), Err(Error::Irreps(_))));
        }
    }
}
