use asanoha_core::Real;

/// Multilayer perceptron without biases and with SiLU between layers and none
/// after the last, as nequix's `MLP(sizes, use_bias=False)`.
///
/// Inputs and outputs are batched over edges: `x` is `[E][sizes[0]]` and the
/// output `[E][sizes[last]]`, both row-major.
#[derive(Debug, Clone)]
pub struct Mlp<T> {
    /// Layer widths, input first.
    pub sizes: Vec<usize>,
    /// Weights of each layer, `[sizes[l]][sizes[l + 1]]` row-major.
    pub weights: Vec<Vec<T>>,
}

/// Buffers of [`Mlp::forward`]; it keeps the pre-activations of the hidden
/// layers for [`Mlp::backward`].
#[derive(Debug, Clone, Default)]
pub struct MlpWorkspace<T> {
    /// `[E][sizes[l + 1]]` before the SiLU of each hidden layer `l`.
    pre: Vec<Vec<T>>,
    /// The SiLU of the latest hidden layer.
    act: Vec<T>,
}

/// `out = x · w` for `x` `[E][n_in]`, `w` `[n_in][n_out]` and `out` `[E][n_out]`.
fn matmul<T: Real>(x: &[T], w: &[T], n_out: usize, out: &mut [T]) {
    let n_in = w.len() / n_out;
    for (xe, oe) in x.chunks_exact(n_in).zip(out.chunks_exact_mut(n_out)) {
        oe.fill(T::ZERO);
        for (&xi, wi) in xe.iter().zip(w.chunks_exact(n_out)) {
            for (o, &wio) in oe.iter_mut().zip(wi) {
                *o += xi * wio;
            }
        }
    }
}

/// `out = d · wᵀ` for `d` `[E][n_out]`, `w` `[n_in][n_out]` and `out` `[E][n_in]`.
fn matmul_t<T: Real>(d: &[T], w: &[T], n_in: usize, out: &mut [T]) {
    let n_out = w.len() / n_in;
    for (de, oe) in d.chunks_exact(n_out).zip(out.chunks_exact_mut(n_in)) {
        for (o, wi) in oe.iter_mut().zip(w.chunks_exact(n_out)) {
            *o = de.iter().zip(wi).fold(T::ZERO, |s, (&a, &b)| s + a * b);
        }
    }
}

/// `d silu(z) / dz = σ(z) · (1 + z · (1 − σ(z)))`.
fn silu_grad<T: Real>(z: T) -> T {
    let s = z.sigmoid();
    s * (T::ONE + z * (T::ONE - s))
}

impl<T: Real> Mlp<T> {
    /// Writes the output of every row of `x` into `out`.
    pub fn forward(&self, x: &[T], ws: &mut MlpWorkspace<T>, out: &mut [T]) {
        let n_edges = x.len() / self.sizes[0];
        let n_hidden = self.weights.len() - 1;
        ws.pre.resize_with(n_hidden, Vec::new);
        for l in 0..n_hidden {
            let (pre, act) = (&mut ws.pre[l], &mut ws.act);
            pre.resize(n_edges * self.sizes[l + 1], T::ZERO);
            let input = if l == 0 { x } else { act.as_slice() };
            matmul(input, &self.weights[l], self.sizes[l + 1], pre);
            act.clear();
            act.extend(pre.iter().map(|&z| z.silu()));
        }
        let input = if n_hidden == 0 { x } else { ws.act.as_slice() };
        matmul(
            input,
            &self.weights[n_hidden],
            self.sizes[n_hidden + 1],
            out,
        );
    }

    /// Writes `∂L/∂x` into `d_x` from `∂L/∂out` in `d_out`, at the input of
    /// the latest [`Mlp::forward`] with `ws`. The weights are constants.
    pub fn backward(&self, ws: &MlpWorkspace<T>, d_out: &[T], d_x: &mut [T]) {
        let mut d = d_out.to_vec();
        for l in (1..self.weights.len()).rev() {
            let mut d_in = vec![T::ZERO; ws.pre[l - 1].len()];
            matmul_t(&d, &self.weights[l], self.sizes[l], &mut d_in);
            for (g, &z) in d_in.iter_mut().zip(&ws.pre[l - 1]) {
                *g *= silu_grad(z);
            }
            d = d_in;
        }
        matmul_t(&d, &self.weights[0], self.sizes[0], d_x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asanoha_core::assert_close;
    use asanoha_core::testing::{XorShift64, finite_difference};

    /// MLP with weights `N(0, 4 / sizes[l])`, nequix's initialization with
    /// `mlp_init_scale = 4`.
    fn random_mlp(rng: &mut XorShift64, sizes: &[usize]) -> Mlp<f64> {
        let weights = sizes
            .windows(2)
            .map(|s| {
                let scale = (4.0 / s[0] as f64).sqrt();
                (0..s[0] * s[1]).map(|_| rng.normal() * scale).collect()
            })
            .collect();
        Mlp {
            sizes: sizes.to_vec(),
            weights,
        }
    }

    fn forward(mlp: &Mlp<f64>, x: &[f64]) -> Vec<f64> {
        let n_edges = x.len() / mlp.sizes[0];
        let mut out = vec![0.0; n_edges * mlp.sizes.last().unwrap()];
        mlp.forward(x, &mut MlpWorkspace::default(), &mut out);
        out
    }

    #[test]
    fn known_values() {
        // z = [1·1 + (−1)·(−1), 1·2 + (−1)·0.5] = [2, 1.5].
        let mlp = Mlp {
            sizes: vec![2, 2, 1],
            weights: vec![vec![1.0, 2.0, -1.0, 0.5], vec![0.5, -2.0]],
        };
        let expected = 0.5 * 2f64.silu() - 2.0 * 1.5f64.silu();
        assert_close!(forward(&mlp, &[1.0, -1.0]), [expected], 1e-16, 0.0);
    }

    #[test]
    fn single_layer_is_linear() {
        let mlp = Mlp {
            sizes: vec![2, 3],
            weights: vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        };
        assert_eq!(forward(&mlp, &[1.0, -1.0]), [-3.0, -3.0, -3.0]);
    }

    #[test]
    fn rows_are_independent() {
        let mut rng = XorShift64::new(1);
        let mlp = random_mlp(&mut rng, &[8, 16, 16, 12]);
        let x: Vec<f64> = (0..5 * 8).map(|_| rng.normal()).collect();
        let single: Vec<f64> = x.chunks(8).flat_map(|row| forward(&mlp, row)).collect();
        assert_eq!(forward(&mlp, &x), single);
    }

    #[test]
    fn backward_matches_finite_differences() {
        let mut rng = XorShift64::new(2);
        for sizes in [&[8, 16, 16, 12][..], &[8, 16, 12], &[8, 12]] {
            let mlp = random_mlp(&mut rng, sizes);
            let n_edges = 3;
            let x: Vec<f64> = (0..n_edges * 8).map(|_| rng.normal()).collect();
            let d_out: Vec<f64> = (0..n_edges * 12).map(|_| rng.normal()).collect();

            let mut ws = MlpWorkspace::default();
            let mut out = vec![0.0; n_edges * 12];
            mlp.forward(&x, &mut ws, &mut out);
            let mut d_x = vec![0.0; n_edges * 8];
            mlp.backward(&ws, &d_out, &mut d_x);

            let loss = |v: &[f64]| {
                forward(&mlp, v)
                    .iter()
                    .zip(&d_out)
                    .map(|(y, d)| y * d)
                    .sum()
            };
            assert_close!(finite_difference(loss, &x, 1e-6), d_x, 1e-7, 0.0);
        }
    }

    // The tolerance is about twice the largest error measured; the outputs
    // reach 14.
    #[test]
    fn f32_matches_f64() {
        let mut rng = XorShift64::new(3);
        let mlp = random_mlp(&mut rng, &[8, 64, 64, 32]);
        let mlp32 = Mlp {
            sizes: mlp.sizes.clone(),
            weights: mlp
                .weights
                .iter()
                .map(|w| w.iter().map(|&v| v as f32).collect())
                .collect(),
        };
        let x: Vec<f64> = (0..4 * 8).map(|_| rng.normal()).collect();
        let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let mut out = vec![0f32; 4 * 32];
        mlp32.forward(&x32, &mut MlpWorkspace::default(), &mut out);
        assert_close!(out, forward(&mlp, &x), 7e-6, 0.0);
    }
}
