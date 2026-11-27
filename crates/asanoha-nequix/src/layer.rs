use asanoha_core::{Features, Irreps, Layout, Real};
use asanoha_ops::{Gate, Linear, RmsNorm};
use asanoha_radial::{Mlp, MlpWorkspace};
use asanoha_tp::TensorProduct;

/// nequix's `NequixConvolution` with a species-independent skip connection
/// and the layer norm:
///
/// ```text
/// x₁  = linear_1(x)
/// w   = mlp(R)
/// M   = tp(x₁, Y, w)            summed over the edges of each receiver
/// out = gate(norm(linear_2(M) + skip(x)))
/// ```
///
/// The output irreps of each operation must be the input irreps of the next,
/// `skip` must map the irreps of `x` to those of `linear_2`'s output, and
/// `mlp` must have one output per channel of `tp`'s output irreps.
#[derive(Debug, Clone)]
pub struct Layer<T> {
    /// Linear map of the input features into the tensor product.
    pub linear_1: Linear<T>,
    /// Radial MLP from the radial basis to the tensor-product weights.
    pub mlp: Mlp<T>,
    /// Tensor product with the spherical harmonics.
    pub tp: TensorProduct<T>,
    /// Linear map of the messages.
    pub linear_2: Linear<T>,
    /// Skip connection from the input features.
    pub skip: Linear<T>,
    /// RMS layer norm.
    pub norm: RmsNorm<T>,
    /// Gate activation.
    pub gate: Gate<T>,
}

/// Values of a [`Layer::forward`] that [`Layer::backward`] uses.
#[derive(Debug, Clone)]
pub struct Activations<T: Real> {
    /// `x₁`.
    x1: Features<T>,
    /// Pre-activations of the MLP's hidden layers.
    mlp: MlpWorkspace<T>,
    /// `w`.
    w: Vec<T>,
    /// Input of the norm.
    y: Features<T>,
    /// Input of the gate.
    z: Features<T>,
}

impl<T: Real> Layer<T> {
    /// Writes the output features of every node into `out` and returns what
    /// [`Layer::backward`] needs.
    ///
    /// Receiver `i` sums over the edges `row_offsets[i]..row_offsets[i + 1]`.
    /// `x` and `out` are in [`Layout::IrMul`], `sh` is
    /// `[E][(sh_lmax + 1)²]` and `radial` is `[E][mlp.sizes[0]]`.
    pub fn forward(
        &self,
        row_offsets: &[usize],
        senders: &[u32],
        x: &Features<T>,
        sh: &[T],
        radial: &[T],
        out: &mut Features<T>,
    ) -> Activations<T> {
        let n_edges = senders.len();
        assert_eq!(radial.len(), n_edges * self.mlp.sizes[0]);
        let zeros = |irreps: &Irreps| Features::zeros(irreps.clone(), Layout::IrMul, x.n);
        let mut x1 = zeros(self.linear_1.irreps_out());
        self.linear_1.forward(x, &mut x1);
        let mut mlp = MlpWorkspace::default();
        let mut w = vec![T::ZERO; n_edges * self.mlp.sizes[self.mlp.sizes.len() - 1]];
        self.mlp.forward(radial, &mut mlp, &mut w);
        let mut m = zeros(self.tp.irreps_out());
        self.tp.forward(row_offsets, senders, &x1, sh, &w, &mut m);
        let mut y = zeros(self.linear_2.irreps_out());
        self.linear_2.forward(&m, &mut y);
        // `z` holds the skip connection until it holds the norm.
        let mut z = zeros(self.skip.irreps_out());
        self.skip.forward(x, &mut z);
        for (a, &b) in y.data.iter_mut().zip(&z.data) {
            *a += b;
        }
        self.norm.forward(&y, &mut z);
        self.gate.forward(&z, out);
        Activations { x1, mlp, w, y, z }
    }

    /// Adds `∂L/∂x`, `∂L/∂Y` and `∂L/∂R` to `d_x`, `d_sh` and `d_radial` from
    /// `∂L/∂out` in `d_out`, for the arguments of the [`Layer::forward`] that
    /// returned `act`. The gradients have the shapes and layouts of `x`, `sh`
    /// and `radial`.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        row_offsets: &[usize],
        senders: &[u32],
        sh: &[T],
        act: &Activations<T>,
        d_out: &Features<T>,
        d_x: &mut Features<T>,
        d_sh: &mut [T],
        d_radial: &mut [T],
    ) {
        assert_eq!(d_radial.len(), senders.len() * self.mlp.sizes[0]);
        let zeros = |irreps: &Irreps| Features::zeros(irreps.clone(), Layout::IrMul, d_out.n);
        let mut d_z = zeros(self.skip.irreps_out());
        self.gate.backward(&act.z, d_out, &mut d_z);
        let mut d_y = zeros(self.skip.irreps_out());
        self.norm.backward(&act.y, &d_z, &mut d_y);
        self.skip.backward(&d_y, d_x);
        let mut d_m = zeros(self.tp.irreps_out());
        self.linear_2.backward(&d_y, &mut d_m);
        let mut d_x1 = zeros(self.linear_1.irreps_out());
        let mut d_w = vec![T::ZERO; act.w.len()];
        self.tp.backward(
            row_offsets,
            senders,
            &act.x1,
            sh,
            &act.w,
            &d_m,
            &mut d_x1,
            d_sh,
            &mut d_w,
        );
        self.linear_1.backward(&d_x1, d_x);
        let mut d_r = vec![T::ZERO; d_radial.len()];
        self.mlp.backward(&act.mlp, &d_w, &mut d_r);
        for (a, &b) in d_radial.iter_mut().zip(&d_r) {
            *a += b;
        }
    }
}
