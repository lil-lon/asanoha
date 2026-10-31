//! Channel-wise ("uvu") tensor product of node features with the spherical
//! harmonics of each edge, scaled by per-edge weights and summed over the
//! edges of each receiver, as nequix's `NequixConvolution` computes
//! `scatter_sum(tensor_product(x[senders], Y) * w, receivers) / √n̄` with
//! e3nn-jax.
//!
//! The paths and their Clebsch–Gordan coefficients are data from the
//! exporter; this crate knows no group theory.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod naive;

use asanoha_core::{Error, Features, Irreps, Layout, Real};

/// Nonzero coefficient `C[m1][m2][m3]` of a path, including e3nn's component
/// normalization `√(2l₃ + 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CgEntry {
    /// Component of the input block.
    pub m1: u8,
    /// Component of the spherical harmonics of degree `l_sh`.
    pub m2: u8,
    /// Component of the output block.
    pub m3: u8,
    /// The coefficient.
    pub c: f64,
}

/// Input block ⊗ spherical harmonics of degree `l_sh` → output block, channel
/// by channel. The two blocks have the same multiplicity `mul`, and the path
/// takes the `mul` weights of the output block's channels.
#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    /// Block of the input irreps.
    pub in_block: usize,
    /// Degree of the spherical harmonics.
    pub l_sh: u8,
    /// Block of the output irreps.
    pub out_block: usize,
    /// Nonzero coefficients.
    pub cg: Vec<CgEntry>,
}

/// Everything that defines a tensor product.
#[derive(Debug, Clone, PartialEq)]
pub struct TpDescriptor {
    /// Irreps of the node features.
    pub irreps_in: Irreps,
    /// Irreps of the output; for nequix one block per path, in the sorted
    /// (not simplified) order that `e3nn.tensor_product` emits. The per-edge
    /// weights follow its channels, and paths into the same block share them.
    pub irreps_out: Irreps,
    /// Spherical harmonics `l = 0..=sh_lmax` per edge, `(sh_lmax + 1)²`
    /// components.
    pub sh_lmax: u8,
    /// The paths.
    pub paths: Vec<Path>,
}

/// A path with its offsets resolved.
#[derive(Debug, Clone)]
struct Kernel<T> {
    mul: usize,
    /// Offset of the output block in an output row.
    out_start: usize,
    /// `2l₃ + 1`.
    out_dim: usize,
    /// Offset of the path's weights in a weight row.
    w_start: usize,
    /// Offset of the input component `m1` in a feature row, offset of
    /// `Y[l_sh][m2]` in a spherical-harmonics row, offset of the output
    /// component `m3` in an output row, and the coefficient.
    cg: Vec<(usize, usize, usize, T)>,
}

/// Tensor product ready to evaluate in `T`.
#[derive(Debug, Clone)]
pub struct TensorProduct<T> {
    irreps_in: Irreps,
    irreps_out: Irreps,
    sh_dim: usize,
    /// Weights per edge.
    n_weights: usize,
    kernels: Vec<Kernel<T>>,
    avg_n_neighbors: T,
}

impl<T: Real> TensorProduct<T> {
    /// Checks the paths against the irreps and resolves their offsets; the
    /// sum over the edges of each receiver is divided by `√avg_n_neighbors`.
    pub fn new(desc: TpDescriptor, avg_n_neighbors: f64) -> Result<Self, Error> {
        let (blocks_in, blocks_out) = (desc.irreps_in.blocks(), desc.irreps_out.blocks());
        let (slices_in, slices_out) = (desc.irreps_in.slices(), desc.irreps_out.slices());
        let channels = desc.irreps_out.channel_offsets();
        let mut kernels = Vec::with_capacity(desc.paths.len());
        for (k, p) in desc.paths.iter().enumerate() {
            let err = |msg: &str| Err(Error::Irreps(format!("path {k}: {msg}")));
            let (Some(b_in), Some(b_out)) =
                (blocks_in.get(p.in_block), blocks_out.get(p.out_block))
            else {
                return err("block out of range");
            };
            if p.l_sh > desc.sh_lmax {
                return err("degree of the spherical harmonics above sh_lmax");
            }
            if b_in.mul != b_out.mul {
                return err("input and output multiplicities differ");
            }
            let mul = b_in.mul as usize;
            let (d1, d2, d3) = (b_in.ir.dim(), 2 * p.l_sh as usize + 1, b_out.ir.dim());
            let sh_start = (p.l_sh as usize).pow(2);
            let mut cg = Vec::with_capacity(p.cg.len());
            for e in &p.cg {
                let (m1, m2, m3) = (e.m1 as usize, e.m2 as usize, e.m3 as usize);
                if m1 >= d1 || m2 >= d2 || m3 >= d3 {
                    return err("coefficient index out of range");
                }
                cg.push((
                    slices_in[p.in_block].start + m1 * mul,
                    sh_start + m2,
                    slices_out[p.out_block].start + m3 * mul,
                    T::from_f64(e.c),
                ));
            }
            kernels.push(Kernel {
                mul,
                out_start: slices_out[p.out_block].start,
                out_dim: d3,
                w_start: channels[p.out_block],
                cg,
            });
        }
        Ok(Self {
            sh_dim: (desc.sh_lmax as usize + 1).pow(2),
            n_weights: desc.irreps_out.num_irreps(),
            irreps_in: desc.irreps_in,
            irreps_out: desc.irreps_out,
            kernels,
            avg_n_neighbors: T::from_f64(avg_n_neighbors),
        })
    }

    /// Checks the shapes of the arguments of [`TensorProduct::forward`] and
    /// [`naive::forward`].
    fn check(
        &self,
        row_offsets: &[usize],
        senders: &[u32],
        x: &Features<T>,
        sh: &[T],
        w: &[T],
        out: &Features<T>,
    ) {
        let n_edges = senders.len();
        assert_eq!(row_offsets.last(), Some(&n_edges));
        assert_eq!((&x.irreps, x.layout), (&self.irreps_in, Layout::IrMul));
        assert_eq!((&out.irreps, out.layout), (&self.irreps_out, Layout::IrMul));
        assert_eq!(out.n, row_offsets.len() - 1);
        assert_eq!(sh.len(), n_edges * self.sh_dim);
        assert_eq!(w.len(), n_edges * self.n_weights);
    }

    /// Writes into `out` the message `M_i = Σ_e w_e ⊙ (x[senders[e]] ⊗ Y_e) / √n̄`
    /// of every receiver `i`, over its edges `row_offsets[i]..row_offsets[i + 1]`.
    ///
    /// `x` and `out` are in [`Layout::IrMul`] with the input and output
    /// irreps, `sh` is `[E][(sh_lmax + 1)²]` and `w` is `[E][num_irreps]` of
    /// the output irreps.
    pub fn forward(
        &self,
        row_offsets: &[usize],
        senders: &[u32],
        x: &Features<T>,
        sh: &[T],
        w: &[T],
        out: &mut Features<T>,
    ) {
        self.check(row_offsets, senders, x, sh, w, out);
        let (dim_in, dim_out) = (self.irreps_in.dim(), self.irreps_out.dim());
        let scale = T::ONE / self.avg_n_neighbors.sqrt();
        for (i, row) in out.data.chunks_exact_mut(dim_out).enumerate() {
            row.fill(T::ZERO);
            for e in row_offsets[i]..row_offsets[i + 1] {
                let j = senders[e] as usize;
                let xj = &x.data[j * dim_in..][..dim_in];
                let y = &sh[e * self.sh_dim..][..self.sh_dim];
                let we = &w[e * self.n_weights..][..self.n_weights];
                for k in &self.kernels {
                    let wk = &we[k.w_start..][..k.mul];
                    for &(x_off, y_off, o_off, c) in &k.cg {
                        let s = c * y[y_off];
                        let xin = &xj[x_off..][..k.mul];
                        let o = &mut row[o_off..][..k.mul];
                        for ((o, &wu), &xu) in o.iter_mut().zip(wk).zip(xin) {
                            *o += s * wu * xu;
                        }
                    }
                }
            }
            for v in row {
                *v *= scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asanoha_core::assert_close;
    use asanoha_core::testing::XorShift64;

    fn features(irreps: &str, n: usize, data: Vec<f64>) -> Features<f64> {
        Features {
            irreps: irreps.parse().unwrap(),
            layout: Layout::IrMul,
            n,
            data,
        }
    }

    fn entry(m1: u8, m2: u8, m3: u8, c: f64) -> CgEntry {
        CgEntry { m1, m2, m3, c }
    }

    /// `2x1o ⊗ Y_0 → 2x1o` and `2x1o ⊗ Y_1 → 2x0e` with e3nn's coefficients
    /// `C = δ` and `C = δ_{m1 m2} / √3`.
    fn vector_tp() -> TensorProduct<f64> {
        let c = 1.0 / 3f64.sqrt();
        let desc = TpDescriptor {
            irreps_in: "2x1o".parse().unwrap(),
            irreps_out: "2x0e+2x1o".parse().unwrap(),
            sh_lmax: 1,
            paths: vec![
                Path {
                    in_block: 0,
                    l_sh: 0,
                    out_block: 1,
                    cg: (0..3).map(|m| entry(m, 0, m, 1.0)).collect(),
                },
                Path {
                    in_block: 0,
                    l_sh: 1,
                    out_block: 0,
                    cg: (0..3).map(|m| entry(m, m, 0, c)).collect(),
                },
            ],
        };
        TensorProduct::new(desc, 4.0).unwrap()
    }

    #[test]
    fn known_values() {
        let tp = vector_tp();
        // Node 0 receives from nodes 1 and 0; node 1 receives nothing.
        let (row_offsets, senders) = ([0, 2, 2], [1, 0]);
        // Channels u = 0, 1 of the vectors (1, 2, 3), (0, 1, 0) at node 0 and
        // (−1, 0, 1), (2, 2, 2) at node 1, as [m][u].
        let x = features(
            "2x1o",
            2,
            vec![1.0, 0.0, 2.0, 1.0, 3.0, 0.0, -1.0, 2.0, 0.0, 2.0, 1.0, 2.0],
        );
        let sh = [1.0, 0.0, 0.0, 3.0, 2.0, 0.5, -1.0, 2.0];
        // Weights of the 0e channels, then of the 1o channels.
        let w = [1.0, 2.0, 3.0, 4.0, -1.0, 1.0, 0.5, 2.0];
        let mut out = features("2x0e+2x1o", 2, vec![0.0; 16]);
        tp.forward(&row_offsets, &senders, &x, &sh, &w, &mut out);

        // Edge 0 from node 1: 0e = w·(x·Y₁)/√3 = (1·3, 2·6)/√3,
        // 1o = w·Y₀·x = 3·(−1, 0, 1), 4·(2, 2, 2).
        // Edge 1 from node 0: 0e = (−1·4.5, 1·(−1))/√3,
        // 1o = 0.5·2·(1, 2, 3), 2·2·(0, 1, 0).
        // The sum over both edges is divided by √4.
        let c = 1.0 / 3f64.sqrt();
        let scalars = [-1.5 * c, 11.0 * c];
        let vectors = [[-2.0, 8.0], [2.0, 12.0], [6.0, 8.0]];
        let mut expected: Vec<f64> = scalars
            .iter()
            .chain(vectors.as_flattened())
            .map(|v| v / 2.0)
            .collect();
        expected.resize(16, 0.0);
        assert_close!(out.data, expected, 1e-15, 0.0);
    }

    /// Paths with random coefficients between random irreps, several of them
    /// into one output block.
    fn random_tp(rng: &mut XorShift64) -> TensorProduct<f64> {
        let irreps_in: Irreps = "3x0e+2x1o+3x2e+1x3o".parse().unwrap();
        let irreps_out: Irreps = "3x0e+2x1e+3x2o+1x3o+3x1o".parse().unwrap();
        let blocks = [
            (0, 2, 0),
            (1, 1, 1),
            (2, 3, 2),
            (3, 0, 3),
            (0, 1, 4),
            (2, 2, 0),
            (0, 3, 2),
        ];
        let paths = blocks
            .into_iter()
            .map(|(in_block, l_sh, out_block): (usize, u8, usize)| {
                let d1 = irreps_in.blocks()[in_block].ir.dim() as u64;
                let d3 = irreps_out.blocks()[out_block].ir.dim() as u64;
                let d2 = 2 * l_sh as u64 + 1;
                let cg = (0..6)
                    .map(|_| {
                        let m1 = (rng.next_u64() % d1) as u8;
                        let m2 = (rng.next_u64() % d2) as u8;
                        let m3 = (rng.next_u64() % d3) as u8;
                        entry(m1, m2, m3, rng.normal())
                    })
                    .collect();
                Path {
                    in_block,
                    l_sh,
                    out_block,
                    cg,
                }
            })
            .collect();
        let desc = TpDescriptor {
            irreps_in,
            irreps_out,
            sh_lmax: 3,
            paths,
        };
        TensorProduct::new(desc, 7.5).unwrap()
    }

    #[test]
    fn fused_matches_naive() {
        let mut rng = XorShift64::new(1);
        let tp = random_tp(&mut rng);
        let n = 9;
        // Random degrees in 0..6, so some nodes receive nothing.
        let mut row_offsets = vec![0];
        for _ in 0..n {
            row_offsets.push(row_offsets.last().unwrap() + (rng.next_u64() % 6) as usize);
        }
        let n_edges = row_offsets[n];
        let senders: Vec<u32> = (0..n_edges)
            .map(|_| (rng.next_u64() % n as u64) as u32)
            .collect();
        let mut normal = |len| (0..len).map(|_| rng.normal()).collect::<Vec<_>>();
        let x = Features {
            irreps: tp.irreps_in.clone(),
            layout: Layout::IrMul,
            n,
            data: normal(n * tp.irreps_in.dim()),
        };
        let sh = normal(n_edges * 16);
        let w = normal(n_edges * tp.n_weights);

        let mut fused = Features::zeros(tp.irreps_out.clone(), Layout::IrMul, n);
        tp.forward(&row_offsets, &senders, &x, &sh, &w, &mut fused);
        let mut oracle = fused.clone();
        naive::forward(&tp, &row_offsets, &senders, &x, &sh, &w, &mut oracle);
        assert!(fused.data.iter().any(|&v| v != 0.0));
        // About twice the largest difference measured; the outputs reach 13.
        assert_close!(fused.data, oracle.data, 4e-15, 0.0);
    }

    #[test]
    fn rejects_inconsistent_paths() {
        let desc = |in_block, l_sh, out_block, m: (u8, u8, u8)| TpDescriptor {
            irreps_in: "2x1o".parse().unwrap(),
            irreps_out: "2x0e+3x1o".parse().unwrap(),
            sh_lmax: 1,
            paths: vec![Path {
                in_block,
                l_sh,
                out_block,
                cg: vec![entry(m.0, m.1, m.2, 1.0)],
            }],
        };
        assert!(TensorProduct::<f64>::new(desc(0, 1, 0, (2, 2, 0)), 1.0).is_ok());
        for bad in [
            desc(1, 1, 0, (0, 0, 0)),
            desc(0, 1, 2, (0, 0, 0)),
            desc(0, 2, 0, (0, 0, 0)),
            desc(0, 0, 1, (0, 0, 0)),
            desc(0, 1, 0, (3, 0, 0)),
            desc(0, 1, 0, (0, 3, 0)),
            desc(0, 1, 0, (0, 0, 1)),
        ] {
            assert!(matches!(
                TensorProduct::<f64>::new(bad, 1.0),
                Err(Error::Irreps(_))
            ));
        }
    }
}
