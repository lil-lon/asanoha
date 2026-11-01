//! Reference implementation of [`TensorProduct::forward`] in e3nn-jax's
//! order of operations: the tensor product of every edge, then the weights,
//! then the sum over each receiver's edges, then the division by `√n̄`.

use asanoha_core::{Features, Real};

use crate::TensorProduct;

/// [`TensorProduct::forward`], materializing the message of every edge.
pub fn forward<T: Real>(
    tp: &TensorProduct<T>,
    row_offsets: &[usize],
    senders: &[u32],
    x: &Features<T>,
    sh: &[T],
    w: &[T],
    out: &mut Features<T>,
) {
    tp.check(row_offsets, senders, x, sh, w, out);
    let (dim_in, dim_out) = (tp.irreps_in.dim(), tp.irreps_out.dim());
    let mut messages = vec![T::ZERO; senders.len() * dim_out];
    let mut path_out = Vec::new();
    for (e, m) in messages.chunks_exact_mut(dim_out).enumerate() {
        let j = senders[e] as usize;
        let xj = &x.data[j * dim_in..][..dim_in];
        let y = &sh[e * tp.sh_dim..][..tp.sh_dim];
        let we = &w[e * tp.n_weights..][..tp.n_weights];
        for k in &tp.kernels {
            path_out.clear();
            path_out.resize(k.out_dim * k.mul, T::ZERO);
            for &(x_off, y_off, o_off, c) in &k.cg {
                for u in 0..k.mul {
                    path_out[o_off - k.out_start + u] += xj[x_off + u] * y[y_off] * c;
                }
            }
            let block = &mut m[k.out_start..][..k.out_dim * k.mul];
            for (o, t) in block
                .chunks_exact_mut(k.mul)
                .zip(path_out.chunks_exact(k.mul))
            {
                for ((o, &t), &wu) in o.iter_mut().zip(t).zip(&we[k.w_start..]) {
                    *o += t * wu;
                }
            }
        }
    }
    let sqrt = tp.avg_n_neighbors.sqrt();
    for (i, row) in out.data.chunks_exact_mut(dim_out).enumerate() {
        row.fill(T::ZERO);
        for e in row_offsets[i]..row_offsets[i + 1] {
            for (o, &v) in row.iter_mut().zip(&messages[e * dim_out..]) {
                *o += v;
            }
        }
        for v in row {
            *v /= sqrt;
        }
    }
}
