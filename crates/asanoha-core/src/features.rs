use crate::{Irreps, Real};

/// Order of the components inside one `(mul, ir)` block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// `[mul][2l+1]`, e3nn's layout; used only at the Python boundary.
    MulIr,
    /// `[2l+1][mul]`, channel innermost; the only layout kernels accept.
    IrMul,
}

/// Node features `[n][irreps.dim()]` in a given layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Features<T: Real> {
    /// Irreps of one node.
    pub irreps: Irreps,
    /// Order inside each block.
    pub layout: Layout,
    /// Number of nodes.
    pub n: usize,
    /// Row-major `[n][irreps.dim()]`.
    pub data: Vec<T>,
}

impl<T: Real> Features<T> {
    /// All-zero features for `n` nodes.
    pub fn zeros(irreps: Irreps, layout: Layout, n: usize) -> Self {
        let data = vec![T::ZERO; n * irreps.dim()];
        Self {
            irreps,
            layout,
            n,
            data,
        }
    }

    /// Block `b` of node `node`.
    pub fn block(&self, node: usize, b: usize) -> &[T] {
        let (start, len) = self.block_span(node, b);
        &self.data[start..start + len]
    }

    /// Block `b` of node `node`, mutable.
    pub fn block_mut(&mut self, node: usize, b: usize) -> &mut [T] {
        let (start, len) = self.block_span(node, b);
        &mut self.data[start..start + len]
    }

    /// Copy in `layout`, transposing every block between `[mul][2l+1]` and
    /// `[2l+1][mul]`.
    pub fn to_layout(&self, layout: Layout) -> Features<T> {
        if layout == self.layout {
            return self.clone();
        }
        let mut out = Features::zeros(self.irreps.clone(), layout, self.n);
        let dim = self.irreps.dim();
        for node in 0..self.n {
            let mut start = node * dim;
            for b in self.irreps.blocks() {
                let (mul, d) = (b.mul as usize, b.ir.dim());
                let src = &self.data[start..start + mul * d];
                let dst = &mut out.data[start..start + mul * d];
                for u in 0..mul {
                    for m in 0..d {
                        let (mul_ir, ir_mul) = (u * d + m, m * mul + u);
                        match layout {
                            Layout::IrMul => dst[ir_mul] = src[mul_ir],
                            Layout::MulIr => dst[mul_ir] = src[ir_mul],
                        }
                    }
                }
                start += mul * d;
            }
        }
        out
    }

    fn block_span(&self, node: usize, b: usize) -> (usize, usize) {
        let blocks = self.irreps.blocks();
        let offset: usize = blocks[..b].iter().map(|x| x.dim()).sum();
        (node * self.irreps.dim() + offset, blocks[b].dim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::XorShift64;

    #[test]
    fn transpose_2x1o() {
        // [[x0 y0 z0], [x1 y1 z1]] <-> [[x0 x1], [y0 y1], [z0 z1]]
        let f = Features {
            irreps: "2x1o".parse().unwrap(),
            layout: Layout::MulIr,
            n: 1,
            data: vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
        };
        let g = f.to_layout(Layout::IrMul);
        assert_eq!(g.layout, Layout::IrMul);
        assert_eq!(g.data, vec![0.0, 10.0, 1.0, 11.0, 2.0, 12.0]);
        assert_eq!(g.to_layout(Layout::MulIr), f);
    }

    #[test]
    fn round_trip_and_blocks() {
        let irreps: Irreps = "3x0e+2x1o+1x2e+2x3o".parse().unwrap();
        let n = 4;
        let mut rng = XorShift64::new(7);
        let mut f = Features::<f64>::zeros(irreps, Layout::MulIr, n);
        for x in &mut f.data {
            *x = rng.next_f64();
        }

        let g = f.to_layout(Layout::IrMul);
        assert_ne!(g.data, f.data);
        assert_eq!(g.to_layout(Layout::MulIr), f);

        // Scalars are unchanged by the transpose.
        for node in 0..n {
            assert_eq!(f.block(node, 0), g.block(node, 0));
        }
        // Block 1 (2x1o) of node 2: component m of channel u.
        let (a, b) = (f.block(2, 1), g.block(2, 1));
        for u in 0..2 {
            for m in 0..3 {
                assert_eq!(a[u * 3 + m], b[m * 2 + u]);
            }
        }
    }

    #[test]
    fn block_mut_writes_in_place() {
        let mut f = Features::<f32>::zeros("1x0e+1x1o".parse().unwrap(), Layout::IrMul, 2);
        f.block_mut(1, 1).copy_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(f.data, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    }
}
