use asanoha_core::Real;

use crate::NeighborList;
use crate::neighbor_list::Prepared;

/// Checks every pair `(i, j)` against every image `S` in
/// `[−images, images]`, in the order `i`, `j`, `S`, which is the canonical
/// order of the output.
pub(crate) fn search<T: Real>(p: &Prepared, nl: &mut NeighborList<T>) {
    let n = p.n_atoms();
    let [na, nb, nc] = p.images;
    for i in 0..n {
        for j in 0..n {
            for sa in -na..=na {
                for sb in -nb..=nb {
                    for sc in -nc..=nc {
                        let s = [sa, sb, sc];
                        if i == j && s == [0; 3] {
                            continue;
                        }
                        if let Some(e) = p.edge(i, j, s) {
                            nl.push(j, e);
                        }
                    }
                }
            }
        }
        nl.end_row();
    }
}
