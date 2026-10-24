use asanoha_core::Real;

use crate::neighbor_list::Prepared;
use crate::{Error, NeighborList};

/// Bound on the pair images checked, `N²·Π(2·images[a] + 1)`. It bounds the
/// time and the output of the search; the naive search is meant for small
/// structures.
pub(crate) const MAX_CANDIDATES: f64 = (1 << 27) as f64;

/// Fails with [`Error::TooMuchWork`] if `n` atoms with image radius `images`
/// exceed [`MAX_CANDIDATES`].
pub(crate) fn check_work(n: usize, images: [i32; 3]) -> Result<(), Error> {
    let per_pair: f64 = images.iter().map(|&x| f64::from(2 * x + 1)).product();
    if (n as f64) * (n as f64) * per_pair > MAX_CANDIDATES {
        return Err(Error::TooMuchWork);
    }
    Ok(())
}

/// Checks every pair `(i, j)` against every image `S` in
/// `[−images, images]`, in the order `i`, `j`, `S`, which is the canonical
/// order of the output.
pub(crate) fn search<T: Real>(p: &Prepared, nl: &mut NeighborList<T>) -> Result<(), Error> {
    let n = p.n_atoms();
    check_work(n, p.images)?;
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
    Ok(())
}
