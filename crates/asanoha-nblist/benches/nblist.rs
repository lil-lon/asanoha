//! Neighbor-list build time on diamond Si supercells, in ns per atom: the
//! median of 10 builds after 3 warm-up builds, with `build` and with
//! `build_into` rebuilding the same list.

use std::hint::black_box;
use std::time::Instant;

use asanoha_nblist::{Algorithm, Cell, NeighborList, Structure};

/// Lattice constant of diamond Si in Å.
const A: f64 = 5.431;

/// Fractional positions of the 8 atoms in the conventional diamond cell.
const BASIS: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
    [0.25, 0.25, 0.25],
    [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75],
    [0.75, 0.75, 0.25],
];

/// Supercells of `n³` conventional cells: 64 to 32768 atoms.
const REPEATS: [usize; 6] = [2, 3, 4, 5, 8, 16];

/// Largest atom count the naive search is timed on.
const NAIVE_MAX_ATOMS: usize = 512;

const CUTOFFS: [f64; 2] = [5.0, 6.0];

const WARMUP: usize = 3;
const RUNS: usize = 10;

fn diamond(n: usize) -> Vec<[f64; 3]> {
    let mut positions = Vec::with_capacity(8 * n * n * n);
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                for f in BASIS {
                    let c = [x as f64 + f[0], y as f64 + f[1], z as f64 + f[2]];
                    positions.push(c.map(|c| c * A));
                }
            }
        }
    }
    positions
}

/// Median time of `f` in seconds.
fn median(mut f: impl FnMut()) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let mut times: Vec<f64> = (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(f64::total_cmp);
    (times[RUNS / 2 - 1] + times[RUNS / 2]) / 2.0
}

/// Median time in seconds of `build` and of `build_into` rebuilding the same
/// list, and the number of edges.
fn time(s: &Structure<'_, f64>, cutoff: f64, algo: Algorithm) -> (f64, f64, usize) {
    let mut nl = NeighborList::build(s, cutoff, algo).unwrap();
    let build = median(|| {
        black_box(NeighborList::build(black_box(s), cutoff, algo).unwrap());
    });
    let build_into = median(|| {
        nl.build_into(black_box(s), cutoff, algo).unwrap();
        black_box(&nl);
    });
    (build, build_into, nl.n_edges())
}

fn main() {
    println!(
        "{:>7} {:>7} {:>9} {:>10} {:>12} {:>12}",
        "atoms", "cutoff", "algorithm", "edges", "build", "build_into"
    );
    for cutoff in CUTOFFS {
        for n in REPEATS {
            let positions = diamond(n);
            let cell = Cell::cubic(n as f64 * A);
            let s = Structure {
                positions: &positions,
                cell: Some(&cell),
                pbc: [true; 3],
            };
            for (name, algo) in [
                ("naive", Algorithm::Naive),
                ("cell_list", Algorithm::CellList),
            ] {
                if algo == Algorithm::Naive && positions.len() > NAIVE_MAX_ATOMS {
                    continue;
                }
                let (build, build_into, n_edges) = time(&s, cutoff, algo);
                let ns_per_atom = 1e9 / positions.len() as f64;
                println!(
                    "{:>7} {:>7.1} {:>9} {:>10} {:>12.1} {:>12.1}",
                    positions.len(),
                    cutoff,
                    name,
                    n_edges,
                    build * ns_per_atom,
                    build_into * ns_per_atom
                );
            }
        }
    }
}
