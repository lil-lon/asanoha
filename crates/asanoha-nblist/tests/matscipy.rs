//! Neighbor lists against matscipy's, from the fixtures written by
//! `asanoha-export fixtures`.

use asanoha_core::assert_close;
use asanoha_core::safetensors::SafeTensors;
use asanoha_nblist::{Algorithm, Cell, NeighborList, Structure};

const CASES: [&str; 10] = [
    "si8_distorted",
    "self_images",
    "triclinic",
    "left_handed",
    "skewed",
    "unwrapped",
    "slab",
    "wire",
    "isolated",
    "exact_cutoff",
];

#[test]
fn matches_matscipy() {
    for case in CASES {
        let path = format!(
            "{}/../../fixtures/nblist/{case}.safetensors",
            env!("CARGO_MANIFEST_DIR")
        );
        let bytes = std::fs::read(&path).unwrap();
        let st = SafeTensors::parse(&bytes).unwrap();
        let reals = |name| st.tensor(name).unwrap().to_vec::<f64>().unwrap();
        let ints = |name| st.tensor(name).unwrap().to_vec_i64().unwrap();

        let positions: Vec<[f64; 3]> = reals("in.positions").as_chunks().0.to_vec();
        let rows: [[f64; 3]; 3] = reals("in.cell").as_chunks().0.try_into().unwrap();
        let cell = Cell::from_rows(rows).unwrap();
        let p = ints("in.pbc");
        let pbc: [bool; 3] = core::array::from_fn(|a| p[a] != 0);
        let cutoff = reals("in.cutoff")[0];
        // matscipy's edges sorted by (i, j, S), the order of the CSR rows.
        let (i, j, s) = (ints("nl.i"), ints("nl.j"), ints("nl.S"));
        let expected: Vec<(u32, u32, [i32; 3])> = s
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
            .map(|(e, s)| (i[e] as u32, j[e] as u32, s.map(|x| x as i32)))
            .collect();
        let vectors = reals("nl.D");

        let s = Structure {
            positions: &positions,
            cell: Some(&cell),
            pbc,
        };
        for algo in [Algorithm::Naive, Algorithm::CellList] {
            let nl = NeighborList::build(&s, cutoff, algo).unwrap();
            let got: Vec<(u32, u32, [i32; 3])> = nl
                .receivers()
                .zip(&nl.senders)
                .zip(&nl.shifts)
                .map(|((i, &j), &s)| (i, j, s))
                .collect();
            assert_eq!(got, expected, "{case} {algo:?}");
            // A few ulp of the terms `r_j − r_i` and `S·H`, which are below 32.
            assert_close!(nl.vectors.as_flattened(), vectors, 1e-14, 0.0);
        }
    }
}
