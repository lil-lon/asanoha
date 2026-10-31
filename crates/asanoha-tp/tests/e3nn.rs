//! Tensor products against e3nn-jax's, from the fixtures written by
//! `asanoha-export fixtures`.

use asanoha_core::safetensors::SafeTensors;
use asanoha_core::{Features, Irreps, Layout, Real, assert_close};
use asanoha_tp::{CgEntry, Path, TensorProduct, TpDescriptor, naive};

const CASES: [&str; 3] = ["scalar_in", "hidden", "scalar_out"];

/// A fixture's tensor product, inputs and expected messages.
struct Fixture {
    desc: TpDescriptor,
    avg_n_neighbors: f64,
    row_offsets: Vec<usize>,
    senders: Vec<u32>,
    /// `[N][dim_in]` in e3nn's layout.
    x: Vec<f64>,
    sh: Vec<f64>,
    w: Vec<f64>,
    /// `[N][dim_out]` in e3nn's layout.
    messages: Vec<f64>,
}

fn fixture(case: &str) -> Fixture {
    let path = format!(
        "{}/../../fixtures/tp/{case}.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap();
    let st = SafeTensors::parse(&bytes).unwrap();
    let reals = |name: &str| st.tensor(name).unwrap().to_vec::<f64>().unwrap();
    let ints = |name: &str| st.tensor(name).unwrap().to_vec_i64().unwrap();
    let irreps = |key: &str| st.metadata()[key].parse::<Irreps>().unwrap();
    let (irreps_in, irreps_out) = (irreps("irreps_in"), irreps("irreps_tp"));

    let paths = ints("tp.paths")
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&[i, l_sh, k]| {
            let (i, k) = (i as usize, k as usize);
            let l1 = irreps_in.blocks()[i].ir.l;
            let l3 = irreps_out.blocks()[k].ir.l;
            let cg = reals(&format!("tp.cg.{l1}_{l_sh}_{l3}"))
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&[m1, m2, m3, c]| CgEntry {
                    m1: m1 as u8,
                    m2: m2 as u8,
                    m3: m3 as u8,
                    c,
                })
                .collect();
            Path {
                in_block: i,
                l_sh: l_sh as u8,
                out_block: k,
                cg,
            }
        })
        .collect();

    let n = st.tensor("in.x").unwrap().shape()[0];
    let receivers = ints("in.receivers");
    let row_offsets = (0..=n)
        .map(|i| receivers.partition_point(|&r| (r as usize) < i))
        .collect();
    Fixture {
        desc: TpDescriptor {
            irreps_in,
            irreps_out,
            sh_lmax: 3,
            paths,
        },
        avg_n_neighbors: reals("in.avg_n_neighbors")[0],
        row_offsets,
        senders: ints("in.senders").iter().map(|&j| j as u32).collect(),
        x: reals("in.x"),
        sh: reals("in.sh"),
        w: reals("in.w"),
        messages: reals("out.messages"),
    }
}

/// The messages of the fused and the naive forward in e3nn's layout, and the
/// largest expected magnitude.
fn eval<T: Real>(f: &Fixture) -> (Vec<T>, Vec<T>, f64) {
    let tp = TensorProduct::<T>::new(f.desc.clone(), f.avg_n_neighbors).unwrap();
    let reals = |v: &[f64]| v.iter().map(|&a| T::from_f64(a)).collect::<Vec<T>>();
    let n = f.row_offsets.len() - 1;
    let x = Features {
        irreps: f.desc.irreps_in.clone(),
        layout: Layout::MulIr,
        n,
        data: reals(&f.x),
    }
    .to_layout(Layout::IrMul);
    let (sh, w) = (reals(&f.sh), reals(&f.w));
    let mut out = Features::zeros(f.desc.irreps_out.clone(), Layout::IrMul, n);
    tp.forward(&f.row_offsets, &f.senders, &x, &sh, &w, &mut out);
    let fused = out.to_layout(Layout::MulIr).data;
    naive::forward(&tp, &f.row_offsets, &f.senders, &x, &sh, &w, &mut out);
    let oracle = out.to_layout(Layout::MulIr).data;
    let max = f.messages.iter().fold(0.0, |m: f64, v| m.max(v.abs()));
    (fused, oracle, max)
}

// The tolerances are about twice the largest errors measured, relative to the
// largest message.

#[test]
fn matches_e3nn_f64() {
    for case in CASES {
        let f = fixture(case);
        let (fused, oracle, max) = eval::<f64>(&f);
        assert_close!(oracle, f.messages, 1e-15 * max, 0.0);
        assert_close!(fused, f.messages, 1e-15 * max, 0.0);
    }
}

#[test]
fn matches_e3nn_f32() {
    for case in CASES {
        let f = fixture(case);
        let (fused, oracle, max) = eval::<f32>(&f);
        assert_close!(oracle, f.messages, 4e-7 * max, 0.0);
        assert_close!(fused, f.messages, 4e-7 * max, 0.0);
    }
}
