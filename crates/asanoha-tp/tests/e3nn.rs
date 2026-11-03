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
    /// `∂L/∂messages`, in e3nn's layout.
    d_messages: Vec<f64>,
    /// `∂L/∂x` in e3nn's layout, `∂L/∂Y` and `∂L/∂w`.
    grads: [Vec<f64>; 3],
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
        d_messages: reals("in.d_messages"),
        grads: ["out.d_x", "out.d_sh", "out.d_w"].map(reals),
    }
}

/// The messages of the fused and the naive forward in e3nn's layout, and the
/// gradients `∂L/∂x` in e3nn's layout, `∂L/∂Y` and `∂L/∂w`.
fn eval<T: Real>(f: &Fixture) -> ([Vec<T>; 2], [Vec<T>; 3]) {
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

    let d_out = Features {
        irreps: f.desc.irreps_out.clone(),
        layout: Layout::MulIr,
        n,
        data: reals(&f.d_messages),
    }
    .to_layout(Layout::IrMul);
    let mut d_x = Features::zeros(f.desc.irreps_in.clone(), Layout::IrMul, n);
    let (mut d_sh, mut d_w) = (vec![T::ZERO; sh.len()], vec![T::ZERO; w.len()]);
    tp.backward(
        &f.row_offsets,
        &f.senders,
        &x,
        &sh,
        &w,
        &d_out,
        &mut d_x,
        &mut d_sh,
        &mut d_w,
    );
    let d_x = d_x.to_layout(Layout::MulIr).data;
    ([fused, oracle], [d_x, d_sh, d_w])
}

/// Checks the messages and the gradients of every case against e3nn-jax's,
/// to `tol` relative to the largest expected magnitude of each.
fn matches_e3nn<T: Real>(tol: f64) {
    for case in CASES {
        let f = fixture(case);
        let (messages, grads) = eval::<T>(&f);
        let expected = std::iter::repeat_n(&f.messages, 2).chain(&f.grads);
        for (actual, expected) in messages.into_iter().chain(grads).zip(expected) {
            let max = expected.iter().fold(0.0, |m: f64, v| m.max(v.abs()));
            assert_close!(actual, *expected, tol * max, 0.0);
        }
    }
}

// The tolerances are about twice the largest errors measured.

#[test]
fn matches_e3nn_f64() {
    matches_e3nn::<f64>(1e-15);
}

#[test]
fn matches_e3nn_f32() {
    matches_e3nn::<f32>(7e-7);
}
