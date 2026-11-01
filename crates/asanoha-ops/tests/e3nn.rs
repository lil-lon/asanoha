//! Linear, gate and RMS norm against e3nn-jax's, from the fixtures written by
//! `asanoha-export fixtures`.

use asanoha_core::safetensors::SafeTensors;
use asanoha_core::{Features, Irreps, Layout, Real, assert_close};
use asanoha_ops::{Gate, Linear, LinearPath, RmsNorm};

/// The output and the input gradient of a fixture's operation, both in e3nn's
/// layout.
type Eval<T> = fn(&SafeTensors) -> [Vec<T>; 2];

fn reals(st: &SafeTensors, name: &str) -> Vec<f64> {
    st.tensor(name).unwrap().to_vec::<f64>().unwrap()
}

fn irreps(st: &SafeTensors, key: &str) -> Irreps {
    st.metadata()[key].parse().unwrap()
}

/// `[N][dim]` in e3nn's layout, as features in [`Layout::IrMul`].
fn features<T: Real>(irreps: &Irreps, data: &[f64]) -> Features<T> {
    Features {
        irreps: irreps.clone(),
        layout: Layout::MulIr,
        n: data.len() / irreps.dim(),
        data: data.iter().map(|&v| T::from_f64(v)).collect(),
    }
    .to_layout(Layout::IrMul)
}

fn zeros<T: Real>(irreps: &Irreps, like: &Features<T>) -> Features<T> {
    Features::zeros(irreps.clone(), Layout::IrMul, like.n)
}

fn linear<T: Real>(st: &SafeTensors) -> [Vec<T>; 2] {
    let (irreps_in, irreps_out) = (irreps(st, "irreps_in"), irreps(st, "irreps_out"));
    let paths = st
        .names()
        .filter_map(|name| {
            let (i, o) = name.strip_prefix("linear.")?.split_once('_')?;
            Some(LinearPath {
                in_block: i.parse().unwrap(),
                out_block: o.parse().unwrap(),
                weights: reals(st, name),
            })
        })
        .collect();
    let lin = Linear::<T>::new(irreps_in.clone(), irreps_out.clone(), paths).unwrap();
    let x = features(&irreps_in, &reals(st, "in.x"));
    let d_y = features(&irreps_out, &reals(st, "in.d_y"));
    let (mut y, mut d_x) = (zeros(&irreps_out, &x), zeros(&irreps_in, &x));
    lin.forward(&x, &mut y);
    lin.backward(&d_y, &mut d_x);
    [y, d_x].map(|f| f.to_layout(Layout::MulIr).data)
}

fn gate<T: Real>(st: &SafeTensors) -> [Vec<T>; 2] {
    let irreps_in = irreps(st, "irreps_in");
    let gate = Gate::<T>::new(irreps_in.clone(), reals(st, "gate.c_silu")[0]).unwrap();
    let irreps_out = gate.irreps_out().clone();
    assert_eq!(irreps_out, irreps(st, "irreps_out"));
    let x = features(&irreps_in, &reals(st, "in.x"));
    let d_y = features(&irreps_out, &reals(st, "in.d_y"));
    let (mut y, mut d_x) = (zeros(&irreps_out, &x), zeros(&irreps_in, &x));
    gate.forward(&x, &mut y);
    gate.backward(&x, &d_y, &mut d_x);
    [y, d_x].map(|f| f.to_layout(Layout::MulIr).data)
}

fn norm<T: Real>(st: &SafeTensors) -> [Vec<T>; 2] {
    let irreps = irreps(st, "irreps");
    let gamma = (0..irreps.blocks().len())
        .map(|b| reals(st, &format!("norm.gamma.{b}")))
        .collect();
    let norm = RmsNorm::<T>::new(irreps.clone(), gamma, reals(st, "norm.eps")[0]).unwrap();
    let x = features(&irreps, &reals(st, "in.x"));
    let d_y = features(&irreps, &reals(st, "in.d_y"));
    let (mut y, mut d_x) = (zeros(&irreps, &x), zeros(&irreps, &x));
    norm.forward(&x, &mut y);
    norm.backward(&x, &d_y, &mut d_x);
    [y, d_x].map(|f| f.to_layout(Layout::MulIr).data)
}

fn cases<T: Real>() -> [(&'static str, Eval<T>); 7] {
    [
        ("linear_hidden", linear::<T>),
        ("linear_tp", linear::<T>),
        ("linear_scalar_in", linear::<T>),
        ("gate_hidden", gate::<T>),
        ("gate_scalar_out", gate::<T>),
        ("norm_hidden", norm::<T>),
        ("norm_scalar_out", norm::<T>),
    ]
}

/// Checks the output and the input gradient of every case against e3nn-jax's,
/// to `tol` relative to the largest expected magnitude of each.
fn matches_e3nn<T: Real>(tol: f64) {
    for (case, eval) in cases::<T>() {
        let path = format!(
            "{}/../../fixtures/ops/{case}.safetensors",
            env!("CARGO_MANIFEST_DIR")
        );
        let bytes = std::fs::read(&path).unwrap();
        let st = SafeTensors::parse(&bytes).unwrap();
        let [y, d_x] = eval(&st);
        for (actual, name) in [(y, "out.y"), (d_x, "out.d_x")] {
            let expected = reals(&st, name);
            let max = expected.iter().fold(0.0, |m: f64, v| m.max(v.abs()));
            assert_close!(actual, expected, tol * max, 0.0);
        }
    }
}

// The tolerances are about twice the largest errors measured.

#[test]
fn matches_e3nn_f64() {
    matches_e3nn::<f64>(1.5e-15);
}

#[test]
fn matches_e3nn_f32() {
    matches_e3nn::<f32>(4e-7);
}
