//! Layers against nequix's `NequixConvolution` computed with e3nn-jax, from
//! the fixtures written by `asanoha-export fixtures`.

use asanoha_core::safetensors::SafeTensors;
use asanoha_core::{Features, Irreps, Layout, Real, assert_close};
use asanoha_nequix::Layer;
use asanoha_ops::{Gate, Linear, LinearPath, RmsNorm};
use asanoha_radial::Mlp;
use asanoha_tp::{CgEntry, Path, TensorProduct, TpDescriptor};

const CASES: [&str; 3] = ["scalar_in", "hidden", "scalar_out"];

fn reals(st: &SafeTensors, name: &str) -> Vec<f64> {
    st.tensor(name).unwrap().to_vec::<f64>().unwrap()
}

fn ints(st: &SafeTensors, name: &str) -> Vec<i64> {
    st.tensor(name).unwrap().to_vec_i64().unwrap()
}

fn irreps(st: &SafeTensors, key: &str) -> Irreps {
    st.metadata()[key].parse().unwrap()
}

fn to_t<T: Real>(v: &[f64]) -> Vec<T> {
    v.iter().map(|&a| T::from_f64(a)).collect()
}

/// `[N][dim]` in e3nn's layout, as features in [`Layout::IrMul`].
fn features<T: Real>(irreps: &Irreps, data: &[f64]) -> Features<T> {
    Features {
        irreps: irreps.clone(),
        layout: Layout::MulIr,
        n: data.len() / irreps.dim(),
        data: to_t(data),
    }
    .to_layout(Layout::IrMul)
}

/// The linear map with weights `{name}.{i}_{o}`.
fn linear<T: Real>(
    st: &SafeTensors,
    name: &str,
    irreps_in: &Irreps,
    irreps_out: &Irreps,
) -> Linear<T> {
    let prefix = format!("{name}.");
    let paths = st
        .names()
        .filter_map(|n| {
            let (i, o) = n.strip_prefix(&prefix)?.split_once('_')?;
            Some(LinearPath {
                in_block: i.parse().unwrap(),
                out_block: o.parse().unwrap(),
                weights: reals(st, n),
            })
        })
        .collect();
    Linear::new(irreps_in.clone(), irreps_out.clone(), paths).unwrap()
}

fn tensor_product<T: Real>(
    st: &SafeTensors,
    irreps_in: &Irreps,
    irreps_tp: &Irreps,
) -> TensorProduct<T> {
    let paths = ints(st, "tp.paths")
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&[i, l_sh, k]| {
            let (i, k) = (i as usize, k as usize);
            let l1 = irreps_in.blocks()[i].ir.l;
            let l3 = irreps_tp.blocks()[k].ir.l;
            let cg = reals(st, &format!("tp.cg.{l1}_{l_sh}_{l3}"))
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
    let desc = TpDescriptor {
        irreps_in: irreps_in.clone(),
        irreps_out: irreps_tp.clone(),
        sh_lmax: 3,
        paths,
    };
    TensorProduct::new(desc, reals(st, "in.avg_n_neighbors")[0]).unwrap()
}

fn layer<T: Real>(st: &SafeTensors) -> Layer<T> {
    let (irreps_in, irreps_tp) = (irreps(st, "irreps_in"), irreps(st, "irreps_tp"));
    let irreps_gated = irreps(st, "irreps_gated");
    let names = ["mlp.w0", "mlp.w1", "mlp.w2"];
    let mut sizes = vec![st.tensor(names[0]).unwrap().shape()[0]];
    sizes.extend(names.map(|n| st.tensor(n).unwrap().shape()[1]));
    let gamma = (0..irreps_gated.blocks().len())
        .map(|b| reals(st, &format!("norm.gamma.{b}")))
        .collect();
    let layer = Layer {
        linear_1: linear(st, "linear_1", &irreps_in, &irreps_in),
        mlp: Mlp {
            sizes,
            weights: names.map(|n| to_t(&reals(st, n))).to_vec(),
        },
        tp: tensor_product(st, &irreps_in, &irreps_tp),
        linear_2: linear(st, "linear_2", &irreps_tp, &irreps_gated),
        skip: linear(st, "skip", &irreps_in, &irreps_gated),
        norm: RmsNorm::new(irreps_gated.clone(), gamma, reals(st, "norm.eps")[0]).unwrap(),
        gate: Gate::new(irreps_gated, reals(st, "gate.c_silu")[0]).unwrap(),
    };
    assert_eq!(layer.gate.irreps_out(), &irreps(st, "irreps_out"));
    layer
}

/// A fixture's layer and its inputs.
struct Case<T: Real> {
    layer: Layer<T>,
    row_offsets: Vec<usize>,
    senders: Vec<u32>,
    x: Features<T>,
    sh: Vec<T>,
    radial: Vec<T>,
    d_out: Features<T>,
}

fn case<T: Real>(st: &SafeTensors) -> Case<T> {
    let layer = layer(st);
    let x = features(&irreps(st, "irreps_in"), &reals(st, "in.x"));
    let receivers = ints(st, "in.receivers");
    let row_offsets = (0..=x.n)
        .map(|i| receivers.partition_point(|&r| (r as usize) < i))
        .collect();
    Case {
        d_out: features(layer.gate.irreps_out(), &reals(st, "in.d_y")),
        layer,
        row_offsets,
        senders: ints(st, "in.senders").iter().map(|&j| j as u32).collect(),
        x,
        sh: to_t(&reals(st, "in.sh")),
        radial: to_t(&reals(st, "in.radial")),
    }
}

impl<T: Real> Case<T> {
    /// The output and, added to `grads`, the gradients `∂L/∂x`, `∂L/∂Y` and
    /// `∂L/∂R`.
    fn eval(&self, grads: &mut (Features<T>, Vec<T>, Vec<T>)) -> Features<T> {
        let mut out = Features::zeros(
            self.layer.gate.irreps_out().clone(),
            Layout::IrMul,
            self.x.n,
        );
        let act = self.layer.forward(
            &self.row_offsets,
            &self.senders,
            &self.x,
            &self.sh,
            &self.radial,
            &mut out,
        );
        self.layer.backward(
            &self.row_offsets,
            &self.senders,
            &self.sh,
            &act,
            &self.d_out,
            &mut grads.0,
            &mut grads.1,
            &mut grads.2,
        );
        out
    }

    fn zero_grads(&self) -> (Features<T>, Vec<T>, Vec<T>) {
        (
            Features::zeros(self.x.irreps.clone(), Layout::IrMul, self.x.n),
            vec![T::ZERO; self.sh.len()],
            vec![T::ZERO; self.radial.len()],
        )
    }
}

fn read(case: &str) -> Vec<u8> {
    let path = format!(
        "{}/../../fixtures/nequix/layer_{case}.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap()
}

/// Checks the output and the gradients of every case against e3nn-jax's, to
/// `tol` relative to the largest expected magnitude of each.
fn matches_e3nn<T: Real>(tol: f64) {
    for name in CASES {
        let bytes = read(name);
        let st = SafeTensors::parse(&bytes).unwrap();
        let c = case::<T>(&st);
        let mut grads = c.zero_grads();
        let out = c.eval(&mut grads);
        let actual = [
            out.to_layout(Layout::MulIr).data,
            grads.0.to_layout(Layout::MulIr).data,
            grads.1,
            grads.2,
        ];
        for (actual, name) in actual
            .iter()
            .zip(["out.y", "out.d_x", "out.d_sh", "out.d_radial"])
        {
            let expected = reals(&st, name);
            let max = expected.iter().fold(0.0, |m: f64, v| m.max(v.abs()));
            assert_close!(actual, expected, tol * max, 0.0);
        }
    }
}

// The tolerances are about twice the largest errors measured.

#[test]
fn matches_e3nn_f64() {
    matches_e3nn::<f64>(2.5e-15);
}

#[test]
fn matches_e3nn_f32() {
    matches_e3nn::<f32>(1.2e-6);
}

#[test]
fn backward_adds_to_the_gradients() {
    let bytes = read("hidden");
    let c = case::<f64>(&SafeTensors::parse(&bytes).unwrap());
    let mut once = c.zero_grads();
    c.eval(&mut once);
    let mut twice = once.clone();
    c.eval(&mut twice);
    let double = |v: &[f64]| v.iter().map(|a| 2.0 * a).collect::<Vec<_>>();
    // About twice the largest difference measured; the gradients reach 3.5.
    assert_close!(twice.0.data, double(&once.0.data), 5e-16, 0.0);
    assert_close!(twice.1, double(&once.1), 5e-16, 0.0);
    assert_close!(twice.2, double(&once.2), 5e-16, 0.0);
}
