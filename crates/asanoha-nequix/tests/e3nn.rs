//! Layers against nequix's `NequixConvolution` and models against its
//! `Nequix`, computed with e3nn-jax, from the fixtures written by
//! `asanoha-export fixtures`.

use asanoha_core::safetensors::SafeTensors;
use asanoha_core::{Features, Irreps, Layout, Real, assert_close};
use asanoha_nblist::{Algorithm, Cell, NeighborList, Structure};
use asanoha_nequix::{Layer, Model};
use asanoha_ops::{Gate, Linear, LinearPath, RmsNorm};
use asanoha_radial::{Mlp, RadialBasis};
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

/// The tensor product of `{prefix}tp.paths` and `{prefix}tp.cg.*`.
fn tensor_product<T: Real>(
    st: &SafeTensors,
    prefix: &str,
    irreps_in: &Irreps,
    irreps_tp: &Irreps,
) -> TensorProduct<T> {
    let paths = ints(st, &format!("{prefix}tp.paths"))
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&[i, l_sh, k]| {
            let (i, k) = (i as usize, k as usize);
            let l1 = irreps_in.blocks()[i].ir.l;
            let l3 = irreps_tp.blocks()[k].ir.l;
            let cg = reals(st, &format!("{prefix}tp.cg.{l1}_{l_sh}_{l3}"))
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
    let avg_n_neighbors = reals(st, &format!("{prefix}in.avg_n_neighbors"))[0];
    TensorProduct::new(desc, avg_n_neighbors).unwrap()
}

/// The layer whose tensors and irreps are named `{prefix}…`.
fn layer<T: Real>(st: &SafeTensors, prefix: &str) -> Layer<T> {
    let name = |n: &str| format!("{prefix}{n}");
    let (irreps_in, irreps_tp) = (
        irreps(st, &name("irreps_in")),
        irreps(st, &name("irreps_tp")),
    );
    let irreps_gated = irreps(st, &name("irreps_gated"));
    let names = ["mlp.w0", "mlp.w1", "mlp.w2"].map(name);
    let mut sizes = vec![st.tensor(&names[0]).unwrap().shape()[0]];
    sizes.extend(names.iter().map(|n| st.tensor(n).unwrap().shape()[1]));
    let gamma = (0..irreps_gated.blocks().len())
        .map(|b| reals(st, &name(&format!("norm.gamma.{b}"))))
        .collect();
    let eps = reals(st, &name("norm.eps"))[0];
    let layer = Layer {
        linear_1: linear(st, &name("linear_1"), &irreps_in, &irreps_in),
        mlp: Mlp {
            sizes,
            weights: names.iter().map(|n| to_t(&reals(st, n))).collect(),
        },
        tp: tensor_product(st, prefix, &irreps_in, &irreps_tp),
        linear_2: linear(st, &name("linear_2"), &irreps_tp, &irreps_gated),
        skip: linear(st, &name("skip"), &irreps_in, &irreps_gated),
        norm: RmsNorm::new(irreps_gated.clone(), gamma, eps).unwrap(),
        gate: Gate::new(irreps_gated, reals(st, &name("gate.c_silu"))[0]).unwrap(),
    };
    assert_eq!(layer.gate.irreps_out(), &irreps(st, &name("irreps_out")));
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
    let layer = layer(st, "");
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

fn read(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/../../fixtures/nequix/{name}.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap()
}

/// Panics unless `actual` is within `tol` of `expected` relative to the
/// largest expected magnitude.
fn assert_matches<T: Real>(actual: &[T], expected: &[f64], tol: f64) {
    let max = expected.iter().fold(0.0, |m: f64, v| m.max(v.abs()));
    assert_close!(actual, expected, tol * max, 0.0);
}

/// Checks the output and the gradients of every layer case against
/// e3nn-jax's, to `tol` relative to the largest expected magnitude of each.
fn layer_matches_e3nn<T: Real>(tol: f64) {
    for name in CASES {
        let bytes = read(&format!("layer_{name}"));
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
            assert_matches(actual, &reals(&st, name), tol);
        }
    }
}

// The tolerances are about twice the largest errors measured.

#[test]
fn layer_matches_e3nn_f64() {
    layer_matches_e3nn::<f64>(2.5e-15);
}

#[test]
fn layer_matches_e3nn_f32() {
    layer_matches_e3nn::<f32>(1.2e-6);
}

#[test]
fn backward_adds_to_the_gradients() {
    let bytes = read("layer_hidden");
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

/// The model whose layers are named `layers.{k}.…`.
fn model<T: Real>(st: &SafeTensors) -> Model<T> {
    let layers: Vec<Layer<T>> = (0..)
        .map(|k| format!("layers.{k}."))
        .take_while(|prefix| st.metadata().contains_key(&format!("{prefix}irreps_in")))
        .map(|prefix| layer(st, &prefix))
        .collect();
    let last = layers[layers.len() - 1].gate.irreps_out();
    let readout = linear(st, "readout", last, &"0e".parse().unwrap());
    let scalar = |name: &str| T::from_f64(reals(st, name)[0]);
    Model {
        radial: RadialBasis {
            r_max: scalar("radial.cutoff"),
            n_basis: layers[0].mlp.sizes[0],
            p: ints(st, "radial.p")[0] as i32,
        },
        layers,
        readout,
        scale: scalar("scale"),
        shift: scalar("shift"),
        atom_energies: to_t(&reals(st, "atom_energies")),
    }
}

/// Checks the energy, forces and stress against e3nn-jax's, to `tol` of each
/// relative to its largest expected magnitude, with the neighbor list built
/// by asanoha.
fn model_matches_e3nn<T: Real>(tol: [f64; 3]) {
    let bytes = read("model_triclinic");
    let st = SafeTensors::parse(&bytes).unwrap();
    let model = model::<T>(&st);
    let positions: Vec<[T; 3]> = reals(&st, "in.positions")
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| p.map(T::from_f64))
        .collect();
    let rows = reals(&st, "in.cell");
    let cell = Cell::from_rows(core::array::from_fn(|a| {
        core::array::from_fn(|b| T::from_f64(rows[3 * a + b]))
    }))
    .unwrap();
    let structure = Structure {
        positions: &positions,
        cell: Some(&cell),
        pbc: [true; 3],
    };
    let nl = NeighborList::build(&structure, model.radial.r_max, Algorithm::CellList).unwrap();
    let species: Vec<usize> = ints(&st, "in.species")
        .iter()
        .map(|&s| s as usize)
        .collect();
    let out = model.evaluate(&species, &nl);
    let stress = out.virial.as_flattened().iter().map(|&v| v / cell.volume());
    let actual = [
        vec![out.energy],
        out.forces.as_flattened().to_vec(),
        stress.collect(),
    ];
    let names = ["out.energy", "out.forces", "out.stress"];
    for ((actual, name), tol) in actual.iter().zip(names).zip(tol) {
        assert_matches(actual, &reals(&st, name), tol);
    }
}

// The tolerances are about twice the largest errors measured; the forces
// reach 0.03 and the stress 5.5e-4.

#[test]
fn model_matches_e3nn_f64() {
    model_matches_e3nn::<f64>([1.7e-15, 5e-15, 2.4e-14]);
}

#[test]
fn model_matches_e3nn_f32() {
    model_matches_e3nn::<f32>([8.5e-8, 2e-6, 9.5e-6]);
}
