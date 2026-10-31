//! Spherical harmonics and their gradients against e3nn-jax's, from the
//! fixture written by `asanoha-export fixtures`.

use asanoha_core::assert_close;
use asanoha_core::safetensors::SafeTensors;
use asanoha_sph::sh_lmax3_with_grad;

/// The fixture's vectors, `Y` and `∂Y/∂r` (from `jax.jacfwd`).
fn fixture() -> (Vec<[f64; 3]>, Vec<f64>, Vec<f64>) {
    let path = format!(
        "{}/../../fixtures/sph/lmax3.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap();
    let st = SafeTensors::parse(&bytes).unwrap();
    let reals = |name| st.tensor(name).unwrap().to_vec::<f64>().unwrap();
    let vectors = reals("in.vectors").as_chunks().0.to_vec();
    (vectors, reals("out.sh"), reals("out.grad"))
}

/// `Y` and `∂Y/∂r` of every vector, flattened as in the fixture.
fn eval<T: asanoha_core::Real>(vectors: &[[f64; 3]]) -> (Vec<T>, Vec<T>) {
    let (mut y, mut grad) = (Vec::new(), Vec::new());
    for r in vectors {
        let (v, g) = sh_lmax3_with_grad(r.map(T::from_f64));
        y.extend(v);
        grad.extend(g.as_flattened());
    }
    (y, grad)
}

// The tolerances are about twice the largest errors measured; `|∂Y/∂r|`
// reaches 9 at the shortest vectors.

#[test]
fn matches_e3nn_f64() {
    let (vectors, sh, grad) = fixture();
    let (y, g) = eval::<f64>(&vectors);
    assert_close!(y, sh, 4e-15, 0.0);
    assert_close!(g, grad, 1e-14, 0.0);
}

// Rounding the vectors to f32 is part of the error.
#[test]
fn matches_e3nn_f32() {
    let (vectors, sh, grad) = fixture();
    let (y, g) = eval::<f32>(&vectors);
    assert_close!(y, sh, 1.5e-6, 0.0);
    assert_close!(g, grad, 5e-6, 0.0);
}
