"""Linear, gate and RMS-norm fixtures computed with e3nn-jax."""

from importlib.metadata import version
from pathlib import Path

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np
from e3nn_jax._src.activation import normalspace
from safetensors.numpy import save_file

from asanoha_export.tp import HIDDEN, paths

N = 5

# nequix's `RMSLayerNorm` default.
EPS = 1e-12


def gated(irreps_out):
    """The output irreps plus one gate scalar per non-scalar channel, as in nequix."""
    gate_irreps = e3nn.Irreps(f"{irreps_out.num_irreps - irreps_out.count('0e')}x0e")
    return (irreps_out + gate_irreps).regroup()


GATED = gated(HIDDEN)
SCALAR_OUT = gated(HIDDEN.filter("0e"))

# `(name, input irreps, output irreps)`: `linear_1` of the hidden layers,
# `linear_2` from the sorted tensor-product irreps, and the skip connection
# of the first layer, whose non-scalar outputs have no path.
LINEAR_CASES = [
    ("linear_hidden", HIDDEN, HIDDEN),
    ("linear_tp", paths(HIDDEN, HIDDEN)[1], GATED),
    ("linear_scalar_in", e3nn.Irreps("6x0e"), GATED),
]

# `(name, irreps)`: the hidden layers and the last layer.
NODE_CASES = [("hidden", GATED), ("scalar_out", SCALAR_OUT)]


def metadata(**irreps):
    """The irreps as strings and the producing package versions."""
    return {
        **{k: str(v) for k, v in irreps.items()},
        "e3nn-jax": version("e3nn-jax"),
        "jax": version("jax"),
        "jax_enable_x64": "true",
        "numpy": version("numpy"),
    }


def with_vjp(f, x, rng):
    """`in.x`, `out.y`, a random cotangent `in.d_y` and `out.d_x` = `d_y · ∂y/∂x`."""
    y, vjp = jax.vjp(f, x)
    d_y = rng.normal(size=y.shape)
    (d_x,) = vjp(d_y)
    return {"in.x": x, "out.y": np.asarray(y), "in.d_y": d_y, "out.d_x": np.asarray(d_x)}


def linear(irreps_in, irreps_out, rng):
    """`e3nn.equinox.Linear(irreps_in=…, irreps_out=…, force_irreps_out=True)` with random weights.

    Like the equinox module, the linear map is built on the regrouped input
    irreps; its weights are split back by rows into one
    `linear.{i}_{o}` per block `i` of `irreps_in`, with `√α` folded in.
    """
    lin = e3nn.FunctionalLinear(irreps_in.regroup(), irreps_out)
    ws = [rng.normal(size=ins.path_shape) for ins in lin.instructions]

    def f(x):
        return jax.vmap(lambda a: lin(ws, e3nn.IrrepsArray(irreps_in, a).regroup()).array)(x)

    tensors = with_vjp(f, rng.normal(size=(N, irreps_in.dim)), rng)
    rows = {}
    for i, (mul, ir) in enumerate(irreps_in):
        start = rows.get(ir, 0)
        rows[ir] = start + mul
        for ins, w in zip(lin.instructions, ws):
            if lin.irreps_in[ins.i_in].ir == ir:
                tensors[f"linear.{i}_{ins.i_out}"] = ins.path_weight * w[start : start + mul]
    return tensors


def c_silu():
    """The constant `normalize_function(silu)` divides by."""
    x = normalspace(1_000_001)
    return jnp.mean(jax.nn.silu(x) ** 2) ** 0.5


def gate(irreps, rng):
    """nequix's `e3nn.gate(…, even_act=silu, odd_act=tanh, even_gate_act=silu)`."""

    def f(x):
        x = e3nn.IrrepsArray(irreps, x)
        return e3nn.gate(x, even_act=jax.nn.silu, odd_act=jax.nn.tanh, even_gate_act=jax.nn.silu).array

    tensors = with_vjp(f, rng.normal(size=(N, irreps.dim)), rng)
    tensors["gate.c_silu"] = np.asarray(c_silu())
    return tensors


def rms_norm(irreps, gamma, x):
    """nequix's `RMSLayerNorm(irreps, centering=False, std_balance_degrees=True)` with affine weights `gamma`.

    Transcribed from `RMSLayerNorm.__call__` in nequix's `layer_norm.py`;
    the chunks are `[..., mul, 2l + 1]`.
    """
    x = e3nn.IrrepsArray(irreps, x)
    norms = []
    for irr, chunk in zip(irreps, x.chunks):
        weight = 1 / (irr.ir.dim * len(irreps))
        norm = (chunk**2 * weight).sum(axis=-2, keepdims=True)
        norm = norm.mean(axis=-1, keepdims=True)
        norms.append(norm)
    norm = jnp.concatenate(norms, axis=-1).sum(axis=-1, keepdims=True)
    norm = jnp.pow(norm + EPS, -0.5)
    chunks = [chunk * g[:, None] * norm for chunk, g in zip(x.chunks, gamma)]
    return e3nn.from_chunks(irreps, chunks, x.shape[:-1]).array


def norm(irreps, rng):
    """`rms_norm` with random affine weights `norm.gamma.{b}` per block `b`."""
    gamma = [rng.normal(size=mul) for mul, _ in irreps]
    tensors = with_vjp(lambda x: rms_norm(irreps, gamma, x), rng.normal(size=(N, irreps.dim)), rng)
    tensors |= {f"norm.gamma.{b}": g for b, g in enumerate(gamma)}
    tensors["norm.eps"] = np.array(EPS)
    return tensors


def write(out_dir: Path):
    """Writes one `<name>.safetensors` per case into `out_dir`."""
    jax.config.update("jax_enable_x64", True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, irreps_in, irreps_out in LINEAR_CASES:
        tensors = linear(irreps_in, irreps_out, rng)
        save_file(tensors, out_dir / f"{name}.safetensors", metadata(irreps_in=irreps_in, irreps_out=irreps_out))
    for name, irreps in NODE_CASES:
        tensors = gate(irreps, rng)
        irreps_out = e3nn.gate(irreps)
        save_file(tensors, out_dir / f"gate_{name}.safetensors", metadata(irreps_in=irreps, irreps_out=irreps_out))
        save_file(norm(irreps, rng), out_dir / f"norm_{name}.safetensors", metadata(irreps=irreps))
