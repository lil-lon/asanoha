"""Fixtures of nequix's `NequixConvolution` computed with e3nn-jax."""

from pathlib import Path

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save_file

from asanoha_export.ops import EPS, N, c_silu, gated, metadata, random_linear, rms_norm
from asanoha_export.tp import AVG_N_NEIGHBORS, CASES, SH_IRREPS, graph, path_tensors, paths

# Widths of the radial MLP's input and hidden layers; nequix-mp-1 has 8, 64, 64.
MLP_SIZES = [8, 16, 16]


def mlp(ws, x):
    """nequix's `MLP(use_bias=False)`: SiLU between layers and none after the last."""
    for k, w in enumerate(ws):
        x = x @ w
        if k < len(ws) - 1:
            x = jax.nn.silu(x)
    return x


def random_layer(irreps_in, irreps_out, rng, prefix):
    """nequix's `NequixConvolution` with random weights.

    Returns `convolution(x, sh, radial, senders, receivers)`, `NequixConvolution.__call__`
    without the kernel and with a species-independent skip; its weights and constants as
    `{prefix}…` tensors; and its irreps.
    """
    table, irreps_tp = paths(irreps_in, irreps_out)
    irreps_gated = gated(irreps_out)
    linear_1, tensors = random_linear(irreps_in, irreps_in, rng, f"{prefix}linear_1")
    linear_2, weights_2 = random_linear(irreps_tp, irreps_gated, rng, f"{prefix}linear_2")
    skip, weights_skip = random_linear(irreps_in, irreps_gated, rng, f"{prefix}skip")
    sizes = MLP_SIZES + [irreps_tp.num_irreps]
    mlp_ws = [rng.normal(size=(a, b)) / np.sqrt(a) for a, b in zip(sizes, sizes[1:])]
    gamma = [rng.normal(size=mul) for mul, _ in irreps_gated]

    def convolution(x, sh, radial, senders, receivers):
        messages = e3nn.IrrepsArray(irreps_in, linear_1(x))[senders]
        messages = e3nn.tensor_product(messages, e3nn.IrrepsArray(SH_IRREPS, sh), filter_ir_out=irreps_tp)
        messages = messages * mlp(mlp_ws, radial)
        messages = e3nn.scatter_sum(messages, dst=receivers, output_size=x.shape[0]) / jnp.sqrt(AVG_N_NEIGHBORS)
        features = rms_norm(irreps_gated, gamma, linear_2(messages.array) + skip(x))
        features = e3nn.IrrepsArray(irreps_gated, features)
        return e3nn.gate(features, even_act=jax.nn.silu, odd_act=jax.nn.tanh, even_gate_act=jax.nn.silu).array

    tensors |= weights_2 | weights_skip
    tensors |= {prefix + k: v for k, v in path_tensors(irreps_in, irreps_tp, table).items()}
    tensors |= {f"{prefix}mlp.w{k}": w for k, w in enumerate(mlp_ws)}
    tensors |= {f"{prefix}norm.gamma.{b}": g for b, g in enumerate(gamma)}
    tensors |= {
        f"{prefix}norm.eps": np.array(EPS),
        f"{prefix}gate.c_silu": np.asarray(c_silu()),
        f"{prefix}in.avg_n_neighbors": np.array(AVG_N_NEIGHBORS),
    }
    irreps = {
        f"{prefix}irreps_in": irreps_in,
        f"{prefix}irreps_tp": irreps_tp,
        f"{prefix}irreps_gated": irreps_gated,
        f"{prefix}irreps_out": e3nn.gate(irreps_gated),
    }
    return convolution, tensors, irreps


def write(out_dir: Path):
    """Writes one `layer_<name>.safetensors` per case into `out_dir`."""
    jax.config.update("jax_enable_x64", True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, irreps_in, irreps_out in CASES:
        convolution, tensors, irreps = random_layer(irreps_in, irreps_out, rng, "")
        receivers, senders = graph(rng)
        x = rng.normal(size=(N, irreps_in.dim))
        vectors = rng.normal(size=(len(receivers), 3))
        sh = np.asarray(
            e3nn.spherical_harmonics(SH_IRREPS, vectors, normalize=True, normalization="component").array
        )
        radial = rng.normal(size=(len(receivers), MLP_SIZES[0]))
        y, vjp = jax.vjp(lambda x, sh, radial: convolution(x, sh, radial, senders, receivers), x, sh, radial)
        d_y = rng.normal(size=y.shape)
        d_x, d_sh, d_radial = vjp(d_y)
        tensors |= {
            "in.receivers": receivers.astype(np.int64),
            "in.senders": senders.astype(np.int64),
            "in.x": x,
            "in.sh": sh,
            "in.radial": radial,
            "out.y": np.asarray(y),
            "in.d_y": d_y,
            "out.d_x": np.asarray(d_x),
            "out.d_sh": np.asarray(d_sh),
            "out.d_radial": np.asarray(d_radial),
        }
        save_file(tensors, out_dir / f"layer_{name}.safetensors", metadata(**irreps))
