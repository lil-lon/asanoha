"""Tensor-product fixtures computed with e3nn-jax."""

from importlib.metadata import version
from pathlib import Path

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save_file

# The irreps of nequix-mp-1's layers with smaller multiplicities.
HIDDEN = e3nn.Irreps("5x0e+4x1o+3x2e+2x3o")

# `(name, input irreps, output irreps)`: the first layer, the hidden layers
# and the last layer.
CASES = [
    ("scalar_in", e3nn.Irreps("6x0e"), HIDDEN),
    ("hidden", HIDDEN, HIDDEN),
    ("scalar_out", HIDDEN, HIDDEN.filter("0e")),
]

SH_IRREPS = e3nn.s2_irreps(3)

AVG_N_NEIGHBORS = 57.413687022442645


def paths(irreps_in, irreps_out):
    """`(in_block, l_sh, out_block)` of every path and the sorted tensor-product irreps.

    The paths are enumerated as in nequix's `NequixConvolution.__init__`;
    `out_block` indexes the sorted irreps, the order `e3nn.tensor_product`
    emits.
    """
    blocks = []
    irreps_tp = []
    for i, (mul, ir_in) in enumerate(irreps_in):
        for _, ir_sh in SH_IRREPS:
            for ir_out in ir_in * ir_sh:
                if ir_out in irreps_out:
                    blocks.append((i, ir_sh.l))
                    irreps_tp.append((mul, ir_out))
    irreps_sorted, p, _ = e3nn.Irreps(irreps_tp).sort()
    return [(i, l, p[k]) for k, (i, l) in enumerate(blocks)], irreps_sorted


def clebsch_gordan(l1, l2, l3):
    """`[m1, m2, m3, c]` of the nonzero component-normalized coefficients."""
    cg = np.asarray(e3nn.clebsch_gordan(l1, l2, l3)) * np.sqrt(2 * l3 + 1)
    m1, m2, m3 = np.nonzero(np.abs(cg) > 1e-12)
    return np.stack([m1, m2, m3, cg[m1, m2, m3]], axis=1)


def graph(rng):
    """Receiver-sorted edges of 5 nodes; node 3 receives none."""
    receivers = np.sort(rng.choice([0, 1, 2, 4], size=12))
    senders = rng.integers(0, 5, size=12)
    return receivers, senders


def messages(irreps_in, irreps_tp, x, sh, w, receivers, senders):
    """The aggregated messages of nequix's `NequixConvolution.__call__`."""
    x = e3nn.IrrepsArray(irreps_in, x)[senders]
    m = e3nn.tensor_product(x, e3nn.IrrepsArray(SH_IRREPS, sh), filter_ir_out=irreps_tp) * w
    m = e3nn.scatter_sum(m, dst=receivers, output_size=5)
    return (m / jnp.sqrt(AVG_N_NEIGHBORS)).array


def write(out_dir: Path):
    """Writes one `<name>.safetensors` per case into `out_dir`."""
    jax.config.update("jax_enable_x64", True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, irreps_in, irreps_out in CASES:
        table, irreps_tp = paths(irreps_in, irreps_out)
        receivers, senders = graph(rng)
        vectors = rng.normal(size=(len(receivers), 3))
        sh = np.asarray(
            e3nn.spherical_harmonics(SH_IRREPS, vectors, normalize=True, normalization="component").array
        )
        x = rng.normal(size=(5, irreps_in.dim))
        w = rng.normal(size=(len(receivers), irreps_tp.num_irreps))
        tensors = {
            "in.x": x,
            "in.sh": sh,
            "in.w": w,
            "in.receivers": receivers.astype(np.int64),
            "in.senders": senders.astype(np.int64),
            "in.avg_n_neighbors": np.array(AVG_N_NEIGHBORS),
            "tp.paths": np.array(table, dtype=np.int64),
            "out.messages": np.asarray(messages(irreps_in, irreps_tp, x, sh, w, receivers, senders)),
        }
        for i, l2, k in table:
            l1, l3 = irreps_in[i].ir.l, irreps_tp[k].ir.l
            tensors[f"tp.cg.{l1}_{l2}_{l3}"] = clebsch_gordan(l1, l2, l3)
        metadata = {
            "irreps_in": str(irreps_in),
            "irreps_tp": str(irreps_tp),
            "e3nn-jax": version("e3nn-jax"),
            "jax": version("jax"),
            "jax_enable_x64": "true",
            "numpy": version("numpy"),
        }
        save_file(tensors, out_dir / f"{name}.safetensors", metadata)
