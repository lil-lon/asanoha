"""Spherical-harmonic fixtures computed with e3nn-jax."""

from importlib.metadata import version
from itertools import product
from pathlib import Path

import e3nn_jax as e3nn
import jax
import numpy as np
from safetensors.numpy import save_file


def vectors():
    """The six axes, the eight diagonals, and 128 random vectors with lengths in [0.5, 6)."""
    rng = np.random.default_rng(0)
    axes = np.concatenate([np.eye(3), -np.eye(3)])
    diagonals = np.array(list(product([1.0, -1.0], repeat=3)))
    random = rng.normal(size=(128, 3))
    random *= rng.uniform(0.5, 6.0, (128, 1)) / np.linalg.norm(random, axis=1, keepdims=True)
    return np.concatenate([axes, diagonals, random])


def sh(r):
    """The 16 spherical harmonics with l ≤ 3, normalized input and component normalization."""
    irreps = e3nn.s2_irreps(3)
    return e3nn.spherical_harmonics(irreps, r, normalize=True, normalization="component").array


def write(out_dir: Path):
    """Writes `lmax3.safetensors` into `out_dir`."""
    jax.config.update("jax_enable_x64", True)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = vectors()
    tensors = {
        "in.vectors": r,
        "out.sh": np.asarray(jax.vmap(sh)(r)),
        "out.grad": np.asarray(jax.vmap(jax.jacfwd(sh))(r)),
    }
    metadata = {
        "e3nn-jax": version("e3nn-jax"),
        "jax": version("jax"),
        "jax_enable_x64": "true",
        "numpy": version("numpy"),
    }
    save_file(tensors, out_dir / "lmax3.safetensors", metadata)
