"""Fixtures of nequix's `Nequix` model computed with e3nn-jax."""

from importlib.metadata import version
from pathlib import Path

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np
from matscipy.neighbours import neighbour_list
from safetensors.numpy import save_file

from asanoha_export.layer import MLP_SIZES, random_layer
from asanoha_export.nblist import random_cell, random_positions
from asanoha_export.ops import metadata, random_linear
from asanoha_export.tp import HIDDEN, SH_IRREPS

N_SPECIES = 3
N_ATOMS = 6
CUTOFF = 4.0

# nequix-mp-1's cutoff exponent, scale and shift.
P = 6.0
SCALE = 0.8066479563713074
SHIFT = 0.16502578765761478

# `(input irreps, output irreps)` of the first, a hidden and the last layer.
LAYERS = [
    (e3nn.Irreps(f"{N_SPECIES}x0e"), HIDDEN),
    (HIDDEN, HIDDEN),
    (HIDDEN, HIDDEN.filter("0e")),
]


def bessel_basis(x, num_basis, r_max):
    """nequix's `bessel_basis`."""
    prefactor = 2.0 / r_max
    bessel_weights = jnp.linspace(1.0, num_basis, num_basis) * jnp.pi
    x = x[:, None]
    return prefactor * jnp.where(x == 0.0, bessel_weights / r_max, jnp.sin(bessel_weights * x / r_max) / x)


def polynomial_cutoff(x, r_max, p):
    """nequix's `polynomial_cutoff`."""
    x = x / r_max
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * jnp.power(x, p))
    out = out + (p * (p + 2.0) * jnp.power(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * jnp.power(x, p + 2.0))
    return out * jnp.where(x < 1.0, 1.0, 0.0)


def write(out_dir: Path):
    """Writes `model_triclinic.safetensors` into `out_dir`."""
    jax.config.update("jax_enable_x64", True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    layers = []
    tensors = {}
    irreps = {}
    for k, (irreps_in, irreps_out) in enumerate(LAYERS):
        convolution, layer_tensors, layer_irreps = random_layer(irreps_in, irreps_out, rng, f"layers.{k}.")
        layers.append(convolution)
        tensors |= layer_tensors
        irreps |= layer_irreps
    readout, weights = random_linear(HIDDEN.filter("0e"), e3nn.Irreps("0e"), rng, "readout")
    atom_energies = rng.normal(size=N_SPECIES)
    cell = random_cell(rng)
    positions = random_positions(rng, cell, N_ATOMS, 0.0, 1.0)
    species = np.arange(N_ATOMS) % N_SPECIES
    receivers, senders, shifts = neighbour_list("ijS", positions=positions, cell=cell, pbc=[True] * 3, cutoff=CUTOFF)

    def node_energies(r):
        """`Nequix.node_energies` with the layers above."""
        features = jax.nn.one_hot(species, N_SPECIES)
        square_r_norm = jnp.sum(r**2, axis=-1)
        r_norm = jnp.where(square_r_norm == 0.0, 0.0, jnp.sqrt(square_r_norm))
        radial = bessel_basis(r_norm, MLP_SIZES[0], CUTOFF) * polynomial_cutoff(r_norm, CUTOFF, P)[:, None]
        sh = e3nn.spherical_harmonics(SH_IRREPS, r, normalize=True, normalization="component").array
        for convolution in layers:
            features = convolution(features, sh, radial, senders, receivers)
        return readout(features)[:, 0] * SCALE + SHIFT + atom_energies[species]

    def total_energy(positions, eps):
        """The energy of `Nequix.__call__` under the symmetric strain of `eps`."""
        eps_sym = (eps + eps.T) / 2
        positions = positions + positions @ eps_sym
        strained = cell + cell @ eps_sym
        r = positions[senders] - positions[receivers] + shifts @ strained
        return jnp.sum(node_energies(r))

    energy, (d_positions, virial) = jax.value_and_grad(total_energy, argnums=(0, 1))(positions, np.zeros((3, 3)))
    tensors |= weights | {
        "atom_energies": atom_energies,
        "scale": np.array(SCALE),
        "shift": np.array(SHIFT),
        "radial.cutoff": np.array(CUTOFF),
        "radial.p": np.array(int(P)),
        "in.positions": positions,
        "in.cell": cell,
        "in.species": species.astype(np.int64),
        "out.energy": np.asarray(energy),
        "out.forces": -np.asarray(d_positions),
        "out.stress": np.asarray(virial) / abs(np.linalg.det(cell)),
    }
    save_file(tensors, out_dir / "model_triclinic.safetensors", metadata(**irreps) | {"matscipy": version("matscipy")})
