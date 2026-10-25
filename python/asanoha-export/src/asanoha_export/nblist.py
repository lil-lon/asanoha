"""Neighbor-list fixtures computed with matscipy."""

from importlib.metadata import version
from pathlib import Path

import numpy as np
from matscipy.neighbours import neighbour_list
from safetensors.numpy import save_file

# Fractional coordinates of the 8-atom conventional diamond cell.
DIAMOND = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ]
)


def random_cell(rng):
    """Triclinic cell with diagonal entries in [3, 6) and the others in [-1, 1)."""
    return np.diag(rng.uniform(3.0, 6.0, 3)) + (1 - np.eye(3)) * rng.uniform(-1.0, 1.0, (3, 3))


def random_positions(rng, cell, n, lo, hi):
    """`n` atoms at fractional coordinates uniform in [lo, hi) along each axis."""
    return rng.uniform(lo, hi, (n, 3)) @ cell


def cases():
    """`(name, positions, cell, pbc, cutoff)` of every fixture."""
    rng = np.random.default_rng(0)
    out = []

    # A cutoff of 6 in a cell shorter than twice the cutoff.
    cell = 5.431 * np.eye(3)
    pos = DIAMOND @ cell + rng.normal(scale=0.1, size=(8, 3))
    out.append(("si8_distorted", pos, cell, [True] * 3, 6.0))

    # One atom with its images up to three cells away.
    out.append(("self_images", np.array([[0.3, -0.2, 0.7]]), 2.0 * np.eye(3), [True] * 3, 6.0))

    cell = random_cell(rng)
    out.append(("triclinic", random_positions(rng, cell, 12, 0.0, 1.0), cell, [True] * 3, 3.5))

    cell = random_cell(rng)[[0, 2, 1]]
    out.append(("left_handed", random_positions(rng, cell, 10, 0.0, 1.0), cell, [True] * 3, 3.0))

    # The height along b is 1 while |b| ≈ 9.55.
    cell = np.array([[10.0, 0.0, 0.0], [9.5, 1.0, 0.0], [0.0, 0.0, 5.0]])
    out.append(("skewed", random_positions(rng, cell, 4, 0.0, 1.0), cell, [True] * 3, 2.5))

    cell = random_cell(rng)
    out.append(("unwrapped", random_positions(rng, cell, 10, -2.0, 3.0), cell, [True] * 3, 3.0))

    cell = random_cell(rng)
    pos = random_positions(rng, cell, 16, -1.0, 2.0)
    out.append(("slab", pos, cell, [True, True, False], 3.0))

    cell = random_cell(rng)
    pos = random_positions(rng, cell, 12, -1.0, 2.0)
    out.append(("wire", pos, cell, [False, False, True], 3.5))

    cell = random_cell(rng)
    pos = random_positions(rng, cell, 20, -0.5, 1.5)
    out.append(("isolated", pos, cell, [False] * 3, 3.0))

    # Atom 1 and its image are at exactly the cutoff from atom 0; atom 2 is
    # within it.
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 1.25]])
    out.append(("exact_cutoff", pos, 3.0 * np.eye(3), [True] * 3, 1.5))

    return out


def write(out_dir: Path):
    """Writes one `<name>.safetensors` per case into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"matscipy": version("matscipy"), "numpy": version("numpy")}
    for name, positions, cell, pbc, cutoff in cases():
        i, j, S, D = neighbour_list(
            "ijSD", positions=positions, cell=cell, pbc=np.array(pbc), cutoff=cutoff
        )
        order = np.lexsort((S[:, 2], S[:, 1], S[:, 0], j, i))
        tensors = {
            "in.positions": positions,
            "in.cell": cell,
            "in.pbc": np.array(pbc, dtype=np.int64),
            "in.cutoff": np.array(cutoff),
            "nl.i": i[order].astype(np.int64),
            "nl.j": j[order].astype(np.int64),
            "nl.S": S[order].astype(np.int64),
            "nl.D": D[order],
        }
        save_file(tensors, out_dir / f"{name}.safetensors", metadata)
