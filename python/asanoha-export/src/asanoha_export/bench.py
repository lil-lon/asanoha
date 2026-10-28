"""Neighbor-list build time of matscipy and vesin on the structures of the Rust
benchmark `crates/asanoha-nblist/benches/nblist.rs`."""

import statistics
import time

import numpy as np
from matscipy.neighbours import neighbour_list
from vesin import NeighborList

from asanoha_export.nblist import DIAMOND

# Lattice constant of diamond Si in Å.
A = 5.431

# Supercells of n³ conventional cells: 64 to 32768 atoms.
REPEATS = [2, 3, 4, 5, 8, 16]

CUTOFFS = [5.0, 6.0]

WARMUP = 3
RUNS = 10

# Both return the same quantities as asanoha: receiver, sender, shift, vector
# and distance of every directed edge.
QUANTITIES = "ijSDd"


def diamond(n):
    """Cartesian positions of the n³ supercell, cell by cell as in the Rust benchmark."""
    cells = np.array([[x, y, z] for x in range(n) for y in range(n) for z in range(n)])
    return ((cells[:, None, :] + DIAMOND[None, :, :]).reshape(-1, 3)) * A


def median_seconds(build):
    """Median build time in seconds, and the number of edges."""
    n_edges = len(build()[0])
    for _ in range(1, WARMUP):
        build()
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        build()
        times.append(time.perf_counter() - start)
    return statistics.median(times), n_edges


def run():
    """Prints the median ns/atom of each library, single-threaded."""
    print(f"{'atoms':>7} {'cutoff':>7} {'library':>9} {'edges':>10} {'ns/atom':>12}")
    pbc = np.array([True] * 3)
    for cutoff in CUTOFFS:
        for n in REPEATS:
            positions = diamond(n)
            cell = n * A * np.eye(3)
            vesin = NeighborList(cutoff=cutoff, full_list=True, sorted=True, n_threads=1)
            libraries = [
                (
                    "matscipy",
                    lambda: neighbour_list(
                        QUANTITIES, positions=positions, cell=cell, pbc=pbc, cutoff=cutoff
                    ),
                ),
                ("vesin", lambda: vesin.compute(positions, cell, pbc, QUANTITIES)),
            ]
            for name, build in libraries:
                seconds, n_edges = median_seconds(build)
                ns_per_atom = seconds * 1e9 / len(positions)
                print(f"{len(positions):>7} {cutoff:>7.1f} {name:>9} {n_edges:>10} {ns_per_atom:>12.1f}")
