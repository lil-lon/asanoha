# asanoha-nblist

Neighbor list for the asanoha crates, with no third-party dependencies.

The design follows [nblist-rs](https://github.com/lil-lon/nblist-rs) by the
same author: a cell list binned in fractional coordinates with bin counts
from the perpendicular cell heights, per-axis periodic boundary conditions,
and a naive all-pairs oracle that shares the test suite with the cell list.
