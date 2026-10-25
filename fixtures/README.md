# Fixtures

Test data written by `python/asanoha-export`, with the versions in its
`uv.lock`. Each `.safetensors` file records the producing package versions in
`__metadata__`. Regenerate everything from the repository root with

```
uv run --locked --project python/asanoha-export asanoha-export fixtures fixtures
```

## `nblist/`

Neighbor lists from `matscipy.neighbours.neighbour_list("ijSD", ...)`, read by
`crates/asanoha-nblist/tests/matscipy.rs`.

| Tensor | dtype | Shape | |
|---|---|---|---|
| `in.positions` | F64 | `[N, 3]` | Cartesian positions |
| `in.cell` | F64 | `[3, 3]` | lattice vectors as rows |
| `in.pbc` | I64 | `[3]` | periodicity, 0 or 1 |
| `in.cutoff` | F64 | `[]` | cutoff |
| `nl.i`, `nl.j` | I64 | `[E]` | receiver and sender |
| `nl.S` | I64 | `[E, 3]` | shift, `D = positions[j] − positions[i] + S·cell` |
| `nl.D` | F64 | `[E, 3]` | edge vector |

Edges are sorted by `(i, j, S)`.

| File | Case |
|---|---|
| `si8_distorted` | diamond Si, `a = 5.431`, displaced by `N(0, 0.1)`, cutoff 6 |
| `self_images` | one atom in a cubic cell of edge 2, cutoff 6 |
| `triclinic` | random triclinic cell |
| `left_handed` | random triclinic cell with negative determinant |
| `skewed` | height 1 along `b` while `|b| ≈ 9.55` |
| `unwrapped` | fractional coordinates in `[−2, 3)` |
| `slab` | `pbc = (1, 1, 0)`, atoms outside the cell |
| `wire` | `pbc = (0, 0, 1)`, atoms outside the cell |
| `isolated` | `pbc = (0, 0, 0)` |
| `exact_cutoff` | pairs at exactly the cutoff, which are excluded |
