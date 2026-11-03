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

## `sph/`

Spherical harmonics from `e3nn_jax.spherical_harmonics("0e + 1o + 2e + 3o",
r, normalize=True, normalization="component")` with `jax_enable_x64`, read by
`crates/asanoha-sph/tests/e3nn.rs`.

| Tensor | dtype | Shape | |
|---|---|---|---|
| `in.vectors` | F64 | `[E, 3]` | the six axes, the eight diagonals, 128 random vectors with lengths in `[0.5, 6)` |
| `out.sh` | F64 | `[E, 16]` | `Y` |
| `out.grad` | F64 | `[E, 16, 3]` | `∂Y/∂r` from `jax.jacfwd` |

| File | Case |
|---|---|
| `lmax3` | `l ≤ 3` |

## `tp/`

Messages `scatter_sum(e3nn.tensor_product(x[senders], Y, filter_ir_out=…) * w,
receivers) / √n̄` as nequix's `NequixConvolution` computes them, with
`jax_enable_x64`, read by `crates/asanoha-tp/tests/e3nn.rs`. The graph has 5
nodes and 12 edges; node 3 receives none.

| Tensor | dtype | Shape | |
|---|---|---|---|
| `in.x` | F64 | `[N, dim_in]` | node features in e3nn's layout |
| `in.sh` | F64 | `[E, 16]` | spherical harmonics `l ≤ 3` of random vectors |
| `in.w` | F64 | `[E, num_irreps_tp]` | per-edge weights |
| `in.receivers`, `in.senders` | I64 | `[E]` | edges, sorted by receiver |
| `in.avg_n_neighbors` | F64 | `[]` | `n̄` |
| `tp.paths` | I64 | `[P, 3]` | `(in_block, l_sh, out_block)` in nequix's enumeration order; `out_block` indexes `irreps_tp` |
| `tp.cg.{l1}_{l2}_{l3}` | F64 | `[nnz, 4]` | `(m1, m2, m3, c)` of `clebsch_gordan(l1, l2, l3) · √(2l3 + 1)` with `|c| > 1e-12` |
| `out.messages` | F64 | `[N, dim_tp]` | messages in e3nn's layout |

`__metadata__` holds `irreps_in` and `irreps_tp`, the sorted (not simplified)
irreps of the paths' outputs.

| File | Input irreps | Output irreps |
|---|---|---|
| `scalar_in` | `6x0e` | `0e`, `1o`, `2e`, `3o` |
| `hidden` | `5x0e+4x1o+3x2e+2x3o` | `0e`, `1o`, `2e`, `3o` |
| `scalar_out` | `5x0e+4x1o+3x2e+2x3o` | `0e` |

## `ops/`

The node-wise operations of nequix's `NequixConvolution` with
`jax_enable_x64`, read by `crates/asanoha-ops/tests/e3nn.rs`. Each file has 5
nodes, a random cotangent `in.d_y` and the input gradient `out.d_x` from
`jax.vjp`.

| Tensor | dtype | Shape | |
|---|---|---|---|
| `in.x` | F64 | `[N, dim_in]` | input in e3nn's layout |
| `out.y` | F64 | `[N, dim_out]` | output in e3nn's layout |
| `in.d_y` | F64 | `[N, dim_out]` | `∂L/∂y` |
| `out.d_x` | F64 | `[N, dim_in]` | `∂L/∂x` |

`linear_*`: `e3nn.FunctionalLinear` on the regrouped input irreps with random
weights, as `e3nn.equinox.Linear(..., force_irreps_out=True)` computes it.

| Tensor | dtype | Shape | |
|---|---|---|---|
| `linear.{i}_{o}` | F64 | `[mul_in, mul_out]` | weights from block `i` of `irreps_in` to block `o` of `irreps_out`, with `√α` folded in |

`gate_*`: `e3nn.gate(x, even_act=silu, odd_act=tanh, even_gate_act=silu)`.

| Tensor | dtype | Shape | |
|---|---|---|---|
| `gate.c_silu` | F64 | `[]` | `√mean(silu(x)²)` over `normalspace(1_000_001)`, the constant `normalize_function(silu)` divides by |

`norm_*`: nequix's `RMSLayerNorm(irreps, centering=False,
std_balance_degrees=True)`, transcribed from `layer_norm.py`, with random
affine weights.

| Tensor | dtype | Shape | |
|---|---|---|---|
| `norm.gamma.{b}` | F64 | `[mul_b]` | affine weights of block `b` |
| `norm.eps` | F64 | `[]` | `ε` |

`__metadata__` holds `irreps_in` and `irreps_out` (`irreps` for `norm_*`).

| File | Input irreps | Output irreps |
|---|---|---|
| `linear_hidden` | `5x0e+4x1o+3x2e+2x3o` | the same |
| `linear_tp` | `irreps_tp` of `tp/hidden` | `14x0e+4x1o+3x2e+2x3o` |
| `linear_scalar_in` | `6x0e` | `14x0e+4x1o+3x2e+2x3o`; non-scalars have no path |
| `gate_hidden` | `14x0e+4x1o+3x2e+2x3o` | `5x0e+4x1o+3x2e+2x3o` |
| `gate_scalar_out` | `5x0e` | `5x0e` |
| `norm_hidden` | `14x0e+4x1o+3x2e+2x3o` | the same |
| `norm_scalar_out` | `5x0e` | the same |
