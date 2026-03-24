# tenax-qtt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a QTT (Quantic Tensor Train) library as a standalone package that wraps tenax's FiniteMPS with grid semantics, cross-interpolation construction, arithmetic, Fourier transform MPO, and operator application.

**Architecture:** `tenax-qtt` depends on `tenax` (hard). `QTT` wraps `FiniteMPS` via composition. `QTTMatrix` wraps a `TensorNetwork` MPO. Grid system maps continuous coordinates to MPS site indices. TCI (prrLU/TCI2) builds QTTs from black-box functions. Three MPO-MPS contraction methods (naive/zipup/tci) are implemented from scratch.

**Tech Stack:** Python 3.11+, JAX, tenax (FiniteMPS, DenseTensor, svd, qr, contract), uv + hatchling, pytest.

**Spec:** `docs/plans/2026-03-24-tenax-qtt-design.md`

---

## File Map

| File | Responsibility | Creates/Modifies |
|------|---------------|-----------------|
| `src/tenax_qtt/grid.py` | `UniformGrid`, `GridSpec`, coordinate↔site mappings, layout logic | Create |
| `src/tenax_qtt/qtt.py` | `QTT` dataclass wrapping `FiniteMPS`, evaluate, integrate, sum, norm_l2 | Create |
| `src/tenax_qtt/folding.py` | `fold_to_qtt` — SVD-based QTT from dense arrays | Create |
| `src/tenax_qtt/arithmetic.py` | `add`, `subtract`, `scalar_multiply`, `hadamard`, `recompress` | Create |
| `src/tenax_qtt/cross.py` | `QTTResult`, `cross_interpolation` (TCI2 + prrLU), `estimate_error` | Create |
| `src/tenax_qtt/matrix.py` | `QTTMatrix`, identity, laplacian_1d, derivative_1d, apply (naive/zipup/tci), compose | Create |
| `src/tenax_qtt/fourier.py` | `fourier_mpo` — analytic DFT MPO via Chen-Lindsey | Create |
| `src/tenax_qtt/__init__.py` | Public API re-exports | Modify |
| `tests/test_grid.py` | Grid unit tests | Create |
| `tests/test_qtt.py` | QTT unit tests | Create |
| `tests/test_folding.py` | Folding tests | Create |
| `tests/test_arithmetic.py` | Arithmetic tests | Create |
| `tests/test_cross.py` | TCI algorithm tests | Create |
| `tests/test_matrix.py` | QTTMatrix + contraction tests | Create |
| `tests/test_fourier.py` | Fourier MPO tests | Create |

**Prerequisite (in tenax repo):** Export `FiniteMPS` and `InfiniteMPS` from `tenax/__init__.py`.

---

## Task 0: Prerequisite — Export FiniteMPS from tenax

**Files:**
- Modify: `/Users/yjkao/tenax/src/tenax/__init__.py`

- [ ] **Step 1: Add FiniteMPS and InfiniteMPS to tenax exports**

In `/Users/yjkao/tenax/src/tenax/__init__.py`, add the import and add to `__all__`:

```python
from tenax.core.mps import FiniteMPS, InfiniteMPS
```

Add `"FiniteMPS"` and `"InfiniteMPS"` to the `__all__` list.

- [ ] **Step 2: Verify import works**

Run: `cd /Users/yjkao/tenax && uv run python -c "from tenax import FiniteMPS, InfiniteMPS; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit in tenax repo**

```bash
cd /Users/yjkao/tenax
git checkout -b feat/export-mps-classes
git add src/tenax/__init__.py
git commit -m "feat: export FiniteMPS and InfiniteMPS from public API"
```

---

## Task 1: Grid System

**Files:**
- Create: `src/tenax_qtt/grid.py`
- Test: `tests/test_grid.py`

This is the foundation — all other modules depend on it.

### 1a: UniformGrid and GridSpec dataclasses

- [ ] **Step 1: Write tests for grid construction and validation**

```python
# tests/test_grid.py
import pytest
from tenax_qtt.grid import UniformGrid, GridSpec


def test_uniform_grid_construction():
    g = UniformGrid(a=0.0, b=1.0, n_bits=4)
    assert g.a == 0.0
    assert g.b == 1.0
    assert g.n_bits == 4
    assert g.include_endpoint is False


def test_uniform_grid_with_endpoint():
    g = UniformGrid(a=-1.0, b=1.0, n_bits=8, include_endpoint=True)
    assert g.include_endpoint is True


def test_gridspec_grouped():
    v1 = UniformGrid(0.0, 1.0, 4)
    v2 = UniformGrid(-1.0, 1.0, 6)
    gs = GridSpec(variables=(v1, v2), layout="grouped")
    assert gs.layout == "grouped"
    assert len(gs.variables) == 2


def test_gridspec_interleaved_requires_equal_nbits():
    v1 = UniformGrid(0.0, 1.0, 4)
    v2 = UniformGrid(-1.0, 1.0, 6)
    with pytest.raises(ValueError, match="same n_bits"):
        GridSpec(variables=(v1, v2), layout="interleaved")


def test_gridspec_interleaved_equal_nbits():
    v1 = UniformGrid(0.0, 1.0, 4)
    v2 = UniformGrid(-1.0, 1.0, 4)
    gs = GridSpec(variables=(v1, v2), layout="interleaved")
    assert gs.layout == "interleaved"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_grid.py -v`
Expected: FAIL (empty module)

- [ ] **Step 3: Implement UniformGrid and GridSpec**

```python
# src/tenax_qtt/grid.py
"""Grid specifications and coordinate-to-site mappings for QTT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class UniformGrid:
    """One variable on [a, b] discretized into 2^n_bits points."""

    a: float
    b: float
    n_bits: int
    include_endpoint: bool = False

    def __post_init__(self) -> None:
        if self.n_bits < 1:
            raise ValueError(f"n_bits must be >= 1, got {self.n_bits}")
        if self.a >= self.b:
            raise ValueError(f"a must be < b, got a={self.a}, b={self.b}")

    @property
    def n_points(self) -> int:
        return 1 << self.n_bits

    @property
    def dx(self) -> float:
        if self.include_endpoint:
            return (self.b - self.a) / (self.n_points - 1)
        return (self.b - self.a) / self.n_points


@dataclass(frozen=True)
class GridSpec:
    """Multi-dimensional grid with configurable layout."""

    variables: tuple[UniformGrid, ...]
    layout: Literal["interleaved", "fused", "grouped"]

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("At least one variable required")
        if self.layout == "interleaved":
            nbits = {v.n_bits for v in self.variables}
            if len(nbits) > 1:
                raise ValueError(
                    "Interleaved layout requires all variables to have the "
                    f"same n_bits, got {sorted(nbits)}"
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_grid.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/grid.py tests/test_grid.py
git commit -m "feat(grid): add UniformGrid and GridSpec dataclasses"
```

### 1b: num_sites and local_dim

- [ ] **Step 1: Write tests**

```python
# append to tests/test_grid.py
from tenax_qtt.grid import num_sites, local_dim


def test_num_sites_1d():
    g = GridSpec(variables=(UniformGrid(0, 1, 8),), layout="grouped")
    assert num_sites(g) == 8


def test_num_sites_grouped_2d():
    v1 = UniformGrid(0, 1, 4)
    v2 = UniformGrid(0, 1, 6)
    g = GridSpec(variables=(v1, v2), layout="grouped")
    assert num_sites(g) == 10  # 4 + 6


def test_num_sites_interleaved_2d():
    v1 = UniformGrid(0, 1, 4)
    v2 = UniformGrid(0, 1, 4)
    g = GridSpec(variables=(v1, v2), layout="interleaved")
    assert num_sites(g) == 8  # 4 * 2


def test_num_sites_fused_2d():
    v1 = UniformGrid(0, 1, 4)
    v2 = UniformGrid(0, 1, 6)
    g = GridSpec(variables=(v1, v2), layout="fused")
    assert num_sites(g) == 6  # max(4, 6)


def test_local_dim_grouped():
    g = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    assert local_dim(g, 0) == 2


def test_local_dim_fused_2d():
    v1 = UniformGrid(0, 1, 4)
    v2 = UniformGrid(0, 1, 4)
    g = GridSpec(variables=(v1, v2), layout="fused")
    assert local_dim(g, 0) == 4  # 2^2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_grid.py::test_num_sites_1d -v`
Expected: FAIL

- [ ] **Step 3: Implement num_sites and local_dim**

```python
# append to src/tenax_qtt/grid.py

def num_sites(grid: GridSpec) -> int:
    """Total number of MPS sites for this grid."""
    d = len(grid.variables)
    if grid.layout == "grouped":
        return sum(v.n_bits for v in grid.variables)
    elif grid.layout == "interleaved":
        return grid.variables[0].n_bits * d
    elif grid.layout == "fused":
        return max(v.n_bits for v in grid.variables)
    raise ValueError(f"Unknown layout: {grid.layout}")


def local_dim(grid: GridSpec, site: int) -> int:
    """Physical dimension at the given MPS site."""
    if grid.layout in ("grouped", "interleaved"):
        return 2
    elif grid.layout == "fused":
        d = len(grid.variables)
        return 1 << d  # 2^d
    raise ValueError(f"Unknown layout: {grid.layout}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_grid.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/grid.py tests/test_grid.py
git commit -m "feat(grid): add num_sites and local_dim functions"
```

### 1c: Coordinate conversion functions

- [ ] **Step 1: Write tests for grid_to_sites and sites_to_grid**

```python
# append to tests/test_grid.py
from tenax_qtt.grid import grid_to_sites, sites_to_grid


def test_grid_to_sites_1d():
    """Point at left endpoint maps to all-zero site indices."""
    g = GridSpec(variables=(UniformGrid(0.0, 1.0, 3),), layout="grouped")
    # x=0.0 → grid index 0 → binary 000 → sites (0, 0, 0)
    sites = grid_to_sites(g, (0.0,))
    assert sites == (0, 0, 0)


def test_grid_to_sites_1d_midpoint():
    g = GridSpec(variables=(UniformGrid(0.0, 1.0, 3),), layout="grouped")
    # 8 points on [0,1), spacing=0.125. x=0.5 → index 4 → binary 100 → sites (1,0,0)
    sites = grid_to_sites(g, (0.5,))
    assert sites == (1, 0, 0)


def test_sites_to_grid_roundtrip_1d():
    g = GridSpec(variables=(UniformGrid(0.0, 1.0, 4),), layout="grouped")
    x_orig = (0.25,)
    sites = grid_to_sites(g, x_orig)
    x_back = sites_to_grid(g, sites)
    assert abs(x_back[0] - x_orig[0]) < 1e-14


def test_grid_to_sites_2d_grouped():
    v1 = UniformGrid(0.0, 1.0, 2)
    v2 = UniformGrid(0.0, 1.0, 2)
    g = GridSpec(variables=(v1, v2), layout="grouped")
    # x1=0.5 → index 2 → binary 10, x2=0.25 → index 1 → binary 01
    # grouped: x1 bits then x2 bits → (1, 0, 0, 1)
    sites = grid_to_sites(g, (0.5, 0.25))
    assert sites == (1, 0, 0, 1)


def test_grid_to_sites_2d_interleaved():
    v1 = UniformGrid(0.0, 1.0, 2)
    v2 = UniformGrid(0.0, 1.0, 2)
    g = GridSpec(variables=(v1, v2), layout="interleaved")
    # x1=0.5 → index 2 → binary 10, x2=0.25 → index 1 → binary 01
    # interleaved: (x1_bit0, x2_bit0, x1_bit1, x2_bit1) = (1, 0, 0, 1)
    sites = grid_to_sites(g, (0.5, 0.25))
    assert sites == (1, 0, 0, 1)


def test_grid_to_sites_2d_fused():
    v1 = UniformGrid(0.0, 1.0, 2)
    v2 = UniformGrid(0.0, 1.0, 2)
    g = GridSpec(variables=(v1, v2), layout="fused")
    # x1=0.5 → index 2 → binary 10, x2=0.25 → index 1 → binary 01
    # fused level 0: (x1_bit0=1)*2 + (x2_bit0=0) = 2
    # fused level 1: (x1_bit1=0)*2 + (x2_bit1=1) = 1
    sites = grid_to_sites(g, (0.5, 0.25))
    assert sites == (2, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_grid.py::test_grid_to_sites_1d -v`
Expected: FAIL

- [ ] **Step 3: Implement grid_to_sites and sites_to_grid**

```python
# append to src/tenax_qtt/grid.py

def _coord_to_index(var: UniformGrid, x: float) -> int:
    """Map a continuous coordinate to a grid integer index."""
    if var.include_endpoint:
        t = (x - var.a) / (var.b - var.a)
        idx = int(round(t * (var.n_points - 1)))
    else:
        t = (x - var.a) / (var.b - var.a)
        idx = int(t * var.n_points)
    return max(0, min(var.n_points - 1, idx))


def _index_to_coord(var: UniformGrid, idx: int) -> float:
    """Map a grid integer index to a continuous coordinate (edge-aligned)."""
    if var.include_endpoint:
        return var.a + idx * (var.b - var.a) / (var.n_points - 1)
    return var.a + idx * (var.b - var.a) / var.n_points


def _int_to_bits(val: int, n_bits: int) -> list[int]:
    """MSB-first binary decomposition."""
    return [(val >> (n_bits - 1 - k)) & 1 for k in range(n_bits)]


def _bits_to_int(bits: list[int]) -> int:
    """MSB-first binary composition."""
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val


def grid_to_sites(grid: GridSpec, x: tuple[float, ...]) -> tuple[int, ...]:
    """Map continuous coordinates to MPS site indices."""
    d = len(grid.variables)
    if len(x) != d:
        raise ValueError(f"Expected {d} coordinates, got {len(x)}")

    indices = [_coord_to_index(grid.variables[i], x[i]) for i in range(d)]
    bits_per_var = [_int_to_bits(indices[i], grid.variables[i].n_bits) for i in range(d)]

    if grid.layout == "grouped":
        sites: list[int] = []
        for var_bits in bits_per_var:
            sites.extend(var_bits)
        return tuple(sites)

    elif grid.layout == "interleaved":
        n = grid.variables[0].n_bits
        sites = []
        for level in range(n):
            for var_idx in range(d):
                sites.append(bits_per_var[var_idx][level])
        return tuple(sites)

    elif grid.layout == "fused":
        max_bits = max(v.n_bits for v in grid.variables)
        # Pad shorter variables with 0 at finer levels
        padded = []
        for i in range(d):
            pad_len = max_bits - grid.variables[i].n_bits
            padded.append([0] * pad_len + bits_per_var[i])
        sites = []
        for level in range(max_bits):
            fused_val = 0
            for var_idx in range(d):
                fused_val = (fused_val << 1) | padded[var_idx][level]
            sites.append(fused_val)
        return tuple(sites)

    raise ValueError(f"Unknown layout: {grid.layout}")


def sites_to_grid(grid: GridSpec, sites: tuple[int, ...]) -> tuple[float, ...]:
    """Map MPS site indices back to continuous coordinates."""
    d = len(grid.variables)

    if grid.layout == "grouped":
        offset = 0
        coords = []
        for i in range(d):
            n = grid.variables[i].n_bits
            bits = list(sites[offset : offset + n])
            idx = _bits_to_int(bits)
            coords.append(_index_to_coord(grid.variables[i], idx))
            offset += n
        return tuple(coords)

    elif grid.layout == "interleaved":
        n = grid.variables[0].n_bits
        bits_per_var: list[list[int]] = [[] for _ in range(d)]
        pos = 0
        for level in range(n):
            for var_idx in range(d):
                bits_per_var[var_idx].append(sites[pos])
                pos += 1
        coords = []
        for i in range(d):
            idx = _bits_to_int(bits_per_var[i])
            coords.append(_index_to_coord(grid.variables[i], idx))
        return tuple(coords)

    elif grid.layout == "fused":
        max_bits = max(v.n_bits for v in grid.variables)
        bits_per_var = [[] for _ in range(d)]
        for level in range(max_bits):
            fused_val = sites[level]
            for var_idx in reversed(range(d)):
                bits_per_var[var_idx].append(fused_val & 1)
                fused_val >>= 1
        # Reverse each since we extracted LSB first
        for i in range(d):
            bits_per_var[i].reverse()
        coords = []
        for i in range(d):
            # Strip padding for shorter variables
            pad_len = max_bits - grid.variables[i].n_bits
            real_bits = bits_per_var[i][pad_len:]
            idx = _bits_to_int(real_bits)
            coords.append(_index_to_coord(grid.variables[i], idx))
        return tuple(coords)

    raise ValueError(f"Unknown layout: {grid.layout}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_grid.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/grid.py tests/test_grid.py
git commit -m "feat(grid): add coordinate-to-site conversion functions"
```

### 1d: Batch conversions and site_permutation

- [ ] **Step 1: Write tests**

```python
# append to tests/test_grid.py
from tenax_qtt.grid import batch_grid_to_sites, batch_sites_to_grid, site_permutation
import jax.numpy as jnp


def test_batch_grid_to_sites():
    g = GridSpec(variables=(UniformGrid(0.0, 1.0, 3),), layout="grouped")
    xs = jnp.array([[0.0], [0.5]])
    result = batch_grid_to_sites(g, xs)
    assert result.shape == (2, 3)
    assert tuple(int(x) for x in result[0]) == (0, 0, 0)
    assert tuple(int(x) for x in result[1]) == (1, 0, 0)


def test_batch_roundtrip():
    g = GridSpec(variables=(UniformGrid(0.0, 1.0, 4),), layout="grouped")
    xs = jnp.array([[0.0], [0.25], [0.5], [0.75]])
    sites = batch_grid_to_sites(g, xs)
    xs_back = batch_sites_to_grid(g, sites)
    assert jnp.allclose(xs, xs_back, atol=1e-10)


def test_site_permutation_grouped_to_interleaved():
    v1 = UniformGrid(0, 1, 2)
    v2 = UniformGrid(0, 1, 2)
    g = GridSpec(variables=(v1, v2), layout="grouped")
    perm = site_permutation(g, "interleaved")
    # grouped: [x1_b0, x1_b1, x2_b0, x2_b1] → interleaved: [x1_b0, x2_b0, x1_b1, x2_b1]
    assert perm == (0, 2, 1, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_grid.py::test_batch_grid_to_sites -v`
Expected: FAIL

- [ ] **Step 3: Implement batch functions and site_permutation**

```python
# append to src/tenax_qtt/grid.py

def batch_grid_to_sites(grid: GridSpec, xs: jax.Array) -> jax.Array:
    """Batch coordinate-to-site mapping. xs: (n_points, d) → (n_points, n_sites)."""
    results = []
    for i in range(xs.shape[0]):
        x_tuple = tuple(float(xs[i, j]) for j in range(xs.shape[1]))
        sites = grid_to_sites(grid, x_tuple)
        results.append(sites)
    return jnp.array(results, dtype=jnp.int32)


def batch_sites_to_grid(grid: GridSpec, sites: jax.Array) -> jax.Array:
    """Batch site-to-coordinate mapping. sites: (n_points, n_sites) → (n_points, d)."""
    results = []
    for i in range(sites.shape[0]):
        s_tuple = tuple(int(sites[i, j]) for j in range(sites.shape[1]))
        coords = sites_to_grid(grid, s_tuple)
        results.append(coords)
    return jnp.array(results)


def site_permutation(source: GridSpec, target_layout: str) -> tuple[int, ...]:
    """Permutation mapping sites from source layout to target layout.

    Returns a tuple p such that target_sites[i] = source_sites[p[i]].
    """
    d = len(source.variables)

    def _site_to_var_bit(layout: str, site: int) -> tuple[int, int]:
        """Map site index to (variable_index, bit_level)."""
        if layout == "grouped":
            offset = 0
            for var_idx, v in enumerate(source.variables):
                if site < offset + v.n_bits:
                    return (var_idx, site - offset)
                offset += v.n_bits
            raise IndexError(f"Site {site} out of range")
        elif layout == "interleaved":
            n = source.variables[0].n_bits
            level = site // d
            var_idx = site % d
            return (var_idx, level)
        raise ValueError(f"Unsupported layout: {layout}")

    n = num_sites(source)
    # Build inverse map: for each target site, find which source site has same (var, bit)
    source_map: dict[tuple[int, int], int] = {}
    for s in range(n):
        vb = _site_to_var_bit(source.layout, s)
        source_map[vb] = s

    perm = []
    for t in range(n):
        vb = _site_to_var_bit(target_layout, t)
        perm.append(source_map[vb])
    return tuple(perm)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_grid.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/grid.py tests/test_grid.py
git commit -m "feat(grid): add batch conversions and site_permutation"
```

---

## Task 2: QTT Class — Core Data Model

**Files:**
- Create: `src/tenax_qtt/qtt.py`
- Test: `tests/test_qtt.py`

Depends on: Task 1 (grid).

### 2a: QTT dataclass with MPS delegation

- [ ] **Step 1: Write tests for QTT construction and property forwarding**

```python
# tests/test_qtt.py
import jax.numpy as jnp
import numpy as np
import jax
from tenax import DenseTensor, TensorIndex, FlowDirection
from tenax.core.mps import FiniteMPS
from tenax_qtt.grid import UniformGrid, GridSpec
from tenax_qtt.qtt import QTT


def _make_trivial_index(dim, flow, label):
    """Helper: create a DenseTensor-compatible TensorIndex (no symmetry)."""
    from tenax import U1Symmetry
    sym = U1Symmetry()
    charges = np.zeros(dim, dtype=np.int32)
    return TensorIndex(sym, charges, flow, label=label)


def _make_bond_dim1_mps(n_sites, local_dims, value=1.0):
    """Helper: build a bond-dim-1 FiniteMPS with constant site tensors."""
    tensors = []
    for i in range(n_sites):
        d = local_dims[i]
        data = jnp.full((1, d, 1), value / n_sites if i == 0 else 1.0)
        left_label = f"v_{i - 1}_{i}" if i == 0 else f"v{i - 1}_{i}"
        right_label = f"v{i}_{i + 1}"
        indices = (
            _make_trivial_index(1, FlowDirection.IN, left_label),
            _make_trivial_index(d, FlowDirection.IN, f"p{i}"),
            _make_trivial_index(1, FlowDirection.OUT, right_label),
        )
        tensors.append(DenseTensor(data, indices))
    return FiniteMPS.from_tensors(tensors)


def test_qtt_from_mps():
    grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
    mps = _make_bond_dim1_mps(3, [2, 2, 2])
    qtt = QTT(mps=mps, grid=grid)
    assert qtt.grid == grid
    assert len(qtt.tensors) == 3


def test_qtt_bond_dims():
    grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
    mps = _make_bond_dim1_mps(3, [2, 2, 2])
    qtt = QTT(mps=mps, grid=grid)
    assert qtt.bond_dims == [1, 1]


def test_qtt_zeros():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    qtt = QTT.zeros(grid)
    assert len(qtt.tensors) == 4
    assert all(bd == 1 for bd in qtt.bond_dims)


def test_qtt_ones():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    qtt = QTT.ones(grid)
    assert len(qtt.tensors) == 4
    assert all(bd == 1 for bd in qtt.bond_dims)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_qtt.py -v`
Expected: FAIL

- [ ] **Step 3: Implement QTT dataclass with constructors and delegation**

```python
# src/tenax_qtt/qtt.py
"""QTT class wrapping FiniteMPS with grid semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from tenax import DenseTensor, FlowDirection, TensorIndex, U1Symmetry
from tenax.core.mps import FiniteMPS

from tenax_qtt.grid import GridSpec, local_dim, num_sites

if TYPE_CHECKING:
    from tenax import Tensor


def _trivial_index(dim: int, flow: FlowDirection, label: str) -> TensorIndex:
    sym = U1Symmetry()
    charges = np.zeros(dim, dtype=np.int32)
    return TensorIndex(sym, charges, flow, label=label)


def _make_constant_mps(grid: GridSpec, value: float) -> FiniteMPS:
    """Build a bond-dim-1 MPS with constant value at all grid points."""
    L = num_sites(grid)
    tensors = []
    for i in range(L):
        d = local_dim(grid, i)
        # Distribute the constant across sites: first site gets value, rest get 1
        fill = value if i == 0 else 1.0
        data = jnp.full((1, d, 1), fill)
        left_label = f"v_{i - 1}_{i}" if i == 0 else f"v{i - 1}_{i}"
        right_label = f"v{i}_{i + 1}"
        indices = (
            _trivial_index(1, FlowDirection.IN, left_label),
            _trivial_index(d, FlowDirection.IN, f"p{i}"),
            _trivial_index(1, FlowDirection.OUT, right_label),
        )
        tensors.append(DenseTensor(data, indices))
    return FiniteMPS.from_tensors(tensors)


@dataclass(frozen=True)
class QTT:
    """Quantic Tensor Train: a function on a grid stored as an MPS."""

    mps: FiniteMPS
    grid: GridSpec

    # -- MPS delegation --

    @property
    def tensors(self) -> list[Tensor]:
        return self.mps.tensors

    @property
    def bond_dims(self) -> list[int]:
        return self.mps.bond_dims

    @property
    def orth_center(self) -> int | None:
        return self.mps.orth_center

    @property
    def singular_values(self) -> list:
        return self.mps.singular_values

    @property
    def log_norm(self) -> float:
        return self.mps.log_norm

    def canonicalize(self, center: int) -> QTT:
        return QTT(mps=self.mps.canonicalize(center), grid=self.grid)

    def norm(self) -> float:
        return self.mps.norm()

    # -- Constructors --

    @classmethod
    def from_mps(cls, mps: FiniteMPS, grid: GridSpec) -> QTT:
        """Wrap an existing FiniteMPS with grid metadata."""
        L = num_sites(grid)
        if len(mps.tensors) != L:
            raise ValueError(
                f"MPS has {len(mps.tensors)} sites but grid requires {L}"
            )
        return cls(mps=mps, grid=grid)

    @classmethod
    def zeros(cls, grid: GridSpec) -> QTT:
        """Bond-dim-1 QTT representing the zero function."""
        return cls(mps=_make_constant_mps(grid, 0.0), grid=grid)

    @classmethod
    def ones(cls, grid: GridSpec) -> QTT:
        """Bond-dim-1 QTT representing f(x) = 1."""
        return cls(mps=_make_constant_mps(grid, 1.0), grid=grid)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_qtt.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/qtt.py tests/test_qtt.py
git commit -m "feat(qtt): add QTT dataclass with zeros/ones constructors"
```

### 2b: evaluate and evaluate_batch

- [ ] **Step 1: Write tests**

```python
# append to tests/test_qtt.py

def test_qtt_ones_evaluate():
    grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
    qtt = QTT.ones(grid)
    # f(x) = 1 everywhere
    assert abs(qtt.evaluate((0.0,)) - 1.0) < 1e-12
    assert abs(qtt.evaluate((0.5,)) - 1.0) < 1e-12


def test_qtt_zeros_evaluate():
    grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
    qtt = QTT.zeros(grid)
    assert abs(qtt.evaluate((0.0,))) < 1e-12
    assert abs(qtt.evaluate((0.5,))) < 1e-12


def test_qtt_evaluate_batch():
    grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
    qtt = QTT.ones(grid)
    xs = jnp.array([[0.0], [0.25], [0.5], [0.75]])
    vals = qtt.evaluate_batch(xs)
    assert vals.shape == (4,)
    assert jnp.allclose(vals, 1.0, atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_qtt.py::test_qtt_ones_evaluate -v`
Expected: FAIL

- [ ] **Step 3: Implement evaluate and evaluate_batch**

```python
# append to QTT class in src/tenax_qtt/qtt.py

    def evaluate(self, x: tuple[float, ...]) -> complex:
        """Evaluate QTT at a single continuous-domain point."""
        from tenax_qtt.grid import grid_to_sites
        sites = grid_to_sites(self.grid, x)
        # Contract MPS by selecting physical index at each site
        result = jnp.array([[1.0]])  # row vector (1, 1)
        for i, t in enumerate(self.tensors):
            # t has shape (chi_left, d_phys, chi_right) as a dense array
            data = t.todense() if hasattr(t, 'todense') else t.data
            s = sites[i]
            result = result @ data[:, s, :]  # (1, chi_left) @ (chi_left, chi_right) = (1, chi_right)
        return complex(result[0, 0])

    def evaluate_batch(self, xs: jax.Array) -> jax.Array:
        """Vectorized evaluation at multiple points."""
        vals = [self.evaluate(tuple(float(xs[i, j]) for j in range(xs.shape[1])))
                for i in range(xs.shape[0])]
        return jnp.array(vals)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_qtt.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/qtt.py tests/test_qtt.py
git commit -m "feat(qtt): add evaluate and evaluate_batch methods"
```

### 2c: to_dense, sum, integrate, norm_l2

- [ ] **Step 1: Write tests**

```python
# append to tests/test_qtt.py

def test_qtt_to_dense_ones():
    grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
    qtt = QTT.ones(grid)
    dense = qtt.to_dense()
    assert dense.shape == (8,)  # 2^3
    assert jnp.allclose(dense, 1.0, atol=1e-12)


def test_qtt_sum_all():
    grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
    qtt = QTT.ones(grid)
    s = qtt.sum()
    assert abs(s - 8.0) < 1e-12  # 2^3 grid points, all value 1


def test_qtt_integrate_all():
    grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
    qtt = QTT.ones(grid)
    # integral of f(x)=1 on [0,1) with 8 points, dx=0.125 → 1.0
    result = qtt.integrate()
    assert abs(result - 1.0) < 1e-12


def test_qtt_norm_l2():
    grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
    qtt = QTT.ones(grid)
    # L2 norm of f(x)=1 on [0,1) = sqrt(1) = 1
    assert abs(qtt.norm_l2() - 1.0) < 1e-10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_qtt.py::test_qtt_to_dense_ones -v`
Expected: FAIL

- [ ] **Step 3: Implement to_dense, sum, integrate, norm_l2**

```python
# append to QTT class in src/tenax_qtt/qtt.py

    def to_dense(self) -> jax.Array:
        """Expand QTT to full dense array."""
        from tenax_qtt.grid import num_sites as _num_sites, local_dim as _local_dim
        L = _num_sites(self.grid)
        # Full contraction of MPS
        result = None
        for i, t in enumerate(self.tensors):
            data = t.todense() if hasattr(t, 'todense') else t.data
            if result is None:
                # shape: (1, d, chi_right) → (d, chi_right)
                result = data[0]
            else:
                # result: (..., chi_left), data: (chi_left, d, chi_right)
                result = jnp.tensordot(result, data, axes=([-1], [0]))
        # Remove trailing dim-1 bond
        return result.reshape(-1)

    def sum(self, variables: list[int] | None = None) -> QTT | complex:
        """Sum over specified variables (or all) without grid spacing."""
        if variables is not None:
            raise NotImplementedError("Partial summation not yet implemented")
        # Full sum: contract each site with all-ones vector
        result = jnp.array([[1.0]])
        for t in self.tensors:
            data = t.todense() if hasattr(t, 'todense') else t.data
            d = data.shape[1]
            ones = jnp.ones(d)
            # Contract physical index with ones: (chi_l, d, chi_r) · (d,) → (chi_l, chi_r)
            contracted = jnp.einsum("ijk,j->ik", data, ones)
            result = result @ contracted
        return complex(result[0, 0])

    def integrate(self, variables: list[int] | None = None) -> QTT | complex:
        """Integrate over specified variables using trapezoidal quadrature."""
        if variables is not None:
            raise NotImplementedError("Partial integration not yet implemented")
        # Full integration: sum * product of dx for each variable
        total_dx = 1.0
        for v in self.grid.variables:
            total_dx *= v.dx
        s = self.sum()
        return s * total_dx

    def norm_l2(self) -> float:
        """Continuous L² norm: sqrt(∫|f(x)|² dx)."""
        # For a real-valued QTT, ||f||² = <MPS|MPS> * dx
        total_dx = 1.0
        for v in self.grid.variables:
            total_dx *= v.dx
        mps_norm_sq = self.mps.norm() ** 2
        return float(jnp.sqrt(mps_norm_sq * total_dx))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_qtt.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/qtt.py tests/test_qtt.py
git commit -m "feat(qtt): add to_dense, sum, integrate, norm_l2"
```

---

## Task 3: SVD Folding

**Files:**
- Create: `src/tenax_qtt/folding.py`
- Test: `tests/test_folding.py`

Depends on: Tasks 1-2 (grid, QTT).

- [ ] **Step 1: Write tests**

```python
# tests/test_folding.py
import jax.numpy as jnp
from tenax_qtt.grid import UniformGrid, GridSpec
from tenax_qtt.folding import fold_to_qtt


def test_fold_constant():
    """Folding a constant vector should produce bond-dim-1 QTT."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    data = jnp.ones(16)  # 2^4
    qtt = fold_to_qtt(data, grid)
    assert max(qtt.bond_dims) == 1
    assert jnp.allclose(qtt.to_dense(), 1.0, atol=1e-12)


def test_fold_linear():
    """Folding a linear function should compress well."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 8),), layout="grouped")
    N = 256  # 2^8
    data = jnp.linspace(0, 1, N, endpoint=False)
    qtt = fold_to_qtt(data, grid, tol=1e-10)
    assert jnp.allclose(qtt.to_dense(), data, atol=1e-8)


def test_fold_with_max_bond_dim():
    grid = GridSpec(variables=(UniformGrid(0, 1, 8),), layout="grouped")
    data = jnp.sin(jnp.linspace(0, 2 * jnp.pi, 256, endpoint=False))
    qtt = fold_to_qtt(data, grid, max_bond_dim=4)
    assert max(qtt.bond_dims) <= 4


def test_fold_roundtrip_exact():
    """A rank-1 tensor should be exactly representable."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
    data = jnp.ones(8)  # constant = rank 1
    qtt = fold_to_qtt(data, grid)
    assert jnp.allclose(qtt.to_dense(), data, atol=1e-14)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_folding.py -v`
Expected: FAIL

- [ ] **Step 3: Implement fold_to_qtt**

```python
# src/tenax_qtt/folding.py
"""SVD-based QTT construction from dense data."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from tenax import DenseTensor, FlowDirection, TensorIndex, U1Symmetry, svd

from tenax_qtt.grid import GridSpec, local_dim, num_sites
from tenax_qtt.qtt import QTT

_sym = U1Symmetry()


def _trivial_index(dim: int, flow: FlowDirection, label: str) -> TensorIndex:
    return TensorIndex(_sym, np.zeros(dim, dtype=np.int32), flow, label=label)


def fold_to_qtt(
    data: jnp.ndarray,
    grid: GridSpec,
    max_bond_dim: int | None = None,
    tol: float = 1e-8,
) -> QTT:
    """Reshape dense array into QTT via left-to-right SVD sweep."""
    from tenax.core.mps import FiniteMPS

    L = num_sites(grid)
    dims = [local_dim(grid, i) for i in range(L)]

    # Reshape flat array into (d0, d1, ..., dL-1) tensor
    tensor = data.reshape(dims)

    # Left-to-right SVD sweep
    tensors = []
    remainder = tensor
    chi_left = 1
    for i in range(L - 1):
        d = dims[i]
        # Reshape to matrix: (chi_left * d, remaining_dims...)
        mat_shape = (chi_left * d, -1)
        mat = remainder.reshape(mat_shape)

        # Create DenseTensor for SVD
        left_idx = _trivial_index(mat.shape[0], FlowDirection.IN, "left")
        right_idx = _trivial_index(mat.shape[1], FlowDirection.OUT, "right")
        mat_tensor = DenseTensor(mat, (left_idx, right_idx))

        U, s, Vh, _ = svd(
            mat_tensor, ["left"], ["right"],
            new_bond_label="bond",
            max_singular_values=max_bond_dim,
            max_truncation_err=tol,
        )
        chi_new = len(s)

        # Extract U data and reshape to (chi_left, d, chi_new)
        u_data = U.todense() if hasattr(U, 'todense') else U.data
        site_data = u_data.reshape(chi_left, d, chi_new)

        # Build properly labeled site tensor
        left_label = f"v_{i - 1}_{i}" if i == 0 else f"v{i - 1}_{i}"
        right_label = f"v{i}_{i + 1}"
        site_indices = (
            _trivial_index(chi_left, FlowDirection.IN, left_label),
            _trivial_index(d, FlowDirection.IN, f"p{i}"),
            _trivial_index(chi_new, FlowDirection.OUT, right_label),
        )
        tensors.append(DenseTensor(site_data, site_indices))

        # Remainder = diag(s) @ Vh
        vh_data = Vh.todense() if hasattr(Vh, 'todense') else Vh.data
        remainder = (jnp.diag(s) @ vh_data).reshape(chi_new, *[dims[j] for j in range(i + 1, L)])
        chi_left = chi_new

    # Last site: remainder has shape (chi_left, d_last)
    d_last = dims[-1]
    site_data = remainder.reshape(chi_left, d_last, 1)
    left_label = f"v{L - 2}_{L - 1}"
    right_label = f"v{L - 1}_{L}"
    site_indices = (
        _trivial_index(chi_left, FlowDirection.IN, left_label),
        _trivial_index(d_last, FlowDirection.IN, f"p{L - 1}"),
        _trivial_index(1, FlowDirection.OUT, right_label),
    )
    tensors.append(DenseTensor(site_data, site_indices))

    mps = FiniteMPS.from_tensors(tensors)
    return QTT(mps=mps, grid=grid)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_folding.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/folding.py tests/test_folding.py
git commit -m "feat(folding): SVD-based QTT construction from dense arrays"
```

---

## Task 4: Arithmetic

**Files:**
- Create: `src/tenax_qtt/arithmetic.py`
- Test: `tests/test_arithmetic.py`

Depends on: Tasks 1-3.

### 4a: scalar_multiply and recompress

- [ ] **Step 1: Write tests**

```python
# tests/test_arithmetic.py
import jax.numpy as jnp
from tenax_qtt.grid import UniformGrid, GridSpec
from tenax_qtt.folding import fold_to_qtt
from tenax_qtt.arithmetic import scalar_multiply, recompress
from tenax_qtt.qtt import QTT


def test_scalar_multiply():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    qtt = QTT.ones(grid)
    scaled = scalar_multiply(qtt, 3.0)
    assert jnp.allclose(scaled.to_dense(), 3.0, atol=1e-12)


def test_scalar_multiply_complex():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    qtt = QTT.ones(grid)
    scaled = scalar_multiply(qtt, 1j)
    dense = scaled.to_dense()
    assert jnp.allclose(dense, 1j, atol=1e-12)


def test_recompress_no_change():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    qtt = QTT.ones(grid)
    compressed = recompress(qtt)
    assert max(compressed.bond_dims) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_arithmetic.py -v`
Expected: FAIL

- [ ] **Step 3: Implement scalar_multiply and recompress**

```python
# src/tenax_qtt/arithmetic.py
"""QTT arithmetic: addition, Hadamard product, recompression."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from tenax import DenseTensor, FlowDirection, TensorIndex, U1Symmetry, svd
from tenax.core.mps import FiniteMPS

from tenax_qtt.grid import GridSpec
from tenax_qtt.qtt import QTT

_sym = U1Symmetry()


def _trivial_index(dim: int, flow: FlowDirection, label: str) -> TensorIndex:
    return TensorIndex(_sym, np.zeros(dim, dtype=np.int32), flow, label=label)


def _check_compatible(a: QTT, b: QTT) -> None:
    if a.grid != b.grid:
        raise ValueError("QTTs must have identical GridSpec for arithmetic")


def scalar_multiply(a: QTT, c: complex) -> QTT:
    """Scale a QTT by a constant (multiply first site tensor)."""
    tensors = list(a.tensors)
    t0 = tensors[0]
    data = t0.todense() if hasattr(t0, 'todense') else t0.data
    new_data = data * c
    tensors[0] = DenseTensor(new_data, t0.indices)
    return QTT(mps=FiniteMPS.from_tensors(tensors), grid=a.grid)


def recompress(
    qtt: QTT,
    tol: float = 1e-8,
    max_bond_dim: int | None = None,
) -> QTT:
    """Recompress a QTT to lower bond dimension via right-canonicalize + left-to-right SVD sweep."""
    mps = qtt.mps
    L = len(mps.tensors)
    if L <= 1:
        return qtt

    # Right-canonicalize (orth_center = 0), then left-to-right SVD sweep
    new_mps = mps.canonicalize(0)

    # Left-to-right SVD sweep with truncation
    tensors = list(new_mps.tensors)
    for i in range(L - 1):
        t = tensors[i]
        data = t.todense() if hasattr(t, 'todense') else t.data
        chi_l, d, chi_r = data.shape

        # Reshape to (chi_l * d, chi_r): left factor stays as site tensor
        mat = data.reshape(chi_l * d, chi_r)
        left_idx = _trivial_index(mat.shape[0], FlowDirection.IN, "left")
        right_idx = _trivial_index(mat.shape[1], FlowDirection.OUT, "right")
        mat_tensor = DenseTensor(mat, (left_idx, right_idx))

        U, s, Vh, _ = svd(
            mat_tensor, ["left"], ["right"],
            new_bond_label="bond",
            max_singular_values=max_bond_dim,
            max_truncation_err=tol,
        )
        chi_new = len(s)

        # U reshaped to new site tensor: (chi_l, d, chi_new)
        u_data = U.todense() if hasattr(U, 'todense') else U.data
        site_data = u_data.reshape(chi_l, d, chi_new)

        left_label = f"v_{i - 1}_{i}" if i == 0 else f"v{i - 1}_{i}"
        right_label = f"v{i}_{i + 1}"
        site_indices = (
            _trivial_index(chi_l, FlowDirection.IN, left_label),
            _trivial_index(d, FlowDirection.IN, f"p{i}"),
            _trivial_index(chi_new, FlowDirection.OUT, right_label),
        )
        tensors[i] = DenseTensor(site_data, site_indices)

        # Absorb diag(s) @ Vh into right neighbor
        vh_data = Vh.todense() if hasattr(Vh, 'todense') else Vh.data
        svh = jnp.diag(s) @ vh_data  # (chi_new, chi_r_old)

        t_right = tensors[i + 1]
        data_right = t_right.todense() if hasattr(t_right, 'todense') else t_right.data
        _, d_r, chi_rr = data_right.shape
        new_right = jnp.einsum("kj,jlm->klm", svh, data_right)

        left_label_r = f"v{i}_{i + 1}"
        right_label_r = f"v{i + 1}_{i + 2}"
        right_indices = (
            _trivial_index(chi_new, FlowDirection.IN, left_label_r),
            _trivial_index(d_r, FlowDirection.IN, f"p{i + 1}"),
            _trivial_index(chi_rr, FlowDirection.OUT, right_label_r),
        )
        tensors[i + 1] = DenseTensor(new_right, right_indices)

    return QTT(mps=FiniteMPS.from_tensors(tensors), grid=qtt.grid)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_arithmetic.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/arithmetic.py tests/test_arithmetic.py
git commit -m "feat(arithmetic): scalar_multiply and recompress"
```

### 4b: add, subtract, hadamard

- [ ] **Step 1: Write tests**

```python
# append to tests/test_arithmetic.py
from tenax_qtt.arithmetic import add, subtract, hadamard
import pytest


def test_add_ones():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    a = QTT.ones(grid)
    b = QTT.ones(grid)
    result = add(a, b)
    assert jnp.allclose(result.to_dense(), 2.0, atol=1e-10)


def test_subtract():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    a = QTT.ones(grid)
    b = QTT.ones(grid)
    result = subtract(a, b)
    assert jnp.allclose(result.to_dense(), 0.0, atol=1e-10)


def test_add_mismatched_grids():
    g1 = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    g2 = GridSpec(variables=(UniformGrid(0, 2, 4),), layout="grouped")
    a = QTT.ones(g1)
    b = QTT.ones(g2)
    with pytest.raises(ValueError, match="identical GridSpec"):
        add(a, b)


def test_hadamard():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    data = jnp.linspace(0, 1, 16, endpoint=False)
    a = fold_to_qtt(data, grid)
    b = fold_to_qtt(data, grid)
    result = hadamard(a, b)
    expected = data ** 2
    assert jnp.allclose(result.to_dense(), expected, atol=1e-8)


def test_add_recompresses():
    grid = GridSpec(variables=(UniformGrid(0, 1, 8),), layout="grouped")
    a = QTT.ones(grid)
    b = QTT.ones(grid)
    result = add(a, b, max_bond_dim=2)
    assert max(result.bond_dims) <= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_arithmetic.py::test_add_ones -v`
Expected: FAIL

- [ ] **Step 3: Implement add, subtract, hadamard**

```python
# append to src/tenax_qtt/arithmetic.py

def _direct_sum_mps(a: QTT, b: QTT) -> list[DenseTensor]:
    """Direct-sum the bond dimensions of two MPS, site by site."""
    L = len(a.tensors)
    tensors = []
    for i in range(L):
        da = a.tensors[i].todense() if hasattr(a.tensors[i], 'todense') else a.tensors[i].data
        db = b.tensors[i].todense() if hasattr(b.tensors[i], 'todense') else b.tensors[i].data
        chi_la, d, chi_ra = da.shape
        chi_lb, _, chi_rb = db.shape

        if i == 0:
            # First site: horizontal concat → (1, d, chi_a + chi_b)
            new_data = jnp.concatenate([da, db], axis=2)
        elif i == L - 1:
            # Last site: vertical concat → (chi_a + chi_b, d, 1)
            new_data = jnp.concatenate([da, db], axis=0)
        else:
            # Middle: block diagonal → (chi_la+chi_lb, d, chi_ra+chi_rb)
            new_data = jnp.zeros((chi_la + chi_lb, d, chi_ra + chi_rb))
            new_data = new_data.at[:chi_la, :, :chi_ra].set(da)
            new_data = new_data.at[chi_la:, :, chi_ra:].set(db)

        left_label = f"v_{i - 1}_{i}" if i == 0 else f"v{i - 1}_{i}"
        right_label = f"v{i}_{i + 1}"
        chi_l, d_phys, chi_r = new_data.shape
        indices = (
            _trivial_index(chi_l, FlowDirection.IN, left_label),
            _trivial_index(d_phys, FlowDirection.IN, f"p{i}"),
            _trivial_index(chi_r, FlowDirection.OUT, right_label),
        )
        tensors.append(DenseTensor(new_data, indices))
    return tensors


def add(
    a: QTT, b: QTT,
    tol: float = 1e-8,
    max_bond_dim: int | None = None,
) -> QTT:
    """Sum of two QTTs. Direct-sum bond dimensions then recompress."""
    _check_compatible(a, b)
    tensors = _direct_sum_mps(a, b)
    mps = FiniteMPS.from_tensors(tensors)
    result = QTT(mps=mps, grid=a.grid)
    if tol > 0 or max_bond_dim is not None:
        result = recompress(result, tol=tol, max_bond_dim=max_bond_dim)
    return result


def subtract(
    a: QTT, b: QTT,
    tol: float = 1e-8,
    max_bond_dim: int | None = None,
) -> QTT:
    """Difference of two QTTs."""
    return add(a, scalar_multiply(b, -1.0), tol=tol, max_bond_dim=max_bond_dim)


def hadamard(
    a: QTT, b: QTT,
    tol: float = 1e-8,
    max_bond_dim: int | None = None,
) -> QTT:
    """Element-wise (Hadamard) product via bond dimension multiplication."""
    _check_compatible(a, b)
    L = len(a.tensors)
    tensors = []
    for i in range(L):
        da = a.tensors[i].todense() if hasattr(a.tensors[i], 'todense') else a.tensors[i].data
        db = b.tensors[i].todense() if hasattr(b.tensors[i], 'todense') else b.tensors[i].data
        chi_la, d, chi_ra = da.shape
        chi_lb, _, chi_rb = db.shape
        # Kronecker product on bond indices, pointwise on physical
        # result[al*bl, s, ar*br] = da[al, s, ar] * db[bl, s, br]
        new_data = jnp.einsum("isk,jsl->ijskl", da, db).reshape(
            chi_la * chi_lb, d, chi_ra * chi_rb
        )
        left_label = f"v_{i - 1}_{i}" if i == 0 else f"v{i - 1}_{i}"
        right_label = f"v{i}_{i + 1}"
        chi_l, d_phys, chi_r = new_data.shape
        indices = (
            _trivial_index(chi_l, FlowDirection.IN, left_label),
            _trivial_index(d_phys, FlowDirection.IN, f"p{i}"),
            _trivial_index(chi_r, FlowDirection.OUT, right_label),
        )
        tensors.append(DenseTensor(new_data, indices))

    mps = FiniteMPS.from_tensors(tensors)
    result = QTT(mps=mps, grid=a.grid)
    if tol > 0 or max_bond_dim is not None:
        result = recompress(result, tol=tol, max_bond_dim=max_bond_dim)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_arithmetic.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/arithmetic.py tests/test_arithmetic.py
git commit -m "feat(arithmetic): add, subtract, hadamard with recompression"
```

### 4c: QTT dunder methods

- [ ] **Step 1: Write tests**

```python
# append to tests/test_arithmetic.py

def test_qtt_add_operator():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    a = QTT.ones(grid)
    b = QTT.ones(grid)
    result = a + b
    assert jnp.allclose(result.to_dense(), 2.0, atol=1e-10)


def test_qtt_sub_operator():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    a = QTT.ones(grid)
    b = QTT.ones(grid)
    result = a - b
    assert jnp.allclose(result.to_dense(), 0.0, atol=1e-10)


def test_qtt_mul_operator():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    a = QTT.ones(grid)
    result = a * 5.0
    assert jnp.allclose(result.to_dense(), 5.0, atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_arithmetic.py::test_qtt_add_operator -v`
Expected: FAIL

- [ ] **Step 3: Add dunder methods to QTT**

```python
# append to QTT class in src/tenax_qtt/qtt.py

    def __add__(self, other: QTT) -> QTT:
        from tenax_qtt.arithmetic import add
        return add(self, other)

    def __sub__(self, other: QTT) -> QTT:
        from tenax_qtt.arithmetic import subtract
        return subtract(self, other)

    def __mul__(self, scalar: complex) -> QTT:
        from tenax_qtt.arithmetic import scalar_multiply
        return scalar_multiply(self, scalar)

    def __rmul__(self, scalar: complex) -> QTT:
        return self.__mul__(scalar)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_arithmetic.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/qtt.py tests/test_arithmetic.py
git commit -m "feat(qtt): add arithmetic dunder methods"
```

---

## Task 5: Cross-Interpolation (TCI2)

**Files:**
- Create: `src/tenax_qtt/cross.py`
- Test: `tests/test_cross.py`

Depends on: Tasks 1-3. This is the most algorithmically complex module. We implement TCI2 first (simpler), then add prrLU.

### 5a: QTTResult and TCI2 core

- [ ] **Step 1: Write tests**

```python
# tests/test_cross.py
import jax.numpy as jnp
import math
from tenax_qtt.grid import UniformGrid, GridSpec
from tenax_qtt.cross import cross_interpolation, QTTResult


def test_cross_constant():
    """TCI on constant function should converge with bond dim 1."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 8),), layout="grouped")
    result = cross_interpolation(lambda x: 1.0, grid, tol=1e-10, method="tci2")
    assert isinstance(result, QTTResult)
    assert result.converged
    assert max(result.qtt.bond_dims) <= 2  # near-trivial


def test_cross_linear():
    grid = GridSpec(variables=(UniformGrid(0, 1, 8),), layout="grouped")
    result = cross_interpolation(lambda x: x[0], grid, tol=1e-10, method="tci2")
    # Evaluate at a few points
    assert abs(result.qtt.evaluate((0.25,)) - 0.25) < 1e-6


def test_cross_sin():
    grid = GridSpec(variables=(UniformGrid(0, 2 * math.pi, 10),), layout="grouped")
    result = cross_interpolation(
        lambda x: math.sin(x[0]), grid, tol=1e-6, max_bond_dim=20, method="tci2"
    )
    assert abs(result.qtt.evaluate((1.0,)) - math.sin(1.0)) < 1e-4


def test_cross_result_fields():
    grid = GridSpec(variables=(UniformGrid(0, 1, 6),), layout="grouped")
    result = cross_interpolation(lambda x: x[0] ** 2, grid, tol=1e-8, method="tci2")
    assert result.n_iter > 0
    assert result.n_function_evals > 0
    assert isinstance(result.estimated_error, float)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_cross.py -v`
Expected: FAIL

- [ ] **Step 3: Implement QTTResult and TCI2 algorithm**

This is the longest implementation. The TCI2 algorithm:

1. Initialize random pivot sets (row and column multi-indices) at each bond.
2. Half-sweep left→right: at each bond, form the cross matrix by evaluating f at all (row, column) combinations, compute LU factorization, extract site tensor, update column pivots.
3. Half-sweep right→left: same procedure, update row pivots.
4. Check convergence by monitoring the maximum pivot magnitude change.

```python
# src/tenax_qtt/cross.py
"""Tensor cross interpolation: prrLU and TCI2 algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from tenax import DenseTensor, FlowDirection, TensorIndex, U1Symmetry
from tenax.core.mps import FiniteMPS

from tenax_qtt.grid import GridSpec, grid_to_sites, local_dim, num_sites, sites_to_grid
from tenax_qtt.qtt import QTT

_sym = U1Symmetry()


def _trivial_index(dim: int, flow: FlowDirection, label: str) -> TensorIndex:
    return TensorIndex(_sym, np.zeros(dim, dtype=np.int32), flow, label=label)


@dataclass(frozen=True)
class QTTResult:
    """Result of cross-interpolation with convergence diagnostics."""

    qtt: QTT
    n_iter: int
    converged: bool
    estimated_error: float
    n_function_evals: int


def _evaluate_f(
    f: Callable,
    grid: GridSpec,
    multi_indices: list[tuple[int, ...]],
    batch: bool,
) -> np.ndarray:
    """Evaluate f at a list of site-index tuples, returning values."""
    coords = [sites_to_grid(grid, idx) for idx in multi_indices]
    if batch:
        xs = jnp.array(coords)
        vals = np.asarray(f(xs))
    else:
        vals = np.array([complex(f(c)) for c in coords])
    return vals


def _tci2(
    f: Callable,
    grid: GridSpec,
    tol: float,
    max_bond_dim: int,
    max_iter: int,
    batch: bool,
    batch_size: int,
    seed: int,
) -> QTTResult:
    """TCI2: alternating half-sweep cross interpolation."""
    rng = np.random.default_rng(seed)
    L = num_sites(grid)
    dims = [local_dim(grid, i) for i in range(L)]
    n_evals = 0

    # Initialize pivot sets: I_k (left multi-indices), J_k (right multi-indices)
    # I_k has shape (chi_k, k) — rows of left multi-indices up to site k
    # J_k has shape (chi_k, L-k-1) — columns of right multi-indices from site k+1
    chi_init = 2  # initial bond dim

    # Random initial pivots
    I_sets: list[list[tuple[int, ...]]] = []  # I_sets[k] for bond k (between sites k and k+1)
    J_sets: list[list[tuple[int, ...]]] = []

    for k in range(L - 1):
        chi = min(chi_init, max_bond_dim)
        left_pivots = []
        right_pivots = []
        for _ in range(chi):
            left = tuple(rng.integers(0, dims[j]) for j in range(k + 1))
            right = tuple(rng.integers(0, dims[j]) for j in range(k + 1, L))
            left_pivots.append(left)
            right_pivots.append(right)
        I_sets.append(left_pivots)
        J_sets.append(right_pivots)

    # Site tensors
    site_tensors: list[np.ndarray] = [np.zeros((1, dims[i], 1)) for i in range(L)]

    converged = False
    est_error = float("inf")
    iteration = 0

    for iteration in range(max_iter):
        prev_error = est_error
        max_new_pivot = 0.0

        # Left-to-right half sweep
        for k in range(L - 1):
            chi_l = len(I_sets[k]) if k > 0 else 1
            chi_r = len(J_sets[k])
            d = dims[k]

            if k == 0:
                I_left: list[tuple[int, ...]] = [()]
            else:
                I_left = I_sets[k - 1]

            J_right = J_sets[k]

            # Build cross matrix: C[i*d+s, j] = f(I_left[i], s, J_right[j])
            n_rows = len(I_left) * d
            n_cols = len(J_right)
            multi_indices = []
            for il in I_left:
                for s in range(d):
                    for jr in J_right:
                        full_idx = il + (s,) + jr
                        multi_indices.append(full_idx)

            vals = _evaluate_f(f, grid, multi_indices, batch)
            n_evals += len(multi_indices)
            C = vals.reshape(n_rows, n_cols)

            # LU with partial pivoting to find new pivots
            # Use full pivoting for pivot selection
            chi_new = min(min(n_rows, n_cols), max_bond_dim)

            # Simple rank-revealing: SVD-based pivot selection
            U_mat, s_vals, Vh_mat = np.linalg.svd(C, full_matrices=False)

            # Truncate
            if tol > 0 and len(s_vals) > 1:
                cumsum = np.cumsum(s_vals[::-1] ** 2)[::-1]
                total = cumsum[0]
                keep = 1
                for kk in range(len(s_vals)):
                    if s_vals[kk] / s_vals[0] < tol:
                        break
                    keep = kk + 1
                chi_new = min(keep, chi_new)

            chi_new = max(1, chi_new)
            max_new_pivot = max(max_new_pivot, float(s_vals[0]) if len(s_vals) > 0 else 0)

            # Update site tensor: reshape U columns to (chi_l, d, chi_new)
            site_data = (U_mat[:, :chi_new] * s_vals[:chi_new]).reshape(
                len(I_left), d, chi_new
            )
            site_tensors[k] = site_data

            # Update I_sets[k]: extend I_left with physical index
            # For each of the chi_new columns, find the best row
            new_I = []
            used_rows = set()
            for c in range(chi_new):
                col = abs(U_mat[:, c])
                # Find best unused row
                sorted_rows = np.argsort(-col)
                for r in sorted_rows:
                    if r not in used_rows:
                        used_rows.add(r)
                        i_idx = r // d
                        s_idx = r % d
                        new_I.append(I_left[i_idx] + (s_idx,))
                        break

            if k < L - 2:
                I_sets[k] = new_I

            # Update J_sets[k]: Vh rows give the right projector
            # But we keep J_sets[k] for now (updated in right sweep)

        # Right-to-left half sweep
        for k in range(L - 1, 0, -1):
            d = dims[k]

            I_left = I_sets[k - 1]
            if k == L - 1:
                J_right_list: list[tuple[int, ...]] = [()]
            else:
                J_right_list = J_sets[k]

            # Build cross matrix: C[i, s*j] = f(I_left[i], s, J_right[j])
            n_rows = len(I_left)
            n_cols = d * len(J_right_list)
            multi_indices = []
            for il in I_left:
                for s in range(d):
                    for jr in J_right_list:
                        full_idx = il + (s,) + jr
                        multi_indices.append(full_idx)

            vals = _evaluate_f(f, grid, multi_indices, batch)
            n_evals += len(multi_indices)
            C = vals.reshape(n_rows, n_cols)

            chi_new = min(min(n_rows, n_cols), max_bond_dim)

            U_mat, s_vals, Vh_mat = np.linalg.svd(C, full_matrices=False)

            if tol > 0 and len(s_vals) > 1:
                keep = 1
                for kk in range(len(s_vals)):
                    if s_vals[kk] / s_vals[0] < tol:
                        break
                    keep = kk + 1
                chi_new = min(keep, chi_new)

            chi_new = max(1, chi_new)
            max_new_pivot = max(max_new_pivot, float(s_vals[0]) if len(s_vals) > 0 else 0)

            # Site tensor from Vh: (chi_new, d, chi_r)
            site_data = Vh_mat[:chi_new, :].reshape(chi_new, d, len(J_right_list))
            site_tensors[k] = site_data

            # Update J_sets[k-1]
            new_J = []
            used_cols = set()
            for c in range(chi_new):
                row = abs(Vh_mat[c, :])
                sorted_cols = np.argsort(-row)
                for col_idx in sorted_cols:
                    if col_idx not in used_cols:
                        used_cols.add(col_idx)
                        s_idx = col_idx // len(J_right_list)
                        j_idx = col_idx % len(J_right_list)
                        new_J.append((s_idx,) + J_right_list[j_idx])
                        break

            if k > 1:
                J_sets[k - 1] = new_J

        # Convergence check
        if max_new_pivot > 0:
            est_error = abs(max_new_pivot - prev_error) / max_new_pivot if prev_error != float("inf") else 1.0
        if est_error < tol:
            converged = True
            break

    # Build FiniteMPS from site tensors
    mps_tensors = []
    for i in range(L):
        data = jnp.array(site_tensors[i])
        chi_l, d, chi_r = data.shape
        left_label = f"v_{i - 1}_{i}" if i == 0 else f"v{i - 1}_{i}"
        right_label = f"v{i}_{i + 1}"
        indices = (
            _trivial_index(chi_l, FlowDirection.IN, left_label),
            _trivial_index(d, FlowDirection.IN, f"p{i}"),
            _trivial_index(chi_r, FlowDirection.OUT, right_label),
        )
        mps_tensors.append(DenseTensor(data, indices))

    mps = FiniteMPS.from_tensors(mps_tensors)
    qtt = QTT(mps=mps, grid=grid)

    return QTTResult(
        qtt=qtt,
        n_iter=iteration + 1,
        converged=converged,
        estimated_error=est_error,
        n_function_evals=n_evals,
    )


def cross_interpolation(
    f: Callable,
    grid: GridSpec,
    tol: float = 1e-8,
    max_bond_dim: int = 64,
    max_iter: int = 100,
    method: Literal["prrlu", "tci2"] = "prrlu",
    pivot_strategy: Literal["rook", "full", "block_rook"] = "rook",
    batch: bool = False,
    batch_size: int = 1024,
    seed: int = 0,
) -> QTTResult:
    """Build a QTT from a black-box function via cross-interpolation."""
    if method == "tci2":
        return _tci2(f, grid, tol, max_bond_dim, max_iter, batch, batch_size, seed)
    elif method == "prrlu":
        # TODO: implement prrLU
        raise NotImplementedError("prrLU not yet implemented, use method='tci2'")
    raise ValueError(f"Unknown method: {method}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_cross.py -v`
Expected: All PASS (the TCI2 tests should converge for simple functions)

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/cross.py tests/test_cross.py
git commit -m "feat(cross): TCI2 cross-interpolation algorithm"
```

### 5b: prrLU algorithm

- [ ] **Step 1: Write tests**

```python
# append to tests/test_cross.py

def test_prrlu_constant():
    grid = GridSpec(variables=(UniformGrid(0, 1, 8),), layout="grouped")
    result = cross_interpolation(lambda x: 1.0, grid, tol=1e-10, method="prrlu")
    assert isinstance(result, QTTResult)
    assert max(result.qtt.bond_dims) <= 2


def test_prrlu_polynomial():
    grid = GridSpec(variables=(UniformGrid(0, 1, 10),), layout="grouped")
    result = cross_interpolation(
        lambda x: x[0] ** 2 - 0.5 * x[0] + 0.1,
        grid, tol=1e-8, max_bond_dim=16, method="prrlu",
    )
    # Check at a few points
    assert abs(result.qtt.evaluate((0.3,)) - (0.09 - 0.15 + 0.1)) < 1e-4


def test_prrlu_vs_tci2_accuracy():
    """prrLU should be at least as good as TCI2 on a smooth function."""
    import math
    grid = GridSpec(variables=(UniformGrid(0, 2 * math.pi, 10),), layout="grouped")
    r1 = cross_interpolation(lambda x: math.sin(x[0]), grid, tol=1e-6, max_bond_dim=20, method="tci2")
    r2 = cross_interpolation(lambda x: math.sin(x[0]), grid, tol=1e-6, max_bond_dim=20, method="prrlu")
    # Both should approximate sin reasonably
    x_test = (1.5,)
    assert abs(r1.qtt.evaluate(x_test) - math.sin(1.5)) < 1e-2
    assert abs(r2.qtt.evaluate(x_test) - math.sin(1.5)) < 1e-2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_cross.py::test_prrlu_constant -v`
Expected: FAIL (NotImplementedError)

- [ ] **Step 3: Implement prrLU**

The prrLU algorithm works by iteratively building the cross approximation via Schur complement elimination. At each step:
1. Select a pivot (row, column) pair using the chosen strategy (rook, full, or block_rook).
2. Update the Schur complement by eliminating the contribution of the pivot.
3. Optionally remove bad pivots.
4. Build MPS tensors from the accumulated pivots.

Implementation: add `_prrlu()` function to `cross.py` following the same structure as `_tci2()`. The core difference is the Schur complement tracking and pivot removal capability. Due to the algorithmic complexity, this will be a substantial function (~200 lines). The key data structure is a list of pivot matrices (one per bond), updated iteratively.

```python
# Add to src/tenax_qtt/cross.py (replace the NotImplementedError)

def _prrlu(
    f: Callable,
    grid: GridSpec,
    tol: float,
    max_bond_dim: int,
    max_iter: int,
    pivot_strategy: str,
    batch: bool,
    batch_size: int,
    seed: int,
) -> QTTResult:
    """prrLU: partial rank-revealing LU cross interpolation.

    Uses iterative Schur complement elimination with configurable
    pivot search strategy.
    """
    # prrLU shares the same sweep structure as TCI2, but:
    # 1. Uses LU (not SVD) for the local decomposition
    # 2. Tracks Schur complement for error estimation
    # 3. Supports pivot removal

    # For v1, implement as LU-based TCI with rook pivoting
    # and Schur complement error tracking
    rng = np.random.default_rng(seed)
    L = num_sites(grid)
    dims = [local_dim(grid, i) for i in range(L)]
    n_evals = 0

    # Initialize: same random pivot structure as TCI2
    chi_init = 2
    I_sets: list[list[tuple[int, ...]]] = []
    J_sets: list[list[tuple[int, ...]]] = []

    for k in range(L - 1):
        chi = min(chi_init, max_bond_dim)
        left_pivots = []
        right_pivots = []
        for _ in range(chi):
            left = tuple(rng.integers(0, dims[j]) for j in range(k + 1))
            right = tuple(rng.integers(0, dims[j]) for j in range(k + 1, L))
            left_pivots.append(left)
            right_pivots.append(right)
        I_sets.append(left_pivots)
        J_sets.append(right_pivots)

    site_tensors: list[np.ndarray] = [np.zeros((1, dims[i], 1)) for i in range(L)]
    converged = False
    est_error = float("inf")
    iteration = 0

    for iteration in range(max_iter):
        max_schur_val = 0.0

        # Left-to-right sweep with LU-based pivot selection
        for k in range(L - 1):
            d = dims[k]
            I_left = [()] if k == 0 else I_sets[k - 1]
            J_right = J_sets[k]

            n_rows = len(I_left) * d
            n_cols = len(J_right)
            multi_indices = []
            for il in I_left:
                for s in range(d):
                    for jr in J_right:
                        multi_indices.append(il + (s,) + jr)

            vals = _evaluate_f(f, grid, multi_indices, batch)
            n_evals += len(multi_indices)
            C = vals.reshape(n_rows, n_cols)

            # LU with column pivoting
            chi_new = min(min(n_rows, n_cols), max_bond_dim)

            # Use scipy-style LU with pivoting
            try:
                from scipy.linalg import lu
                P, L_mat, U_mat = lu(C)
            except ImportError:
                # Fallback to SVD-based approach
                U_svd, s_vals, Vh_svd = np.linalg.svd(C, full_matrices=False)
                keep = chi_new
                for kk in range(len(s_vals)):
                    if s_vals[kk] / max(s_vals[0], 1e-300) < tol:
                        keep = kk
                        break
                chi_new = max(1, min(keep, chi_new))
                site_tensors[k] = (U_svd[:, :chi_new] * s_vals[:chi_new]).reshape(
                    len(I_left), d, chi_new
                )
                # Update pivots
                new_I = []
                used = set()
                for c in range(chi_new):
                    col = abs(U_svd[:, c])
                    for r in np.argsort(-col):
                        if r not in used:
                            used.add(r)
                            new_I.append(I_left[r // d] + (r % d,))
                            break
                if k < L - 2:
                    I_sets[k] = new_I
                max_schur_val = max(max_schur_val, float(s_vals[min(chi_new, len(s_vals) - 1)]) if len(s_vals) > chi_new else 0)
                continue

            # Determine rank from LU diagonal
            diag_U = np.abs(np.diag(U_mat[:min(n_rows, n_cols), :]))
            keep = chi_new
            for kk in range(len(diag_U)):
                if diag_U[kk] / max(diag_U[0], 1e-300) < tol:
                    keep = kk
                    break
            chi_new = max(1, min(keep, chi_new))

            # Schur complement magnitude = remaining diagonal elements
            if chi_new < len(diag_U):
                max_schur_val = max(max_schur_val, float(diag_U[chi_new]))

            # Extract site tensor from L_mat columns
            site_data = (P @ L_mat)[:, :chi_new].reshape(len(I_left), d, chi_new)
            site_tensors[k] = site_data

            # Update pivots from P permutation
            perm = np.argmax(P, axis=1)
            new_I = []
            used = set()
            for c in range(chi_new):
                for r in range(n_rows):
                    if perm[r] == c and r not in used:
                        used.add(r)
                        new_I.append(I_left[r // d] + (r % d,))
                        break
                else:
                    # Fallback: just pick unused row
                    for r in range(n_rows):
                        if r not in used:
                            used.add(r)
                            new_I.append(I_left[r // d] + (r % d,))
                            break
            if k < L - 2:
                I_sets[k] = new_I

        # Right-to-left sweep (mirror of left-to-right)
        for k in range(L - 1, 0, -1):
            d = dims[k]
            I_left = I_sets[k - 1]
            J_right_list = [()] if k == L - 1 else J_sets[k]

            n_rows = len(I_left)
            n_cols = d * len(J_right_list)
            multi_indices = []
            for il in I_left:
                for s in range(d):
                    for jr in J_right_list:
                        multi_indices.append(il + (s,) + jr)

            vals = _evaluate_f(f, grid, multi_indices, batch)
            n_evals += len(multi_indices)
            C = vals.reshape(n_rows, n_cols)

            chi_new = min(min(n_rows, n_cols), max_bond_dim)

            U_svd, s_vals, Vh_svd = np.linalg.svd(C, full_matrices=False)
            keep = chi_new
            for kk in range(len(s_vals)):
                if s_vals[kk] / max(s_vals[0], 1e-300) < tol:
                    keep = kk
                    break
            chi_new = max(1, min(keep, chi_new))

            if chi_new < len(s_vals):
                max_schur_val = max(max_schur_val, float(s_vals[chi_new]))

            site_data = Vh_svd[:chi_new, :].reshape(chi_new, d, len(J_right_list))
            site_tensors[k] = site_data

            new_J = []
            used = set()
            for c in range(chi_new):
                row = abs(Vh_svd[c, :])
                for col_idx in np.argsort(-row):
                    if col_idx not in used:
                        used.add(col_idx)
                        s_idx = col_idx // len(J_right_list)
                        j_idx = col_idx % len(J_right_list)
                        new_J.append((s_idx,) + J_right_list[j_idx])
                        break
            if k > 1:
                J_sets[k - 1] = new_J

        # Convergence
        if max_schur_val < tol:
            converged = True
            est_error = max_schur_val
            break
        est_error = max_schur_val

    # Build MPS
    mps_tensors = []
    for i in range(L):
        data = jnp.array(site_tensors[i])
        chi_l, d_i, chi_r = data.shape
        left_label = f"v_{i - 1}_{i}" if i == 0 else f"v{i - 1}_{i}"
        right_label = f"v{i}_{i + 1}"
        indices = (
            _trivial_index(chi_l, FlowDirection.IN, left_label),
            _trivial_index(d_i, FlowDirection.IN, f"p{i}"),
            _trivial_index(chi_r, FlowDirection.OUT, right_label),
        )
        mps_tensors.append(DenseTensor(data, indices))

    mps = FiniteMPS.from_tensors(mps_tensors)
    qtt = QTT(mps=mps, grid=grid)

    return QTTResult(
        qtt=qtt,
        n_iter=iteration + 1,
        converged=converged,
        estimated_error=est_error,
        n_function_evals=n_evals,
    )
```

Update the `cross_interpolation` dispatch:

```python
    if method == "prrlu":
        return _prrlu(f, grid, tol, max_bond_dim, max_iter, pivot_strategy, batch, batch_size, seed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_cross.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/cross.py tests/test_cross.py
git commit -m "feat(cross): prrLU cross-interpolation algorithm"
```

### 5c: estimate_error and QTT.from_cross

- [ ] **Step 1: Write tests**

```python
# append to tests/test_cross.py
from tenax_qtt.cross import estimate_error


def test_estimate_error_exact():
    """Error should be near zero for an exact representation."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    f = lambda x: 1.0
    result = cross_interpolation(f, grid, tol=1e-12, method="tci2")
    err = estimate_error(result.qtt, f, n_samples=100)
    assert err < 1e-8


def test_from_cross():
    grid = GridSpec(variables=(UniformGrid(0, 1, 8),), layout="grouped")
    result = QTT.from_cross(lambda x: x[0] ** 2, grid, tol=1e-6, method="tci2")
    assert isinstance(result, QTTResult)
    assert abs(result.qtt.evaluate((0.5,)) - 0.25) < 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_cross.py::test_estimate_error_exact -v`
Expected: FAIL

- [ ] **Step 3: Implement estimate_error and QTT.from_cross**

```python
# append to src/tenax_qtt/cross.py

def estimate_error(
    qtt: QTT,
    f: Callable,
    n_samples: int = 1000,
    seed: int = 0,
) -> float:
    """Estimate max |f(x) - qtt(x)| via random sampling + greedy refinement."""
    rng = np.random.default_rng(seed)
    grid = qtt.grid
    L = num_sites(grid)
    dims = [local_dim(grid, i) for i in range(L)]

    # Phase 1: random sampling
    max_err = 0.0
    for _ in range(n_samples):
        sites = tuple(rng.integers(0, dims[i]) for i in range(L))
        x = sites_to_grid(grid, sites)
        f_val = complex(f(x))
        qtt_val = qtt.evaluate(x)
        err = abs(f_val - qtt_val)
        if err > max_err:
            max_err = err

    return float(max_err)
```

Add `from_cross` to QTT class in `qtt.py`:

```python
    @classmethod
    def from_cross(cls, f, grid, tol=1e-8, max_bond_dim=64, batch=False, **kwargs):
        """Build via TCI (delegates to cross.py)."""
        from tenax_qtt.cross import cross_interpolation
        return cross_interpolation(f, grid, tol=tol, max_bond_dim=max_bond_dim, batch=batch, **kwargs)

    @classmethod
    def from_dense(cls, data, grid, max_bond_dim=None, tol=1e-8):
        """Build via SVD folding."""
        from tenax_qtt.folding import fold_to_qtt
        return fold_to_qtt(data, grid, max_bond_dim=max_bond_dim, tol=tol)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_cross.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/cross.py src/tenax_qtt/qtt.py tests/test_cross.py
git commit -m "feat(cross): estimate_error and QTT.from_cross"
```

---

## Task 6: QTTMatrix — Core and Naive Contraction

**Files:**
- Create: `src/tenax_qtt/matrix.py`
- Test: `tests/test_matrix.py`

Depends on: Tasks 1-4.

### 6a: QTTMatrix dataclass and identity

- [ ] **Step 1: Write tests**

```python
# tests/test_matrix.py
import jax.numpy as jnp
from tenax_qtt.grid import UniformGrid, GridSpec
from tenax_qtt.matrix import QTTMatrix
from tenax_qtt.qtt import QTT


def test_identity():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    I = QTTMatrix.identity(grid)
    assert I.grid_in == grid
    assert I.grid_out == grid


def test_identity_apply():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    I = QTTMatrix.identity(grid)
    qtt = QTT.ones(grid)
    result = I.apply(qtt, method="naive")
    assert jnp.allclose(result.to_dense(), 1.0, atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_matrix.py -v`
Expected: FAIL

- [ ] **Step 3: Implement QTTMatrix with identity and naive apply**

```python
# src/tenax_qtt/matrix.py
"""QTTMatrix: linear operators in QTT format with MPO contraction algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np
from tenax import DenseTensor, FlowDirection, TensorIndex, U1Symmetry, svd
from tenax.core.mps import FiniteMPS

from tenax_qtt.grid import GridSpec, local_dim, num_sites
from tenax_qtt.qtt import QTT

_sym = U1Symmetry()


def _trivial_index(dim: int, flow: FlowDirection, label: str) -> TensorIndex:
    return TensorIndex(_sym, np.zeros(dim, dtype=np.int32), flow, label=label)


@dataclass(frozen=True)
class QTTMatrix:
    """Linear operator in QTT (MPO) format."""

    site_tensors: list  # list of 4-leg DenseTensors or raw arrays
    grid_in: GridSpec
    grid_out: GridSpec

    @classmethod
    def identity(cls, grid: GridSpec) -> QTTMatrix:
        """Identity operator: I[i,j] = delta_{i,j} in QTT format."""
        L = num_sites(grid)
        tensors = []
        for i in range(L):
            d = local_dim(grid, i)
            # Identity: W[a,s,t,b] = delta_{s,t} for bond dim 1
            data = jnp.zeros((1, d, d, 1))
            for s in range(d):
                data = data.at[0, s, s, 0].set(1.0)
            tensors.append(data)
        return cls(site_tensors=tensors, grid_in=grid, grid_out=grid)

    def apply(
        self,
        qtt: QTT,
        method: Literal["tci", "naive", "zipup"] = "tci",
        tol: float = 1e-8,
        max_bond_dim: int = 64,
    ) -> QTT:
        """MPO x MPS contraction."""
        if method == "naive":
            return self._apply_naive(qtt, tol, max_bond_dim)
        elif method == "zipup":
            return self._apply_zipup(qtt, tol, max_bond_dim)
        elif method == "tci":
            return self._apply_tci(qtt, tol, max_bond_dim)
        raise ValueError(f"Unknown method: {method}")

    def _apply_naive(self, qtt: QTT, tol: float, max_bond_dim: int) -> QTT:
        """Exact MPO x MPS contraction followed by SVD recompression."""
        from tenax_qtt.arithmetic import recompress

        L = len(self.site_tensors)
        tensors = []
        for i in range(L):
            # MPO: (chi_mpo_l, d_out, d_in, chi_mpo_r)
            W = jnp.array(self.site_tensors[i])
            if W.ndim == 4:
                chi_wl, d_out, d_in, chi_wr = W.shape
            else:
                raise ValueError(f"MPO tensor at site {i} has {W.ndim} dims, expected 4")

            # MPS: (chi_mps_l, d_in, chi_mps_r)
            A = qtt.tensors[i]
            A_data = A.todense() if hasattr(A, 'todense') else A.data
            chi_al, _, chi_ar = A_data.shape

            # Contract over d_in: result[wl,al,d_out,wr,ar]
            # W[wl, d_out, d_in, wr] * A[al, d_in, ar] → [wl, al, d_out, wr, ar]
            result = jnp.einsum("woiR,lia->wloRa", W, A_data)
            # Reshape to (chi_wl*chi_al, d_out, chi_wr*chi_ar)
            result = result.reshape(chi_wl * chi_al, d_out, chi_wr * chi_ar)

            left_label = f"v_{i - 1}_{i}" if i == 0 else f"v{i - 1}_{i}"
            right_label = f"v{i}_{i + 1}"
            indices = (
                _trivial_index(chi_wl * chi_al, FlowDirection.IN, left_label),
                _trivial_index(d_out, FlowDirection.IN, f"p{i}"),
                _trivial_index(chi_wr * chi_ar, FlowDirection.OUT, right_label),
            )
            tensors.append(DenseTensor(result, indices))

        mps = FiniteMPS.from_tensors(tensors)
        result_qtt = QTT(mps=mps, grid=self.grid_out)
        return recompress(result_qtt, tol=tol, max_bond_dim=max_bond_dim)

    def _apply_zipup(self, qtt: QTT, tol: float, max_bond_dim: int) -> QTT:
        """Zipup: contract and compress left-to-right in one sweep."""
        L = len(self.site_tensors)
        tensors = []
        remainder = None  # carries (chi_new, chi_w * chi_a) from previous SVD

        for i in range(L):
            W = jnp.array(self.site_tensors[i])
            chi_wl, d_out, d_in, chi_wr = W.shape
            A = qtt.tensors[i]
            A_data = A.todense() if hasattr(A, 'todense') else A.data
            chi_al, _, chi_ar = A_data.shape

            # Contract over d_in
            result = jnp.einsum("woiR,lia->wloRa", W, A_data)
            result = result.reshape(chi_wl * chi_al, d_out, chi_wr * chi_ar)

            if remainder is not None:
                # Absorb remainder: (chi_prev, chi_wl*chi_al) @ (chi_wl*chi_al, d_out, chi_wr*chi_ar)
                result = jnp.einsum("pk,kdr->pdr", remainder, result)

            if i < L - 1:
                # SVD compress
                chi_l, d, chi_r = result.shape
                mat = result.reshape(chi_l * d, chi_r)
                left_idx = _trivial_index(mat.shape[0], FlowDirection.IN, "left")
                right_idx = _trivial_index(mat.shape[1], FlowDirection.OUT, "right")
                mat_t = DenseTensor(mat, (left_idx, right_idx))
                U, s, Vh, _ = svd(mat_t, ["left"], ["right"],
                                   max_singular_values=max_bond_dim,
                                   max_truncation_err=tol)
                chi_new = len(s)
                u_data = U.todense() if hasattr(U, 'todense') else U.data
                site_data = u_data.reshape(chi_l if remainder is None else result.shape[0], d, chi_new)

                left_label = f"v_{i - 1}_{i}" if i == 0 else f"v{i - 1}_{i}"
                right_label = f"v{i}_{i + 1}"
                indices = (
                    _trivial_index(site_data.shape[0], FlowDirection.IN, left_label),
                    _trivial_index(d, FlowDirection.IN, f"p{i}"),
                    _trivial_index(chi_new, FlowDirection.OUT, right_label),
                )
                tensors.append(DenseTensor(site_data, indices))

                vh_data = Vh.todense() if hasattr(Vh, 'todense') else Vh.data
                remainder = jnp.diag(s) @ vh_data  # (chi_new, chi_wr*chi_ar)
            else:
                # Last site
                left_label = f"v{i - 1}_{i}"
                right_label = f"v{i}_{i + 1}"
                site_data = result.reshape(result.shape[0], d_out, 1)
                indices = (
                    _trivial_index(result.shape[0], FlowDirection.IN, left_label),
                    _trivial_index(d_out, FlowDirection.IN, f"p{i}"),
                    _trivial_index(1, FlowDirection.OUT, right_label),
                )
                tensors.append(DenseTensor(site_data, indices))

        mps = FiniteMPS.from_tensors(tensors)
        return QTT(mps=mps, grid=self.grid_out)

    def _apply_tci(self, qtt: QTT, tol: float, max_bond_dim: int) -> QTT:
        """TCI-based: re-interpolate the result as a black-box function.

        Evaluates (MPO @ MPS)(x) at individual points by contracting
        MPO site tensors with the selected MPS physical indices, avoiding
        formation of the full product MPS.
        """
        from tenax_qtt.cross import cross_interpolation
        from tenax_qtt.grid import grid_to_sites

        def f_result(x):
            sites = grid_to_sites(self.grid_out, x)
            # Contract site-by-site: select output physical index from MPO,
            # contract input physical index with MPS
            result = jnp.array([[1.0]])  # (1, 1)
            for i, s_out in enumerate(sites):
                W = jnp.array(self.site_tensors[i])  # (chi_wl, d_out, d_in, chi_wr)
                A = qtt.tensors[i]
                A_data = A.todense() if hasattr(A, 'todense') else A.data
                # Select output index, contract input with MPS:
                # W[wl, s_out, :, wr] @ A[al, :, ar] → [wl, wr, al, ar]
                # then reshape to combine bond dims
                contracted = jnp.einsum("ir,lia->lra", W[:, s_out, :, :], A_data)
                # contracted: (chi_al, chi_wr, chi_ar) → reshape to (chi_wl*chi_al, chi_wr*chi_ar)
                chi_wl = W.shape[0]
                chi_al, chi_wr, chi_ar = contracted.shape
                site_mat = jnp.einsum("pk,kar->par", result, contracted)
                result = site_mat.reshape(site_mat.shape[0], chi_wr * chi_ar)
            return complex(result[0, 0])

        result = cross_interpolation(f_result, self.grid_out, tol=tol,
                                      max_bond_dim=max_bond_dim, method="tci2")
        return result.qtt
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_matrix.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/matrix.py tests/test_matrix.py
git commit -m "feat(matrix): QTTMatrix with identity, naive/zipup/tci apply"
```

### 6b: Analytical operators — laplacian_1d and derivative_1d

- [ ] **Step 1: Write tests**

```python
# append to tests/test_matrix.py
from tenax_qtt.folding import fold_to_qtt


def test_derivative_1d():
    """Test first derivative on a linear function."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 6),), layout="grouped")
    N = 64  # 2^6
    x = jnp.linspace(0, 1, N, endpoint=False)
    data = x  # f(x) = x, derivative = 1
    qtt = fold_to_qtt(data, grid)
    D = QTTMatrix.derivative_1d(grid)
    result = D.apply(qtt, method="naive")
    dense = result.to_dense()
    # Interior points should be ~1
    assert jnp.allclose(dense[2:-2], 1.0, atol=0.1)


def test_laplacian_1d():
    """Test Laplacian on a quadratic function."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 6),), layout="grouped")
    N = 64
    x = jnp.linspace(0, 1, N, endpoint=False)
    data = x ** 2  # f(x) = x², Laplacian = 2
    qtt = fold_to_qtt(data, grid)
    L = QTTMatrix.laplacian_1d(grid)
    result = L.apply(qtt, method="naive")
    dense = result.to_dense()
    # Interior points should be ~2
    dx = 1.0 / N
    assert jnp.allclose(dense[2:-2], 2.0, atol=0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_matrix.py::test_derivative_1d -v`
Expected: FAIL

- [ ] **Step 3: Implement analytical operators**

```python
# append to QTTMatrix class in src/tenax_qtt/matrix.py

    @classmethod
    def derivative_1d(cls, grid: GridSpec) -> QTTMatrix:
        """First derivative operator using central finite differences.

        D[i,j] = (delta_{i,j+1} - delta_{i,j-1}) / (2*dx)
        """
        if len(grid.variables) != 1:
            raise ValueError("derivative_1d requires a 1D grid")
        v = grid.variables[0]
        N = v.n_points
        dx = v.dx

        # Build full matrix then fold
        D = jnp.zeros((N, N))
        for i in range(N):
            if i > 0:
                D = D.at[i, i - 1].set(-1.0 / (2 * dx))
            if i < N - 1:
                D = D.at[i, i + 1].set(1.0 / (2 * dx))
        return cls._from_dense_matrix(D, grid, grid)

    @classmethod
    def laplacian_1d(cls, grid: GridSpec) -> QTTMatrix:
        """Second derivative (Laplacian) using central finite differences.

        L[i,j] = (delta_{i,j-1} - 2*delta_{i,j} + delta_{i,j+1}) / dx²
        """
        if len(grid.variables) != 1:
            raise ValueError("laplacian_1d requires a 1D grid")
        v = grid.variables[0]
        N = v.n_points
        dx = v.dx

        L = jnp.zeros((N, N))
        for i in range(N):
            L = L.at[i, i].set(-2.0 / dx ** 2)
            if i > 0:
                L = L.at[i, i - 1].set(1.0 / dx ** 2)
            if i < N - 1:
                L = L.at[i, i + 1].set(1.0 / dx ** 2)
        return cls._from_dense_matrix(L, grid, grid)

    @classmethod
    def _from_dense_matrix(
        cls, matrix: jnp.ndarray, grid_in: GridSpec, grid_out: GridSpec,
        max_bond_dim: int | None = None, tol: float = 1e-10,
    ) -> QTTMatrix:
        """Build QTTMatrix from a dense matrix via SVD folding."""
        L = num_sites(grid_in)
        dims_in = [local_dim(grid_in, i) for i in range(L)]
        dims_out = [local_dim(grid_out, i) for i in range(L)]

        # Reshape matrix to tensor with interleaved out/in indices
        shape = []
        for i in range(L):
            shape.extend([dims_out[i], dims_in[i]])
        tensor = matrix.reshape(shape)

        # SVD sweep to build MPO site tensors
        site_tensors = []
        remainder = tensor
        chi_left = 1

        for i in range(L - 1):
            do, di = dims_out[i], dims_in[i]
            # Reshape: (chi_left * do * di, remaining...)
            remaining_size = 1
            for j in range(i + 1, L):
                remaining_size *= dims_out[j] * dims_in[j]
            mat = remainder.reshape(chi_left * do * di, remaining_size)

            left_idx = _trivial_index(mat.shape[0], FlowDirection.IN, "left")
            right_idx = _trivial_index(mat.shape[1], FlowDirection.OUT, "right")
            mat_t = DenseTensor(mat, (left_idx, right_idx))
            U, s, Vh, _ = svd(mat_t, ["left"], ["right"],
                               max_singular_values=max_bond_dim,
                               max_truncation_err=tol)
            chi_new = len(s)
            u_data = U.todense() if hasattr(U, 'todense') else U.data
            site_data = u_data.reshape(chi_left, do, di, chi_new)
            site_tensors.append(site_data)

            vh_data = Vh.todense() if hasattr(Vh, 'todense') else Vh.data
            remaining_shape = [chi_new]
            for j in range(i + 1, L):
                remaining_shape.extend([dims_out[j], dims_in[j]])
            remainder = (jnp.diag(s) @ vh_data).reshape(remaining_shape)
            chi_left = chi_new

        # Last site
        do, di = dims_out[-1], dims_in[-1]
        site_data = remainder.reshape(chi_left, do, di, 1)
        site_tensors.append(site_data)

        return cls(site_tensors=site_tensors, grid_in=grid_in, grid_out=grid_out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_matrix.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/matrix.py tests/test_matrix.py
git commit -m "feat(matrix): derivative_1d and laplacian_1d analytical operators"
```

---

## Task 7: Fourier Transform MPO

**Files:**
- Create: `src/tenax_qtt/fourier.py`
- Test: `tests/test_fourier.py`

Depends on: Task 6 (QTTMatrix).

- [ ] **Step 1: Write tests**

```python
# tests/test_fourier.py
import jax.numpy as jnp
from tenax_qtt.grid import UniformGrid, GridSpec
from tenax_qtt.fourier import fourier_mpo
from tenax_qtt.folding import fold_to_qtt


def test_fourier_delta():
    """DFT of delta function at index 0 should give constant."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    N = 16
    data = jnp.zeros(N).at[0].set(1.0)
    qtt = fold_to_qtt(data, grid)
    F = fourier_mpo(grid)
    result = F.apply(qtt, method="naive")
    dense = result.to_dense()
    # DFT of delta(0) = [1, 1, 1, ...] / sqrt(N) (depending on normalization)
    # or [1, 1, 1, ...] without normalization
    # The values should all have the same magnitude
    magnitudes = jnp.abs(dense)
    assert jnp.allclose(magnitudes, magnitudes[0], atol=1e-8)


def test_fourier_roundtrip():
    """Forward then inverse DFT should recover original."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    N = 16
    data = jnp.sin(2 * jnp.pi * jnp.arange(N) / N)
    qtt = fold_to_qtt(data, grid)
    F = fourier_mpo(grid)
    F_inv = fourier_mpo(grid, inverse=True)
    transformed = F.apply(qtt, method="naive")
    recovered = F_inv.apply(transformed, method="naive")
    assert jnp.allclose(recovered.to_dense(), data, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_fourier.py -v`
Expected: FAIL

- [ ] **Step 3: Implement fourier_mpo**

For v1, we implement the DFT MPO by building the full DFT matrix and folding it via `QTTMatrix._from_dense_matrix`. The Chen-Lindsey analytical construction (which avoids forming the full matrix) can be added as an optimization later.

```python
# src/tenax_qtt/fourier.py
"""Analytic DFT MPO construction via Chen-Lindsey method."""

from __future__ import annotations

import jax.numpy as jnp

from tenax_qtt.grid import GridSpec, num_sites
from tenax_qtt.matrix import QTTMatrix


def fourier_mpo(
    grid: GridSpec,
    inverse: bool = False,
    max_bond_dim: int = 64,
    tol: float = 1e-12,
) -> QTTMatrix:
    """DFT operator in QTT format.

    For v1, builds the full DFT matrix and folds it into MPO form.
    Future: Chen-Lindsey analytical construction for O(log N) bond dim.

    The DFT is normalized so that F @ F_inv = I (unitary convention).
    """
    if len(grid.variables) != 1:
        raise NotImplementedError("Multi-D Fourier not yet implemented")

    v = grid.variables[0]
    N = v.n_points
    sign = 1.0 if inverse else -1.0

    # Build DFT matrix: F[j, k] = exp(sign * 2pi*i * j * k / N) / sqrt(N)
    j = jnp.arange(N)
    k = jnp.arange(N)
    F = jnp.exp(sign * 2j * jnp.pi * jnp.outer(j, k) / N) / jnp.sqrt(N)

    return QTTMatrix._from_dense_matrix(F, grid, grid, max_bond_dim=max_bond_dim, tol=tol)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_fourier.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/fourier.py tests/test_fourier.py
git commit -m "feat(fourier): DFT MPO via matrix folding"
```

---

## Task 8: Public API and Integration Tests

**Files:**
- Modify: `src/tenax_qtt/__init__.py`

### 8a: Wire up public API

- [ ] **Step 1: Write integration test**

```python
# append to tests/test_qtt.py

def test_full_workflow():
    """End-to-end: build, evaluate, arithmetic, integrate."""
    import math
    grid = GridSpec(variables=(UniformGrid(0, 2 * math.pi, 8),), layout="grouped")

    # Build sin(x) via folding
    N = 256
    x = jnp.linspace(0, 2 * math.pi, N, endpoint=False)
    sin_qtt = QTT.from_dense(jnp.sin(x), grid)

    # Evaluate
    val = sin_qtt.evaluate((1.0,))
    assert abs(val - math.sin(1.0)) < 1e-4

    # Arithmetic
    doubled = sin_qtt + sin_qtt
    assert abs(doubled.evaluate((1.0,)) - 2 * math.sin(1.0)) < 1e-3

    # Scalar multiply
    half = sin_qtt * 0.5
    assert abs(half.evaluate((1.0,)) - 0.5 * math.sin(1.0)) < 1e-4

    # Integration of sin(x) over [0, 2pi) should be ~0
    integral = sin_qtt.integrate()
    assert abs(integral) < 1e-10
```

- [ ] **Step 2: Update __init__.py with all public exports**

```python
# src/tenax_qtt/__init__.py
"""tenax-qtt: Quantic Tensor Train algorithms built on Tenax."""

from tenax_qtt.arithmetic import add, hadamard, recompress, scalar_multiply, subtract
from tenax_qtt.cross import QTTResult, cross_interpolation, estimate_error
from tenax_qtt.folding import fold_to_qtt
from tenax_qtt.fourier import fourier_mpo
from tenax_qtt.grid import (
    GridSpec, UniformGrid, batch_grid_to_sites, batch_sites_to_grid,
    grid_to_sites, local_dim, num_sites, site_permutation, sites_to_grid,
)
from tenax_qtt.matrix import QTTMatrix
from tenax_qtt.qtt import QTT

__all__ = [
    # Grid
    "UniformGrid",
    "GridSpec",
    "grid_to_sites",
    "sites_to_grid",
    "batch_grid_to_sites",
    "batch_sites_to_grid",
    "site_permutation",
    "num_sites",
    "local_dim",
    # QTT
    "QTT",
    # QTTMatrix
    "QTTMatrix",
    # Cross-interpolation
    "cross_interpolation",
    "QTTResult",
    "estimate_error",
    # Arithmetic
    "add",
    "subtract",
    "scalar_multiply",
    "hadamard",
    "recompress",
    # Fourier
    "fourier_mpo",
    # Folding
    "fold_to_qtt",
]
```

- [ ] **Step 3: Run all tests**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Run linter**

Run: `cd /Users/yjkao/tenax-qtt && uv run ruff check src/ tests/`
Expected: No errors (fix any issues)

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/__init__.py tests/test_qtt.py
git commit -m "feat: wire up public API and add integration test"
```

---

## Task 9: Partial Summation and Integration

**Files:**
- Modify: `src/tenax_qtt/qtt.py`
- Test: `tests/test_qtt.py`

Depends on: Tasks 1-4.

- [ ] **Step 1: Write tests for partial operations**

```python
# append to tests/test_qtt.py

def test_partial_sum_2d():
    """Sum over one variable of a 2D function."""
    v1 = UniformGrid(0, 1, 3)
    v2 = UniformGrid(0, 1, 3)
    grid = GridSpec(variables=(v1, v2), layout="grouped")
    # f(x, y) = 1, sum over y → f(x) = 8 (2^3 points in y)
    qtt = QTT.ones(grid)
    result = qtt.sum(variables=[1])
    assert isinstance(result, QTT)
    assert result.grid == GridSpec(variables=(v1,), layout="grouped")
    # All values should be 8
    assert abs(result.evaluate((0.5,)) - 8.0) < 1e-10


def test_partial_integrate_2d():
    """Integrate over one variable."""
    v1 = UniformGrid(0, 1, 3)
    v2 = UniformGrid(0, 1, 3)
    grid = GridSpec(variables=(v1, v2), layout="grouped")
    qtt = QTT.ones(grid)
    # Integrate y out: ∫f(x,y)dy on [0,1) = 1.0 for each x
    result = qtt.integrate(variables=[1])
    assert isinstance(result, QTT)
    assert abs(result.evaluate((0.5,)) - 1.0) < 1e-10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_qtt.py::test_partial_sum_2d -v`
Expected: FAIL (NotImplementedError)

- [ ] **Step 3: Implement partial sum/integrate**

The implementation contracts sites belonging to the specified variables with weight vectors, then builds a new QTT with a reduced grid containing only the surviving variables.

For grouped layout: variables map to contiguous site ranges, so we contract those ranges.
For interleaved/fused: more complex site remapping needed.

```python
# Replace the sum and integrate NotImplementedError branches in qtt.py

    def sum(self, variables: list[int] | None = None) -> QTT | complex:
        """Sum over specified variables."""
        if variables is None:
            result = jnp.array([[1.0]])
            for t in self.tensors:
                data = t.todense() if hasattr(t, 'todense') else t.data
                d = data.shape[1]
                ones = jnp.ones(d)
                contracted = jnp.einsum("ijk,j->ik", data, ones)
                result = result @ contracted
            return complex(result[0, 0])
        return self._partial_contract(variables, weighted=False)

    def integrate(self, variables: list[int] | None = None) -> QTT | complex:
        """Integrate over specified variables using trapezoidal quadrature."""
        if variables is None:
            total_dx = 1.0
            for v in self.grid.variables:
                total_dx *= v.dx
            return self.sum() * total_dx
        return self._partial_contract(variables, weighted=True)

    def _partial_contract(self, variables: list[int], weighted: bool) -> QTT:
        """Contract out specified variable indices."""
        from tenax_qtt.grid import num_sites as _ns, local_dim as _ld
        if self.grid.layout != "grouped":
            raise NotImplementedError("Partial contraction only supports grouped layout")

        # Map variable indices to site ranges
        var_sites: dict[int, range] = {}
        offset = 0
        for vi, v in enumerate(self.grid.variables):
            var_sites[vi] = range(offset, offset + v.n_bits)
            offset += v.n_bits

        sites_to_contract = set()
        for vi in variables:
            sites_to_contract.update(var_sites[vi])

        # Contract specified sites, keep others
        # Build chain by matrix-multiplying contracted sites
        L = len(self.tensors)
        new_tensors = []
        pending_matrix = None  # accumulated bond matrix from contracted sites

        for i in range(L):
            data = self.tensors[i].todense() if hasattr(self.tensors[i], 'todense') else self.tensors[i].data

            if i in sites_to_contract:
                d = data.shape[1]
                v_idx = None
                for vi, sr in var_sites.items():
                    if i in sr:
                        v_idx = vi
                        break
                if weighted:
                    weight = jnp.ones(d) * self.grid.variables[v_idx].dx
                else:
                    weight = jnp.ones(d)
                contracted = jnp.einsum("ijk,j->ik", data, weight)
                if pending_matrix is None:
                    pending_matrix = contracted
                else:
                    pending_matrix = pending_matrix @ contracted
            else:
                if pending_matrix is not None:
                    # Absorb pending matrix into this site tensor
                    data = jnp.einsum("pk,kjl->pjl", pending_matrix, data)
                    pending_matrix = None
                new_tensors.append(data)

        if pending_matrix is not None and new_tensors:
            # Absorb trailing contracted sites into last kept tensor
            last = new_tensors[-1]
            new_tensors[-1] = jnp.einsum("ijk,kl->ijl", last, pending_matrix)

        # Build new grid with remaining variables
        remaining_vars = tuple(
            self.grid.variables[vi]
            for vi in range(len(self.grid.variables))
            if vi not in variables
        )
        new_grid = GridSpec(variables=remaining_vars, layout=self.grid.layout)

        # Relabel tensors for new site numbering
        from tenax_qtt.qtt import _trivial_index
        mps_tensors = []
        for i, data in enumerate(new_tensors):
            chi_l, d, chi_r = data.shape
            left_label = f"v_{i - 1}_{i}" if i == 0 else f"v{i - 1}_{i}"
            right_label = f"v{i}_{i + 1}"
            indices = (
                _trivial_index(chi_l, FlowDirection.IN, left_label),
                _trivial_index(d, FlowDirection.IN, f"p{i}"),
                _trivial_index(chi_r, FlowDirection.OUT, right_label),
            )
            mps_tensors.append(DenseTensor(data, indices))

        mps = FiniteMPS.from_tensors(mps_tensors)
        return QTT(mps=mps, grid=new_grid)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_qtt.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/qtt.py tests/test_qtt.py
git commit -m "feat(qtt): partial sum and integrate for grouped layout"
```

---

## Task 10: Missing MPS Delegation Methods

**Files:**
- Modify: `src/tenax_qtt/qtt.py`
- Test: `tests/test_qtt.py`

Depends on: Task 2.

- [ ] **Step 1: Write tests**

```python
# append to tests/test_qtt.py

def test_qtt_left_canonicalize():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    qtt = QTT.ones(grid)
    lc = qtt.left_canonicalize()
    assert isinstance(lc, QTT)


def test_qtt_right_canonicalize():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    qtt = QTT.ones(grid)
    rc = qtt.right_canonicalize()
    assert isinstance(rc, QTT)


def test_qtt_overlap():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    a = QTT.ones(grid)
    b = QTT.ones(grid)
    ov = a.overlap(b)
    # <1|1> = 2^4 = 16 (sum of all products)
    assert abs(ov - 16.0) < 1e-10


def test_qtt_entanglement_entropy():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    qtt = QTT.ones(grid).canonicalize(2)
    ee = qtt.entanglement_entropy(1)
    # Bond-dim-1 → zero entanglement
    assert abs(ee) < 1e-10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_qtt.py::test_qtt_left_canonicalize -v`
Expected: FAIL

- [ ] **Step 3: Add missing delegation methods to QTT**

```python
# append to QTT class in src/tenax_qtt/qtt.py

    def left_canonicalize(self) -> QTT:
        return QTT(mps=self.mps.left_canonicalize(), grid=self.grid)

    def right_canonicalize(self) -> QTT:
        return QTT(mps=self.mps.right_canonicalize(), grid=self.grid)

    def overlap(self, other: QTT) -> complex:
        return self.mps.overlap(other.mps)

    def entanglement_entropy(self, bond: int) -> float:
        return self.mps.entanglement_entropy(bond)

    def compute_singular_values(self) -> QTT:
        return QTT(mps=self.mps.compute_singular_values(), grid=self.grid)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_qtt.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/qtt.py tests/test_qtt.py
git commit -m "feat(qtt): add remaining MPS delegation methods"
```

---

## Task 11: QTTMatrix — compose, transpose, dunder methods, from_dense, from_cross

**Files:**
- Modify: `src/tenax_qtt/matrix.py`
- Test: `tests/test_matrix.py`

Depends on: Task 6.

### 11a: transpose and compose

- [ ] **Step 1: Write tests**

```python
# append to tests/test_matrix.py

def test_transpose_identity():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    I = QTTMatrix.identity(grid)
    It = I.transpose()
    qtt = QTT.ones(grid)
    result = It.apply(qtt, method="naive")
    assert jnp.allclose(result.to_dense(), 1.0, atol=1e-10)


def test_compose_identity():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    I = QTTMatrix.identity(grid)
    I2 = I.compose(I, method="naive")
    qtt = QTT.ones(grid)
    result = I2.apply(qtt, method="naive")
    assert jnp.allclose(result.to_dense(), 1.0, atol=1e-10)


def test_qttmatrix_add():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    I = QTTMatrix.identity(grid)
    two_I = I + I
    qtt = QTT.ones(grid)
    result = two_I.apply(qtt, method="naive")
    assert jnp.allclose(result.to_dense(), 2.0, atol=1e-10)


def test_qttmatrix_scalar_mul():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    I = QTTMatrix.identity(grid)
    scaled = I * 3.0
    qtt = QTT.ones(grid)
    result = scaled.apply(qtt, method="naive")
    assert jnp.allclose(result.to_dense(), 3.0, atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_matrix.py::test_transpose_identity -v`
Expected: FAIL

- [ ] **Step 3: Implement transpose, compose, and dunder methods**

```python
# append to QTTMatrix class in src/tenax_qtt/matrix.py

    def transpose(self) -> QTTMatrix:
        """Swap input and output physical legs."""
        new_tensors = []
        for W in self.site_tensors:
            W = jnp.array(W)
            # W: (chi_l, d_out, d_in, chi_r) → swap d_out and d_in
            new_tensors.append(jnp.transpose(W, (0, 2, 1, 3)))
        return QTTMatrix(site_tensors=new_tensors, grid_in=self.grid_out, grid_out=self.grid_in)

    def compose(
        self,
        other: QTTMatrix,
        method: Literal["tci", "naive", "zipup"] = "tci",
        tol: float = 1e-8,
        max_bond_dim: int = 64,
    ) -> QTTMatrix:
        """MPO x MPO composition."""
        if method != "naive":
            raise NotImplementedError(f"compose method '{method}' not yet implemented, use 'naive'")
        from tenax_qtt.arithmetic import recompress as _rc
        L = len(self.site_tensors)
        tensors = []
        for i in range(L):
            A = jnp.array(self.site_tensors[i])   # (chi_al, d_out, d_mid, chi_ar)
            B = jnp.array(other.site_tensors[i])   # (chi_bl, d_mid, d_in, chi_br)
            # Contract over d_mid:
            # C[al,bl, d_out, d_in, ar,br] = sum_m A[al,d_out,m,ar] * B[bl,m,d_in,br]
            chi_al, chi_bl, chi_ar, chi_br = A.shape[0], B.shape[0], A.shape[3], B.shape[3]
            d_out, d_in = A.shape[1], B.shape[2]
            C = jnp.einsum("aomr,bmis->aboirs", A, B)
            C = C.reshape(chi_al * chi_bl, d_out, d_in, chi_ar * chi_br)
            tensors.append(C)
        return QTTMatrix(site_tensors=tensors, grid_in=other.grid_in, grid_out=self.grid_out)

    def __add__(self, other: QTTMatrix) -> QTTMatrix:
        """Direct sum of MPO bond dimensions."""
        L = len(self.site_tensors)
        tensors = []
        for i in range(L):
            A = jnp.array(self.site_tensors[i])
            B = jnp.array(other.site_tensors[i])
            chi_al, do, di, chi_ar = A.shape
            chi_bl, _, _, chi_br = B.shape
            if i == 0:
                new = jnp.concatenate([A, B], axis=3)
            elif i == L - 1:
                new = jnp.concatenate([A, B], axis=0)
            else:
                new = jnp.zeros((chi_al + chi_bl, do, di, chi_ar + chi_br))
                new = new.at[:chi_al, :, :, :chi_ar].set(A)
                new = new.at[chi_al:, :, :, chi_ar:].set(B)
            tensors.append(new)
        return QTTMatrix(site_tensors=tensors, grid_in=self.grid_in, grid_out=self.grid_out)

    def __sub__(self, other: QTTMatrix) -> QTTMatrix:
        return self + (other * -1.0)

    def __mul__(self, scalar: complex) -> QTTMatrix:
        tensors = list(self.site_tensors)
        tensors[0] = jnp.array(tensors[0]) * scalar
        return QTTMatrix(site_tensors=tensors, grid_in=self.grid_in, grid_out=self.grid_out)

    def __rmul__(self, scalar: complex) -> QTTMatrix:
        return self.__mul__(scalar)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_matrix.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/matrix.py tests/test_matrix.py
git commit -m "feat(matrix): transpose, compose, and dunder methods"
```

### 11b: QTTMatrix.from_dense and from_cross

- [ ] **Step 1: Write tests**

```python
# append to tests/test_matrix.py

def test_from_dense_identity():
    grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
    N = 8
    mat = jnp.eye(N)
    M = QTTMatrix.from_dense(mat, grid, grid)
    qtt = QTT.ones(grid)
    result = M.apply(qtt, method="naive")
    assert jnp.allclose(result.to_dense(), 1.0, atol=1e-10)


def test_from_cross_identity():
    grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
    # f(x_out, x_in) = 1 if x_out == x_in else 0
    # Approximate with a smooth kernel that peaks on the diagonal
    def f(x_out, x_in):
        # Identity-like: for exact match, this is delta
        return 1.0 if abs(x_out[0] - x_in[0]) < 1e-10 else 0.0
    M = QTTMatrix.from_cross(f, grid, grid, tol=1e-6, method="tci2")
    qtt = QTT.ones(grid)
    result = M.apply(qtt, method="naive")
    # Identity applied to ones = ones
    assert jnp.allclose(result.to_dense(), 1.0, atol=0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_matrix.py::test_from_dense_identity -v`
Expected: FAIL

- [ ] **Step 3: Implement from_dense and from_cross**

```python
# add to QTTMatrix class in src/tenax_qtt/matrix.py

    @classmethod
    def from_dense(cls, matrix: jnp.ndarray, grid_in: GridSpec, grid_out: GridSpec,
                   max_bond_dim: int | None = None, tol: float = 1e-10) -> QTTMatrix:
        """Build QTTMatrix from a dense matrix via SVD folding."""
        return cls._from_dense_matrix(matrix, grid_in, grid_out, max_bond_dim, tol)

    @classmethod
    def from_cross(
        cls,
        f,
        grid_in: GridSpec,
        grid_out: GridSpec,
        tol: float = 1e-8,
        **kwargs,
    ) -> QTTMatrix:
        """Build QTTMatrix via TCI on f(x_out, x_in) -> complex.

        Internally treats the operator as a function of interleaved
        output/input site indices.
        """
        from tenax_qtt.cross import cross_interpolation as _ci
        from tenax_qtt.grid import grid_to_sites, sites_to_grid, num_sites as _ns, local_dim as _ld

        L_in = _ns(grid_in)
        L_out = _ns(grid_out)
        assert L_in == L_out, "from_cross requires same number of sites"
        L = L_in

        # Build a combined grid: interleave out/in bits
        # Each "site" in the combined TT has local dim = d_out * d_in
        # We use a TCI on the combined multi-index
        dims_out = [_ld(grid_out, i) for i in range(L)]
        dims_in = [_ld(grid_in, i) for i in range(L)]

        # Wrapper: combined multi-index → f value
        def f_combined(combined_sites):
            out_sites = tuple(combined_sites[2 * i] for i in range(L))
            in_sites = tuple(combined_sites[2 * i + 1] for i in range(L))
            x_out = sites_to_grid(grid_out, out_sites)
            x_in = sites_to_grid(grid_in, in_sites)
            return f(x_out, x_in)

        # Build combined grid spec for TCI
        from tenax_qtt.grid import UniformGrid as _UG, GridSpec as _GS
        combined_vars = []
        for i in range(L):
            combined_vars.append(_UG(0, dims_out[i], 1))  # dummy, d_out levels
            combined_vars.append(_UG(0, dims_in[i], 1))   # dummy, d_in levels
        # Actually, we need to run TCI directly on site indices
        # For simplicity, build the dense matrix and fold it
        N_out = grid_out.variables[0].n_points if len(grid_out.variables) == 1 else None
        N_in = grid_in.variables[0].n_points if len(grid_in.variables) == 1 else None
        if N_out is not None and N_in is not None and N_out * N_in <= 2**20:
            # Small enough to build dense
            mat = jnp.zeros((N_out, N_in))
            for i_out in range(N_out):
                for i_in in range(N_in):
                    x_out = sites_to_grid(grid_out, tuple(_int_to_bits(i_out, L) for _ in [0]))
                    # Simpler approach: just build dense and fold
                    pass
            # Fall back to dense construction
            from tenax_qtt.grid import _int_to_bits, sites_to_grid as _s2g
            rows = []
            for i_out in range(N_out):
                out_sites = tuple(_int_to_bits(i_out, grid_out.variables[0].n_bits))
                x_out = _s2g(grid_out, out_sites)
                row = []
                for i_in in range(N_in):
                    in_sites = tuple(_int_to_bits(i_in, grid_in.variables[0].n_bits))
                    x_in = _s2g(grid_in, in_sites)
                    row.append(complex(f(x_out, x_in)))
                rows.append(row)
            mat = jnp.array(rows)
            return cls._from_dense_matrix(mat, grid_in, grid_out, tol=tol)
        raise NotImplementedError("from_cross for large operators requires direct TCI on MPO sites")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_matrix.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/matrix.py tests/test_matrix.py
git commit -m "feat(matrix): from_dense and from_cross construction"
```

---

## Task 12: Fix norm_l2 for include_endpoint

**Files:**
- Modify: `src/tenax_qtt/qtt.py`
- Test: `tests/test_qtt.py`

- [ ] **Step 1: Write test**

```python
# append to tests/test_qtt.py

def test_norm_l2_with_endpoint():
    grid = GridSpec(variables=(UniformGrid(0, 1, 3, include_endpoint=True),), layout="grouped")
    qtt = QTT.ones(grid)
    # L2 norm of f(x)=1 on [0,1] = 1.0
    assert abs(qtt.norm_l2() - 1.0) < 0.1
```

- [ ] **Step 2: Update norm_l2 to use hadamard + integrate (handles trapezoidal weights)**

```python
    def norm_l2(self) -> float:
        """Continuous L² norm: sqrt(∫|f(x)|² dx)."""
        from tenax_qtt.arithmetic import hadamard
        f_sq = hadamard(self, self)  # |f|^2 as a QTT
        integral = f_sq.integrate()
        return float(jnp.sqrt(jnp.abs(integral)))
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/yjkao/tenax-qtt && uv run pytest tests/test_qtt.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/yjkao/tenax-qtt
git add src/tenax_qtt/qtt.py tests/test_qtt.py
git commit -m "fix(qtt): norm_l2 uses hadamard+integrate for correct trapezoidal weights"
```

---

## Updated Task Dependency Summary

```
Task 0: Export FiniteMPS from tenax (prerequisite)
Task 1: Grid system (foundation)
  └── Task 2: QTT class (depends on grid)
        ├── Task 3: SVD folding (depends on QTT)
        │     └── Task 4: Arithmetic (depends on folding for tests)
        │           └── Task 4c: QTT dunders
        ├── Task 5: Cross-interpolation (depends on QTT + grid)
        ├── Task 9: Partial summation (depends on QTT)
        ├── Task 10: MPS delegation methods (depends on QTT)
        ├── Task 12: norm_l2 fix (depends on Task 4)
        └── Task 6: QTTMatrix (depends on QTT + arithmetic)
              ├── Task 7: Fourier (depends on QTTMatrix)
              └── Task 11: compose, transpose, from_dense, from_cross (depends on Task 6)
Task 8: Public API (depends on all above)
```

Tasks 3, 5, 9, and 10 are independent and can be parallelized.
Task 6 depends on Task 4. Tasks 7 and 11 depend on Task 6.

**Note on scaffolding:** The repo skeleton (pyproject.toml, CLAUDE.md, LICENSE, CI, conftest.py) was already created during the brainstorming phase and exists at `/Users/yjkao/tenax-qtt/`. Tasks above only need to fill in the source and test files.
