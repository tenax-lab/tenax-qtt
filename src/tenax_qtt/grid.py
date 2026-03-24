"""Grid specifications and coordinate-to-site mappings for QTT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp


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
    bits_per_var = [
        _int_to_bits(indices[i], grid.variables[i].n_bits) for i in range(d)
    ]

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


def batch_grid_to_sites(grid: GridSpec, xs: jax.Array) -> jax.Array:
    """Batch coordinate-to-site mapping. xs: (n_points, d) -> (n_points, n_sites)."""
    results = []
    for i in range(xs.shape[0]):
        x_tuple = tuple(float(xs[i, j]) for j in range(xs.shape[1]))
        sites = grid_to_sites(grid, x_tuple)
        results.append(sites)
    return jnp.array(results, dtype=jnp.int32)


def batch_sites_to_grid(grid: GridSpec, sites: jax.Array) -> jax.Array:
    """Batch site-to-coordinate mapping. sites: (n_points, n_sites) -> (n_points, d)."""
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
