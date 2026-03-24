"""Tests for grid specifications and coordinate mappings."""

import pytest
import jax.numpy as jnp
from tenax_qtt.grid import (
    UniformGrid,
    GridSpec,
    num_sites,
    local_dim,
    grid_to_sites,
    sites_to_grid,
    batch_grid_to_sites,
    batch_sites_to_grid,
    site_permutation,
)


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


def test_grid_to_sites_1d():
    """Point at left endpoint maps to all-zero site indices."""
    g = GridSpec(variables=(UniformGrid(0.0, 1.0, 3),), layout="grouped")
    # x=0.0 -> grid index 0 -> binary 000 -> sites (0, 0, 0)
    sites = grid_to_sites(g, (0.0,))
    assert sites == (0, 0, 0)


def test_grid_to_sites_1d_midpoint():
    g = GridSpec(variables=(UniformGrid(0.0, 1.0, 3),), layout="grouped")
    # 8 points on [0,1), spacing=0.125. x=0.5 -> index 4 -> binary 100 -> sites (1,0,0)
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
    # x1=0.5 -> index 2 -> binary 10, x2=0.25 -> index 1 -> binary 01
    # grouped: x1 bits then x2 bits -> (1, 0, 0, 1)
    sites = grid_to_sites(g, (0.5, 0.25))
    assert sites == (1, 0, 0, 1)


def test_grid_to_sites_2d_interleaved():
    v1 = UniformGrid(0.0, 1.0, 2)
    v2 = UniformGrid(0.0, 1.0, 2)
    g = GridSpec(variables=(v1, v2), layout="interleaved")
    # x1=0.5 -> index 2 -> binary 10, x2=0.25 -> index 1 -> binary 01
    # interleaved: (x1_bit0, x2_bit0, x1_bit1, x2_bit1) = (1, 0, 0, 1)
    sites = grid_to_sites(g, (0.5, 0.25))
    assert sites == (1, 0, 0, 1)


def test_grid_to_sites_2d_fused():
    v1 = UniformGrid(0.0, 1.0, 2)
    v2 = UniformGrid(0.0, 1.0, 2)
    g = GridSpec(variables=(v1, v2), layout="fused")
    # x1=0.5 -> index 2 -> binary 10, x2=0.25 -> index 1 -> binary 01
    # fused level 0: (x1_bit0=1)*2 + (x2_bit0=0) = 2
    # fused level 1: (x1_bit1=0)*2 + (x2_bit1=1) = 1
    sites = grid_to_sites(g, (0.5, 0.25))
    assert sites == (2, 1)


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
    # grouped: [x1_b0, x1_b1, x2_b0, x2_b1] -> interleaved: [x1_b0, x2_b0, x1_b1, x2_b1]
    assert perm == (0, 2, 1, 3)
