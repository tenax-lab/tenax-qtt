"""Tests for grid specifications and coordinate mappings."""

import pytest
from tenax_qtt.grid import UniformGrid, GridSpec, num_sites, local_dim


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
