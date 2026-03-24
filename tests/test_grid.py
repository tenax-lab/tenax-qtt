"""Tests for grid specifications and coordinate mappings."""

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
