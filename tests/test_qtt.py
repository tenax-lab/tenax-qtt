"""Tests for QTT class."""

import jax
import jax.numpy as jnp
import numpy as np
from tenax import DenseTensor, FlowDirection, TensorIndex
from tenax.core.mps import FiniteMPS

from tenax_qtt.grid import GridSpec, UniformGrid
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
    # integral of f(x)=1 on [0,1) with 8 points, dx=0.125 -> 1.0
    result = qtt.integrate()
    assert abs(result - 1.0) < 1e-12


def test_qtt_norm_l2():
    grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
    qtt = QTT.ones(grid)
    # L2 norm of f(x)=1 on [0,1) = sqrt(1) = 1
    assert abs(qtt.norm_l2() - 1.0) < 1e-10


def test_partial_sum_2d():
    """Sum over one variable of a 2D function."""
    v1 = UniformGrid(0, 1, 3)
    v2 = UniformGrid(0, 1, 3)
    grid = GridSpec(variables=(v1, v2), layout="grouped")
    # f(x, y) = 1, sum over y -> f(x) = 8 (2^3 points in y)
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
    # Integrate y out: integral f(x,y)dy on [0,1) = 1.0 for each x
    result = qtt.integrate(variables=[1])
    assert isinstance(result, QTT)
    assert abs(result.evaluate((0.5,)) - 1.0) < 1e-10


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
    assert abs(ov - 16.0) < 1e-10


def test_qtt_entanglement_entropy():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    qtt = QTT.ones(grid).canonicalize(2)
    ee = qtt.entanglement_entropy(1)
    assert abs(ee) < 1e-10


def test_norm_l2_with_endpoint():
    grid = GridSpec(
        variables=(UniformGrid(0, 1, 3, include_endpoint=True),),
        layout="grouped",
    )
    qtt = QTT.ones(grid)
    assert abs(qtt.norm_l2() - 1.0) < 0.1


def test_full_workflow():
    """End-to-end: build, evaluate, arithmetic, integrate."""
    import math

    grid = GridSpec(variables=(UniformGrid(0, 2 * math.pi, 8),), layout="grouped")
    N = 256
    x = jnp.linspace(0, 2 * math.pi, N, endpoint=False)
    sin_qtt = QTT.from_dense(jnp.sin(x), grid)
    val = sin_qtt.evaluate((1.0,))
    # Nearest-neighbor lookup on 256-point grid: dx ~ 0.025, so error ~ O(dx)
    assert abs(val - math.sin(1.0)) < 0.02
    doubled = sin_qtt + sin_qtt
    assert abs(doubled.evaluate((1.0,)) - 2 * math.sin(1.0)) < 0.05
    half = sin_qtt * 0.5
    assert abs(half.evaluate((1.0,)) - 0.5 * math.sin(1.0)) < 0.02
    integral = sin_qtt.integrate()
    # Trapezoidal sum of sin over full period: should be ~0
    assert abs(integral) < 1e-10
