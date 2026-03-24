"""QTT arithmetic: addition, Hadamard product, recompression."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from tenax import DenseTensor, FlowDirection, TensorIndex, U1Symmetry, svd
from tenax.core.mps import FiniteMPS

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
    data = t0.todense() if hasattr(t0, "todense") else t0.data
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

    # Restore the overall scale absorbed during canonicalization
    tensors = list(new_mps.tensors)
    if new_mps.log_norm != 0.0:
        t0 = tensors[0]
        d0 = t0.todense() if hasattr(t0, "todense") else t0.data
        scale = jnp.exp(new_mps.log_norm)
        tensors[0] = DenseTensor(d0 * scale, t0.indices)
    for i in range(L - 1):
        t = tensors[i]
        data = t.todense() if hasattr(t, "todense") else t.data
        chi_l, d, chi_r = data.shape

        # Reshape to (chi_l * d, chi_r): left factor stays as site tensor
        mat = data.reshape(chi_l * d, chi_r)
        left_idx = _trivial_index(mat.shape[0], FlowDirection.IN, "left")
        right_idx = _trivial_index(mat.shape[1], FlowDirection.OUT, "right")
        mat_tensor = DenseTensor(mat, (left_idx, right_idx))

        U, s, Vh, _ = svd(
            mat_tensor,
            ["left"],
            ["right"],
            new_bond_label="bond",
            max_singular_values=max_bond_dim,
            max_truncation_err=tol,
        )
        chi_new = len(s)

        # U reshaped to new site tensor: (chi_l, d, chi_new)
        u_data = U.todense() if hasattr(U, "todense") else U.data
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
        vh_data = Vh.todense() if hasattr(Vh, "todense") else Vh.data
        svh = jnp.diag(s) @ vh_data  # (chi_new, chi_r_old)

        t_right = tensors[i + 1]
        data_right = t_right.todense() if hasattr(t_right, "todense") else t_right.data
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


def _direct_sum_mps(a: QTT, b: QTT) -> list[DenseTensor]:
    """Direct-sum the bond dimensions of two MPS, site by site."""
    L = len(a.tensors)
    tensors = []
    for i in range(L):
        da = (
            a.tensors[i].todense()
            if hasattr(a.tensors[i], "todense")
            else a.tensors[i].data
        )
        db = (
            b.tensors[i].todense()
            if hasattr(b.tensors[i], "todense")
            else b.tensors[i].data
        )
        chi_la, d, chi_ra = da.shape
        chi_lb, _, chi_rb = db.shape

        if i == 0:
            # First site: horizontal concat -> (1, d, chi_a + chi_b)
            new_data = jnp.concatenate([da, db], axis=2)
        elif i == L - 1:
            # Last site: vertical concat -> (chi_a + chi_b, d, 1)
            new_data = jnp.concatenate([da, db], axis=0)
        else:
            # Middle: block diagonal -> (chi_la+chi_lb, d, chi_ra+chi_rb)
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
    a: QTT,
    b: QTT,
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
    a: QTT,
    b: QTT,
    tol: float = 1e-8,
    max_bond_dim: int | None = None,
) -> QTT:
    """Difference of two QTTs."""
    return add(a, scalar_multiply(b, -1.0), tol=tol, max_bond_dim=max_bond_dim)


def hadamard(
    a: QTT,
    b: QTT,
    tol: float = 1e-8,
    max_bond_dim: int | None = None,
) -> QTT:
    """Element-wise (Hadamard) product via bond dimension multiplication."""
    _check_compatible(a, b)
    L = len(a.tensors)
    tensors = []
    for i in range(L):
        da = (
            a.tensors[i].todense()
            if hasattr(a.tensors[i], "todense")
            else a.tensors[i].data
        )
        db = (
            b.tensors[i].todense()
            if hasattr(b.tensors[i], "todense")
            else b.tensors[i].data
        )
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
