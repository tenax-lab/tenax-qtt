"""Tensor cross interpolation: prrLU and TCI2 algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import jax.numpy as jnp
import numpy as np
from tenax import DenseTensor, FlowDirection, TensorIndex, U1Symmetry
from tenax.core.mps import FiniteMPS

from tenax_qtt.grid import GridSpec, local_dim, num_sites, sites_to_grid
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


def _build_mps_from_site_tensors(
    site_tensors: list[np.ndarray], grid: GridSpec
) -> QTT:
    """Build a QTT from a list of numpy site tensors with proper tenax labels."""
    L = len(site_tensors)
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
    return QTT(mps=mps, grid=grid)


def _svd_truncate(
    C: np.ndarray, tol: float, max_chi: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """SVD with truncation. Returns (U, s, Vh, chi_new)."""
    U, s, Vh = np.linalg.svd(C, full_matrices=False)
    chi = min(len(s), max_chi)
    if tol > 0 and len(s) > 1 and s[0] > 0:
        for kk in range(len(s)):
            if s[kk] / s[0] < tol:
                chi = min(kk, chi)
                break
    chi = max(1, chi)
    return U[:, :chi], s[:chi], Vh[:chi, :], chi


def _maxvol_pivots(A: np.ndarray, n_pivots: int) -> list[int]:
    """Select n_pivots row indices from matrix A using greedy selection."""
    used: set[int] = set()
    pivots: list[int] = []

    if A.ndim == 1:
        A = A.reshape(-1, 1)

    n_cols = min(A.shape[1], n_pivots)
    for c in range(n_cols):
        col = np.abs(A[:, c])
        for r in np.argsort(-col):
            r = int(r)
            if r not in used:
                used.add(r)
                pivots.append(r)
                break

    if len(pivots) < n_pivots:
        norms = np.linalg.norm(A, axis=1)
        for r in np.argsort(-norms):
            r = int(r)
            if r not in used:
                used.add(r)
                pivots.append(r)
                if len(pivots) >= n_pivots:
                    break

    return pivots[:n_pivots]


def _build_mps_ci(
    f: Callable,
    grid: GridSpec,
    dims: list[int],
    I_sets: list[list[tuple[int, ...]]],
    J_sets: list[list[tuple[int, ...]]],
    batch: bool,
    tol: float = 0.0,
    max_bond_dim: int = 64,
) -> tuple[list[np.ndarray], int]:
    """Build MPS from cross-interpolation pivot sets using skeleton decomposition.

    At each non-terminal bond k:
      C_k = f(I_{k-1} x s_k, J_k)  -- cross matrix (chi_{k-1}*d) x chi_k
      P_k = C_k[pivot_rows, :]      -- pivot submatrix chi_k x chi_k
      A_k = C_k @ P_k^{-1}          -- site tensor (chi_{k-1}, d, chi_k)

    At the last site:
      A_{L-1} = f(I_{L-2} x s_{L-1})  -- (chi_{L-2}, d_{L-1}, 1)
    """
    L = len(dims)
    n_evals = 0
    site_tensors: list[np.ndarray] = []

    for k in range(L):
        d = dims[k]
        I_left: list[tuple[int, ...]] = [()] if k == 0 else I_sets[k - 1]
        n_il = len(I_left)

        if k < L - 1:
            J_right = J_sets[k]
            n_jr = len(J_right)

            # Cross matrix
            multi_indices = []
            for il in I_left:
                for s in range(d):
                    for jr in J_right:
                        multi_indices.append(il + (s,) + jr)
            vals = _evaluate_f(f, grid, multi_indices, batch)
            n_evals += len(multi_indices)
            C = vals.reshape(n_il * d, n_jr)

            # Find pivot rows corresponding to I_sets[k]
            i_left_to_idx = {il: idx for idx, il in enumerate(I_left)}
            pivot_rows = []
            for pivot_tuple in I_sets[k]:
                prefix = pivot_tuple[:-1]
                s_val = pivot_tuple[-1]
                if prefix in i_left_to_idx:
                    row = i_left_to_idx[prefix] * d + s_val
                    pivot_rows.append(row)

            if len(pivot_rows) == 0:
                # Fallback: use first rows
                pivot_rows = list(range(min(n_il * d, n_jr)))

            chi_k = len(pivot_rows)
            P = C[pivot_rows, :]

            # Skeleton decomposition: A = C @ P^{-1}
            if chi_k == n_jr and chi_k > 0:
                try:
                    A = np.linalg.solve(P.T, C.T).T
                except np.linalg.LinAlgError:
                    A = C @ np.linalg.pinv(P)
            else:
                A = C @ np.linalg.pinv(P)

            site_tensors.append(A.reshape(n_il, d, chi_k))
        else:
            # Last site
            multi_indices = []
            for il in I_left:
                for s in range(d):
                    multi_indices.append(il + (s,))
            vals = _evaluate_f(f, grid, multi_indices, batch)
            n_evals += len(multi_indices)
            site_tensors.append(vals.reshape(n_il, d, 1))

    return site_tensors, n_evals


def _deduplicate_tuples(lst: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Remove duplicate tuples preserving order."""
    seen: set[tuple[int, ...]] = set()
    result = []
    for t in lst:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _tci_sweep(
    f: Callable,
    grid: GridSpec,
    dims: list[int],
    I_sets: list[list[tuple[int, ...]]],
    J_sets: list[list[tuple[int, ...]]],
    tol: float,
    max_bond_dim: int,
    batch: bool,
    rng: np.random.Generator,
    use_lu: bool = False,
) -> int:
    """One full left-right-left sweep to update pivot sets. Returns n_evals."""
    L = len(dims)
    n_evals = 0

    # ---- Left-to-right: update I_sets (left pivots) ----
    for k in range(L - 1):
        d = dims[k]
        I_left: list[tuple[int, ...]] = [()] if k == 0 else I_sets[k - 1]
        J_right = J_sets[k]
        n_il = len(I_left)
        n_jr = len(J_right)

        multi_indices = []
        for il in I_left:
            for s in range(d):
                for jr in J_right:
                    multi_indices.append(il + (s,) + jr)
        vals = _evaluate_f(f, grid, multi_indices, batch)
        n_evals += len(multi_indices)
        C = vals.reshape(n_il * d, n_jr)

        # Use SVD to find the most important rows; keep up to
        # min(matrix_rank, max_bond_dim) pivots WITHOUT dropping
        # below the current pivot count (to preserve enrichment).
        U, s_vals, Vh = np.linalg.svd(C, full_matrices=False)
        chi_svd = min(len(s_vals), max_bond_dim)
        if tol > 0 and len(s_vals) > 1 and s_vals[0] > 0:
            for kk in range(len(s_vals)):
                if s_vals[kk] / s_vals[0] < tol:
                    chi_svd = min(kk, chi_svd)
                    break
        chi_svd = max(1, chi_svd)
        # Never shrink below the current I_sets[k] size (preserve enrichment)
        chi = max(chi_svd, min(len(I_sets[k]), min(n_il * d, n_jr)))
        chi = min(chi, max_bond_dim, n_il * d)

        # Select pivot rows from U columns
        pivot_rows = _maxvol_pivots(U[:, :chi], chi)
        new_I = []
        for r in pivot_rows:
            i_idx = r // d
            s_idx = r % d
            new_I.append(I_left[i_idx] + (s_idx,))
        I_sets[k] = _deduplicate_tuples(new_I)
        if len(I_sets[k]) == 0:
            I_sets[k] = new_I[:1]

    # ---- Right-to-left: update J_sets (right pivots) ----
    for k in range(L - 2, -1, -1):
        d = dims[k + 1]
        I_left = I_sets[k]
        J_right: list[tuple[int, ...]] = (
            [()] if k == L - 2 else J_sets[k + 1]
        )
        n_il = len(I_left)
        n_jr = len(J_right)

        multi_indices = []
        for il in I_left:
            for s in range(d):
                for jr in J_right:
                    multi_indices.append(il + (s,) + jr)
        vals = _evaluate_f(f, grid, multi_indices, batch)
        n_evals += len(multi_indices)
        C = vals.reshape(n_il, d * n_jr)

        U, s_vals, Vh = np.linalg.svd(C, full_matrices=False)
        chi_svd = min(len(s_vals), max_bond_dim)
        if tol > 0 and len(s_vals) > 1 and s_vals[0] > 0:
            for kk in range(len(s_vals)):
                if s_vals[kk] / s_vals[0] < tol:
                    chi_svd = min(kk, chi_svd)
                    break
        chi_svd = max(1, chi_svd)
        # Never shrink below the current J_sets[k] size
        chi = max(chi_svd, min(len(J_sets[k]), min(n_il, d * n_jr)))
        chi = min(chi, max_bond_dim, d * n_jr)

        pivot_cols = _maxvol_pivots(Vh[:chi, :].T, chi)
        new_J = []
        for c in pivot_cols:
            s_idx = c // n_jr
            j_idx = c % n_jr
            new_J.append((s_idx,) + J_right[j_idx])
        J_sets[k] = _deduplicate_tuples(new_J)
        if len(J_sets[k]) == 0:
            J_sets[k] = new_J[:1]

    return n_evals


def _enrich_pivots(
    I_sets: list[list[tuple[int, ...]]],
    J_sets: list[list[tuple[int, ...]]],
    dims: list[int],
    max_bond_dim: int,
    rng: np.random.Generator,
    n_add: int = 2,
) -> None:
    """Add random pivots to grow bond dimension (pivot enrichment)."""
    L = len(dims)
    for k in range(L - 1):
        chi_current = len(I_sets[k])
        chi_add = min(n_add, max_bond_dim - chi_current)
        if chi_add <= 0:
            continue
        existing_I = set(I_sets[k])
        existing_J = set(J_sets[k])
        for _ in range(chi_add * 10):  # try multiple times
            left = tuple(
                int(rng.integers(0, dims[j])) for j in range(k + 1)
            )
            right = tuple(
                int(rng.integers(0, dims[j])) for j in range(k + 1, L)
            )
            if left not in existing_I:
                I_sets[k].append(left)
                existing_I.add(left)
            if right not in existing_J:
                J_sets[k].append(right)
                existing_J.add(right)
            if (
                len(I_sets[k]) >= chi_current + chi_add
                and len(J_sets[k]) >= chi_current + chi_add
            ):
                break


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
    """TCI2: alternating half-sweep cross interpolation.

    The algorithm:
    1. Initialize random pivot sets at each bond (small initial chi).
    2. Iterate:
       a. Sweep: update left pivots (left-to-right), then right pivots (right-to-left).
       b. Build MPS from current pivots using skeleton decomposition.
       c. Check convergence by random sampling.
       d. If not converged, add random pivots (enrichment) to grow bond dimension.
    """
    rng = np.random.default_rng(seed)
    L = num_sites(grid)
    dims = [local_dim(grid, i) for i in range(L)]
    n_evals = 0

    chi_init = min(2, max_bond_dim)

    I_sets: list[list[tuple[int, ...]]] = []
    J_sets: list[list[tuple[int, ...]]] = []

    for k in range(L - 1):
        chi = chi_init
        left_pivots: list[tuple[int, ...]] = []
        right_pivots: list[tuple[int, ...]] = []
        existing_l: set[tuple[int, ...]] = set()
        existing_r: set[tuple[int, ...]] = set()
        while len(left_pivots) < chi:
            left = tuple(
                int(rng.integers(0, dims[j])) for j in range(k + 1)
            )
            if left not in existing_l:
                existing_l.add(left)
                left_pivots.append(left)
        while len(right_pivots) < chi:
            right = tuple(
                int(rng.integers(0, dims[j])) for j in range(k + 1, L)
            )
            if right not in existing_r:
                existing_r.add(right)
                right_pivots.append(right)
        I_sets.append(left_pivots)
        J_sets.append(right_pivots)

    converged = False
    est_error = float("inf")
    iteration = 0

    for iteration in range(max_iter):
        # Pivot update sweep
        ne = _tci_sweep(
            f, grid, dims, I_sets, J_sets, tol, max_bond_dim, batch, rng
        )
        n_evals += ne

        # Build MPS
        site_tensors, ne = _build_mps_ci(
            f, grid, dims, I_sets, J_sets, batch, tol, max_bond_dim
        )
        n_evals += ne

        # Convergence check
        n_check = min(200, 2 ** L)
        check_err = 0.0
        qtt_tmp = _build_mps_from_site_tensors(site_tensors, grid)
        for _ in range(n_check):
            sites = tuple(int(rng.integers(0, dims[i])) for i in range(L))
            x = sites_to_grid(grid, sites)
            f_val = complex(f(x))
            qtt_val = qtt_tmp.evaluate(x)
            check_err = max(check_err, abs(f_val - qtt_val))
            n_evals += 1

        est_error = check_err
        if est_error < tol:
            converged = True
            break

        # Pivot enrichment: if not converged, add pivots to grow bond dim
        max_chi = max(len(I) for I in I_sets)
        if max_chi < max_bond_dim:
            _enrich_pivots(I_sets, J_sets, dims, max_bond_dim, rng, n_add=2)

    # Final MPS
    qtt = _build_mps_from_site_tensors(site_tensors, grid)

    return QTTResult(
        qtt=qtt,
        n_iter=iteration + 1,
        converged=converged,
        estimated_error=est_error,
        n_function_evals=n_evals,
    )


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

    Same structure as TCI2 but uses LU decomposition for left-sweep
    pivot selection when scipy is available.
    """
    rng = np.random.default_rng(seed)
    L = num_sites(grid)
    dims = [local_dim(grid, i) for i in range(L)]
    n_evals = 0

    chi_init = min(2, max_bond_dim)
    I_sets: list[list[tuple[int, ...]]] = []
    J_sets: list[list[tuple[int, ...]]] = []

    for k in range(L - 1):
        chi = chi_init
        left_pivots: list[tuple[int, ...]] = []
        right_pivots: list[tuple[int, ...]] = []
        existing_l: set[tuple[int, ...]] = set()
        existing_r: set[tuple[int, ...]] = set()
        while len(left_pivots) < chi:
            left = tuple(
                int(rng.integers(0, dims[j])) for j in range(k + 1)
            )
            if left not in existing_l:
                existing_l.add(left)
                left_pivots.append(left)
        while len(right_pivots) < chi:
            right = tuple(
                int(rng.integers(0, dims[j])) for j in range(k + 1, L)
            )
            if right not in existing_r:
                existing_r.add(right)
                right_pivots.append(right)
        I_sets.append(left_pivots)
        J_sets.append(right_pivots)

    converged = False
    est_error = float("inf")
    iteration = 0

    for iteration in range(max_iter):
        # Left-to-right with LU pivot selection
        for k in range(L - 1):
            d = dims[k]
            I_left: list[tuple[int, ...]] = (
                [()] if k == 0 else I_sets[k - 1]
            )
            J_right = J_sets[k]
            n_il = len(I_left)
            n_jr = len(J_right)

            multi_indices = []
            for il in I_left:
                for s in range(d):
                    for jr in J_right:
                        multi_indices.append(il + (s,) + jr)
            vals = _evaluate_f(f, grid, multi_indices, batch)
            n_evals += len(multi_indices)
            C = vals.reshape(n_il * d, n_jr)

            chi = min(min(C.shape), max_bond_dim)
            new_I: list[tuple[int, ...]] = []
            lu_used = False
            try:
                from scipy.linalg import lu as scipy_lu

                P_mat, L_mat, U_lu = scipy_lu(C)
                m = min(C.shape)
                diag = np.abs(np.diag(U_lu[:m, :m]))
                if tol > 0 and len(diag) > 1 and diag[0] > 0:
                    for kk in range(len(diag)):
                        if diag[kk] / diag[0] < tol:
                            chi = min(kk, chi)
                            break
                chi = max(1, chi)
                perm_inv = np.argsort(np.argmax(P_mat.T, axis=1))
                for c_idx in range(chi):
                    r = int(perm_inv[c_idx])
                    i_idx = r // d
                    s_idx = r % d
                    new_I.append(I_left[i_idx] + (s_idx,))
                lu_used = True
            except ImportError:
                pass

            if not lu_used:
                U, s_vals, Vh = np.linalg.svd(C, full_matrices=False)
                if tol > 0 and len(s_vals) > 1 and s_vals[0] > 0:
                    for kk in range(len(s_vals)):
                        if s_vals[kk] / s_vals[0] < tol:
                            chi = min(kk, chi)
                            break
                chi = max(1, chi)
                pivot_rows = _maxvol_pivots(U[:, :chi], chi)
                for r in pivot_rows:
                    i_idx = r // d
                    s_idx = r % d
                    new_I.append(I_left[i_idx] + (s_idx,))

            I_sets[k] = _deduplicate_tuples(new_I)
            if len(I_sets[k]) == 0:
                I_sets[k] = new_I[:1]

        # Right-to-left pivot update
        for k in range(L - 2, -1, -1):
            d = dims[k + 1]
            I_left = I_sets[k]
            J_right_list: list[tuple[int, ...]] = (
                [()] if k == L - 2 else J_sets[k + 1]
            )
            n_il = len(I_left)
            n_jr = len(J_right_list)

            multi_indices = []
            for il in I_left:
                for s in range(d):
                    for jr in J_right_list:
                        multi_indices.append(il + (s,) + jr)
            vals = _evaluate_f(f, grid, multi_indices, batch)
            n_evals += len(multi_indices)
            C = vals.reshape(n_il, d * n_jr)

            U, s_vals, Vh = np.linalg.svd(C, full_matrices=False)
            chi = min(len(s_vals), max_bond_dim)
            if tol > 0 and len(s_vals) > 1 and s_vals[0] > 0:
                for kk in range(len(s_vals)):
                    if s_vals[kk] / s_vals[0] < tol:
                        chi = min(kk, chi)
                        break
            chi = max(1, chi)

            pivot_cols = _maxvol_pivots(Vh[:chi, :].T, chi)
            new_J: list[tuple[int, ...]] = []
            for c_idx in pivot_cols:
                s_idx = c_idx // n_jr
                j_idx = c_idx % n_jr
                new_J.append((s_idx,) + J_right_list[j_idx])
            J_sets[k] = _deduplicate_tuples(new_J)
            if len(J_sets[k]) == 0:
                J_sets[k] = new_J[:1]

        # Build MPS and check convergence
        site_tensors, ne = _build_mps_ci(
            f, grid, dims, I_sets, J_sets, batch, tol, max_bond_dim
        )
        n_evals += ne

        n_check = min(200, 2 ** L)
        check_err = 0.0
        qtt_tmp = _build_mps_from_site_tensors(site_tensors, grid)
        for _ in range(n_check):
            sites = tuple(int(rng.integers(0, dims[i])) for i in range(L))
            x = sites_to_grid(grid, sites)
            f_val = complex(f(x))
            qtt_val = qtt_tmp.evaluate(x)
            check_err = max(check_err, abs(f_val - qtt_val))
            n_evals += 1

        est_error = check_err
        if est_error < tol:
            converged = True
            break

        max_chi = max(len(I) for I in I_sets)
        if max_chi < max_bond_dim:
            _enrich_pivots(I_sets, J_sets, dims, max_bond_dim, rng, n_add=2)

    qtt = _build_mps_from_site_tensors(site_tensors, grid)

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
    """Build a QTT from a black-box function via cross-interpolation.

    Parameters
    ----------
    f : Callable
        Function to approximate.  When ``batch=False`` (default), called as
        ``f(x_tuple) -> complex`` where *x_tuple* is a tuple of floats
        ``(x0, x1, ..., xd)``.  When ``batch=True``, called with a JAX array
        of shape ``(n, d)`` and must return an array of shape ``(n,)``.
    grid : GridSpec
        Multi-dimensional grid specification.
    tol : float
        Convergence tolerance.
    max_bond_dim : int
        Maximum MPS bond dimension.
    max_iter : int
        Maximum number of full sweeps.
    method : {"prrlu", "tci2"}
        Cross-interpolation variant.
    pivot_strategy : {"rook", "full", "block_rook"}
        Pivot search strategy (only used by prrLU).
    batch : bool
        If True, *f* is called in vectorized batch mode.
    batch_size : int
        Batch size for vectorized evaluation.
    seed : int
        Random seed for pivot initialization.

    Returns
    -------
    QTTResult
        Contains the approximating QTT and convergence diagnostics.
    """
    if method == "tci2":
        return _tci2(
            f, grid, tol, max_bond_dim, max_iter, batch, batch_size, seed
        )
    elif method == "prrlu":
        return _prrlu(
            f,
            grid,
            tol,
            max_bond_dim,
            max_iter,
            pivot_strategy,
            batch,
            batch_size,
            seed,
        )
    raise ValueError(f"Unknown method: {method}")


def estimate_error(
    qtt: QTT,
    f: Callable,
    n_samples: int = 1000,
    seed: int = 0,
) -> float:
    """Estimate max |f(x) - qtt(x)| via random sampling.

    Parameters
    ----------
    qtt : QTT
        The QTT approximation to test.
    f : Callable
        Original function (scalar mode: ``f(x_tuple) -> complex``).
    n_samples : int
        Number of random sample points.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Estimated maximum pointwise error.
    """
    rng = np.random.default_rng(seed)
    grid = qtt.grid
    L = num_sites(grid)
    dims = [local_dim(grid, i) for i in range(L)]

    max_err = 0.0
    for _ in range(n_samples):
        sites = tuple(int(rng.integers(0, dims[i])) for i in range(L))
        x = sites_to_grid(grid, sites)
        f_val = complex(f(x))
        qtt_val = qtt.evaluate(x)
        err = abs(f_val - qtt_val)
        if err > max_err:
            max_err = err

    return float(max_err)
