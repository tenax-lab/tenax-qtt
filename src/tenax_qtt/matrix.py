"""QTTMatrix: linear operators in QTT format with MPO contraction algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from tenax import DenseTensor, FlowDirection, svd
from tenax.core.mps import FiniteMPS

from tenax_qtt._utils import trivial_index as _trivial_index
from tenax_qtt.grid import GridSpec, local_dim, num_sites
from tenax_qtt.qtt import QTT


def _flat_to_sites(flat_idx: int, grid: GridSpec) -> tuple[int, ...]:
    """Convert a flat row/column index to MPS site indices for multi-variable grids."""
    from tenax_qtt.grid import grid_to_sites

    # Decompose flat index into per-variable indices
    var_indices = []
    remaining = flat_idx
    for v in reversed(grid.variables):
        var_indices.append(remaining % v.n_points)
        remaining //= v.n_points
    var_indices.reverse()

    # Convert per-variable indices to bits and then to site indices
    # via the grid's coordinate system
    from tenax_qtt.grid import _index_to_coord

    coords = tuple(
        _index_to_coord(grid.variables[i], var_indices[i])
        for i in range(len(grid.variables))
    )
    return grid_to_sites(grid, coords)


@dataclass(frozen=True)
class QTTMatrix:
    """Linear operator in QTT (MPO) format.

    Each site tensor has 4 legs: ``(chi_wl, d_out, d_in, chi_wr)``
    where ``d_out`` is the row (output) physical dimension and ``d_in``
    is the column (input) physical dimension.
    """

    site_tensors: list  # list of 4-leg jax arrays (chi_wl, d_out, d_in, chi_wr)
    grid_in: GridSpec
    grid_out: GridSpec

    # -- Constructors --

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

        L[i,j] = (delta_{i,j-1} - 2*delta_{i,j} + delta_{i,j+1}) / dx^2
        """
        if len(grid.variables) != 1:
            raise ValueError("laplacian_1d requires a 1D grid")
        v = grid.variables[0]
        N = v.n_points
        dx = v.dx

        L = jnp.zeros((N, N))
        for i in range(N):
            L = L.at[i, i].set(-2.0 / dx**2)
            if i > 0:
                L = L.at[i, i - 1].set(1.0 / dx**2)
            if i < N - 1:
                L = L.at[i, i + 1].set(1.0 / dx**2)
        return cls._from_dense_matrix(L, grid, grid)

    @classmethod
    def _from_dense_matrix(
        cls,
        matrix: jnp.ndarray,
        grid_in: GridSpec,
        grid_out: GridSpec,
        max_bond_dim: int | None = None,
        tol: float = 1e-10,
    ) -> QTTMatrix:
        """Build QTTMatrix from a dense matrix via SVD folding.

        The matrix is reshaped into a tensor with interleaved (d_out, d_in)
        pairs per site, then decomposed left-to-right via SVD into MPO
        site tensors.
        """
        L = num_sites(grid_in)
        dims_in = [local_dim(grid_in, i) for i in range(L)]
        dims_out = [local_dim(grid_out, i) for i in range(L)]

        # Reshape matrix to tensor: first all out-dims, then all in-dims
        # M.reshape(d_out_0, ..., d_out_{L-1}, d_in_0, ..., d_in_{L-1})
        # Then transpose to interleave: (d_out_0, d_in_0, d_out_1, d_in_1, ...)
        tensor = matrix.reshape(dims_out + dims_in)
        perm = []
        for i in range(L):
            perm.append(i)  # out index i
            perm.append(L + i)  # in index i
        tensor = tensor.transpose(perm)

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
            U_t, s, Vh_t, _ = svd(
                mat_t,
                ["left"],
                ["right"],
                max_singular_values=max_bond_dim,
                max_truncation_err=tol,
            )
            chi_new = len(s)
            u_data = U_t.todense() if hasattr(U_t, "todense") else U_t.data
            site_data = u_data.reshape(chi_left, do, di, chi_new)
            site_tensors.append(site_data)

            vh_data = Vh_t.todense() if hasattr(Vh_t, "todense") else Vh_t.data
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

    # -- Application (MPO x MPS) --

    def apply(
        self,
        qtt: QTT,
        method: Literal["tci", "naive", "zipup"] = "naive",
        tol: float = 1e-8,
        max_bond_dim: int = 64,
    ) -> QTT:
        """Apply this operator to a QTT vector.

        Parameters
        ----------
        qtt : QTT
            Input vector in QTT format.
        method : {"naive", "zipup", "tci"}
            Contraction algorithm:
            - ``"naive"``: exact contraction then SVD recompression
            - ``"zipup"``: contract and compress left-to-right in one sweep
            - ``"tci"``: re-interpolate the result via cross-interpolation
        tol : float
            Truncation tolerance for SVD compression.
        max_bond_dim : int
            Maximum bond dimension after compression.

        Returns
        -------
        QTT
            Result of operator application.
        """
        if method == "naive":
            return self._apply_naive(qtt, tol, max_bond_dim)
        elif method == "zipup":
            return self._apply_zipup(qtt, tol, max_bond_dim)
        elif method == "tci":
            return self._apply_tci(qtt, tol, max_bond_dim)
        raise ValueError(f"Unknown method: {method}")

    def _apply_naive(self, qtt: QTT, tol: float, max_bond_dim: int) -> QTT:
        """Exact MPO x MPS contraction followed by SVD recompression.

        At each site, contracts the input physical index (d_in) between
        MPO and MPS, producing a combined tensor with bond dimension
        chi_w * chi_a.  The result is then globally recompressed via SVD.
        """
        from tenax_qtt.arithmetic import recompress

        L = len(self.site_tensors)
        tensors = []
        for i in range(L):
            # MPO: (chi_mpo_l, d_out, d_in, chi_mpo_r)
            W = jnp.array(self.site_tensors[i])
            if W.ndim != 4:
                raise ValueError(
                    f"MPO tensor at site {i} has {W.ndim} dims, expected 4"
                )
            chi_wl, d_out, d_in, chi_wr = W.shape

            # MPS: (chi_mps_l, d_in, chi_mps_r)
            A = qtt.tensors[i]
            A_data = A.todense() if hasattr(A, "todense") else A.data
            chi_al, _, chi_ar = A_data.shape

            # Contract over d_in: W[wl, o, i, wr] * A[al, i, ar]
            # -> result[wl, al, o, wr, ar]
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
        """Zipup: contract and compress left-to-right in one sweep.

        Like naive, but after contracting each site, immediately SVD-compress
        and absorb ``diag(s) @ Vh`` into the next site's combined tensor.
        This keeps intermediate bond dimensions bounded.
        """
        L = len(self.site_tensors)
        tensors = []
        remainder = None  # carries (chi_new, chi_w * chi_a) from previous SVD

        for i in range(L):
            W = jnp.array(self.site_tensors[i])
            chi_wl, d_out, d_in, chi_wr = W.shape
            A = qtt.tensors[i]
            A_data = A.todense() if hasattr(A, "todense") else A.data
            chi_al, _, chi_ar = A_data.shape

            # Contract over d_in
            result = jnp.einsum("woiR,lia->wloRa", W, A_data)
            result = result.reshape(chi_wl * chi_al, d_out, chi_wr * chi_ar)

            if remainder is not None:
                # Absorb remainder into left bond:
                # remainder: (chi_prev, chi_wl*chi_al)
                # result: (chi_wl*chi_al, d_out, chi_wr*chi_ar)
                result = jnp.einsum("pk,kdr->pdr", remainder, result)

            if i < L - 1:
                # SVD compress
                chi_l, d, chi_r = result.shape
                mat = result.reshape(chi_l * d, chi_r)
                left_idx = _trivial_index(mat.shape[0], FlowDirection.IN, "left")
                right_idx = _trivial_index(mat.shape[1], FlowDirection.OUT, "right")
                mat_t = DenseTensor(mat, (left_idx, right_idx))
                U_t, s, Vh_t, _ = svd(
                    mat_t,
                    ["left"],
                    ["right"],
                    max_singular_values=max_bond_dim,
                    max_truncation_err=tol,
                )
                chi_new = len(s)
                u_data = U_t.todense() if hasattr(U_t, "todense") else U_t.data
                site_data = u_data.reshape(chi_l, d, chi_new)

                left_label = f"v_{i - 1}_{i}" if i == 0 else f"v{i - 1}_{i}"
                right_label = f"v{i}_{i + 1}"
                indices = (
                    _trivial_index(site_data.shape[0], FlowDirection.IN, left_label),
                    _trivial_index(d, FlowDirection.IN, f"p{i}"),
                    _trivial_index(chi_new, FlowDirection.OUT, right_label),
                )
                tensors.append(DenseTensor(site_data, indices))

                vh_data = Vh_t.todense() if hasattr(Vh_t, "todense") else Vh_t.data
                remainder = jnp.diag(s) @ vh_data  # (chi_new, chi_wr*chi_ar)
            else:
                # Last site: no SVD, just store
                left_label = f"v{i - 1}_{i}"
                right_label = f"v{i}_{i + 1}"
                site_data = result
                if site_data.ndim == 2:
                    site_data = site_data.reshape(result.shape[0], d_out, 1)
                indices = (
                    _trivial_index(site_data.shape[0], FlowDirection.IN, left_label),
                    _trivial_index(d_out, FlowDirection.IN, f"p{i}"),
                    _trivial_index(site_data.shape[2], FlowDirection.OUT, right_label),
                )
                tensors.append(DenseTensor(site_data, indices))

        mps = FiniteMPS.from_tensors(tensors)
        return QTT(mps=mps, grid=self.grid_out)

    def _apply_tci(self, qtt: QTT, tol: float, max_bond_dim: int) -> QTT:
        """TCI-based: re-interpolate the result as a black-box function.

        Evaluates ``(MPO @ MPS)(x)`` at individual points by contracting
        MPO site tensors with the selected MPS physical indices, avoiding
        formation of the full product MPS.
        """
        from tenax_qtt.cross import cross_interpolation
        from tenax_qtt.grid import grid_to_sites

        site_tensors_list = self.site_tensors
        mps_tensors_list = qtt.tensors

        def f_result(x):
            sites = grid_to_sites(self.grid_out, x)
            # Contract site-by-site: select output physical index from MPO,
            # sum over input physical index with MPS, accumulate bond matrices.
            vec = jnp.array([[1.0]])  # row vector (1, 1)
            for i, s_out in enumerate(sites):
                W = jnp.array(site_tensors_list[i])  # (chi_wl, d_out, d_in, chi_wr)
                A = mps_tensors_list[i]
                A_data = A.todense() if hasattr(A, "todense") else A.data
                # Select output index: W_sel[w, i, r] = W[w, s_out, i, r]
                W_sel = W[:, s_out, :, :]  # (chi_wl, d_in, chi_wr)
                # Contract over d_in with MPS:
                # site_mat[w, a, r, b] = sum_i W_sel[w, i, r] * A[a, i, b]
                site_mat = jnp.einsum("wir,aib->warb", W_sel, A_data)
                chi_w, chi_a, chi_wr, chi_ar = site_mat.shape
                # Reshape to transfer matrix: (chi_wl*chi_al, chi_wr*chi_ar)
                site_mat = site_mat.reshape(chi_w * chi_a, chi_wr * chi_ar)
                # Accumulate: vec @ site_mat
                vec = vec @ site_mat
            return complex(vec[0, 0])

        result = cross_interpolation(
            f_result,
            self.grid_out,
            tol=tol,
            max_bond_dim=max_bond_dim,
            method="tci2",
        )
        return result.qtt

    # -- Transpose, compose, arithmetic --

    def transpose(self) -> QTTMatrix:
        """Swap input and output physical legs."""
        new_tensors = []
        for W in self.site_tensors:
            W = jnp.array(W)
            # W: (chi_l, d_out, d_in, chi_r) -> swap d_out and d_in
            new_tensors.append(jnp.transpose(W, (0, 2, 1, 3)))
        return QTTMatrix(
            site_tensors=new_tensors,
            grid_in=self.grid_out,
            grid_out=self.grid_in,
        )

    def compose(
        self,
        other: QTTMatrix,
        method: Literal["tci", "naive", "zipup"] = "naive",
        tol: float = 1e-8,
        max_bond_dim: int = 64,
    ) -> QTTMatrix:
        """MPO x MPO composition: ``self @ other``.

        Parameters
        ----------
        other : QTTMatrix
            The right-hand operator.
        method : {"naive", "zipup", "tci"}
            Composition algorithm.  Only ``"naive"`` is currently implemented.
        tol : float
            Truncation tolerance (reserved for future methods).
        max_bond_dim : int
            Maximum bond dimension (reserved for future methods).

        Returns
        -------
        QTTMatrix
            Result of ``self @ other``.
        """
        if method != "naive":
            raise NotImplementedError(
                f"compose method '{method}' not yet implemented, use 'naive'"
            )
        L = len(self.site_tensors)
        tensors = []
        for i in range(L):
            A = jnp.array(self.site_tensors[i])  # (chi_al, d_out, d_mid, chi_ar)
            B = jnp.array(other.site_tensors[i])  # (chi_bl, d_mid, d_in, chi_br)
            chi_al, d_out, _d_mid, chi_ar = A.shape
            chi_bl, _d_mid2, d_in, chi_br = B.shape
            # Contract over d_mid:
            # C[al,bl, d_out, d_in, ar,br] = sum_m A[al,o,m,ar] * B[bl,m,i,br]
            C = jnp.einsum("aomr,bmis->aboirs", A, B)
            C = C.reshape(chi_al * chi_bl, d_out, d_in, chi_ar * chi_br)
            tensors.append(C)
        return QTTMatrix(
            site_tensors=tensors,
            grid_in=other.grid_in,
            grid_out=self.grid_out,
        )

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
                # First site: concatenate along right bond
                new = jnp.concatenate([A, B], axis=3)
            elif i == L - 1:
                # Last site: concatenate along left bond
                new = jnp.concatenate([A, B], axis=0)
            else:
                # Middle sites: block diagonal
                new = jnp.zeros((chi_al + chi_bl, do, di, chi_ar + chi_br))
                new = new.at[:chi_al, :, :, :chi_ar].set(A)
                new = new.at[chi_al:, :, :, chi_ar:].set(B)
            tensors.append(new)
        return QTTMatrix(
            site_tensors=tensors,
            grid_in=self.grid_in,
            grid_out=self.grid_out,
        )

    def __sub__(self, other: QTTMatrix) -> QTTMatrix:
        """Subtract: ``self - other``."""
        return self + (other * -1.0)

    def __mul__(self, scalar: complex) -> QTTMatrix:
        """Scalar multiplication (right): ``M * c``."""
        tensors = list(self.site_tensors)
        tensors[0] = jnp.array(tensors[0]) * scalar
        return QTTMatrix(
            site_tensors=tensors,
            grid_in=self.grid_in,
            grid_out=self.grid_out,
        )

    def __rmul__(self, scalar: complex) -> QTTMatrix:
        """Scalar multiplication (left): ``c * M``."""
        return self.__mul__(scalar)

    # -- Public constructors (from_dense, from_cross) --

    @classmethod
    def from_dense(
        cls,
        matrix: jnp.ndarray,
        grid_in: GridSpec,
        grid_out: GridSpec,
        max_bond_dim: int | None = None,
        tol: float = 1e-10,
    ) -> QTTMatrix:
        """Build QTTMatrix from a dense matrix via SVD folding.

        Delegates to :meth:`_from_dense_matrix`.
        """
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
        """Build QTTMatrix via function evaluation.

        Parameters
        ----------
        f : callable
            ``f(x_out, x_in) -> complex`` where ``x_out`` and ``x_in``
            are tuples of coordinate values.
        grid_in, grid_out : GridSpec
            Input and output grids.
        tol : float
            SVD truncation tolerance.
        **kwargs
            Additional keyword arguments (reserved for future TCI path).

        For small operators (N_out * N_in <= 2^20) the dense matrix is
        built by evaluating *f* at every grid-point pair and then folded.
        For larger operators, raises ``NotImplementedError``.
        """
        from tenax_qtt.grid import (
            _int_to_bits,
            sites_to_grid,
        )

        L_in = num_sites(grid_in)
        L_out = num_sites(grid_out)
        if L_in != L_out:
            raise ValueError(
                "from_cross requires same number of sites for grid_in and grid_out"
            )

        # Compute total sizes
        N_out = 1
        for v in grid_out.variables:
            N_out *= v.n_points
        N_in = 1
        for v in grid_in.variables:
            N_in *= v.n_points

        if N_out * N_in <= 2**20:
            # Small enough: build dense matrix by evaluating f at all pairs

            rows = []
            for i_out in range(N_out):
                # Convert flat index to site indices for the output grid
                out_sites = (
                    tuple(_int_to_bits(i_out, grid_out.variables[0].n_bits))
                    if len(grid_out.variables) == 1
                    else _flat_to_sites(i_out, grid_out)
                )
                x_out = sites_to_grid(grid_out, out_sites)
                row = []
                for i_in in range(N_in):
                    in_sites = (
                        tuple(_int_to_bits(i_in, grid_in.variables[0].n_bits))
                        if len(grid_in.variables) == 1
                        else _flat_to_sites(i_in, grid_in)
                    )
                    x_in = sites_to_grid(grid_in, in_sites)
                    row.append(complex(f(x_out, x_in)))
                rows.append(row)
            mat = jnp.array(rows)
            return cls._from_dense_matrix(mat, grid_in, grid_out, tol=tol)

        raise NotImplementedError(
            "from_cross for large operators requires direct TCI on MPO sites"
        )

    # -- Dense expansion --

    def to_dense(self) -> jnp.ndarray:
        """Expand QTTMatrix to a full dense matrix."""
        L = len(self.site_tensors)
        dims_in = [local_dim(self.grid_in, i) for i in range(L)]
        dims_out = [local_dim(self.grid_out, i) for i in range(L)]

        # Contract all site tensors
        result = None
        for i in range(L):
            W = jnp.array(self.site_tensors[i])  # (chi_wl, d_out, d_in, chi_wr)
            if result is None:
                # First site: remove left bond dim-1
                result = W[0]  # (d_out, d_in, chi_wr)
            else:
                # result: (..., chi_wl), W: (chi_wl, d_out, d_in, chi_wr)
                result = jnp.tensordot(result, W, axes=([-1], [0]))

        # result shape: (d_out_0, d_in_0, d_out_1, d_in_1, ..., d_out_{L-1}, d_in_{L-1}, 1)
        # Remove trailing bond dim
        shape = result.shape[:-1]  # drop the trailing 1
        result = result.reshape(shape)

        # Transpose to separate all out-indices and all in-indices
        # Current order: (d_out_0, d_in_0, d_out_1, d_in_1, ...)
        # Want: (d_out_0, d_out_1, ..., d_in_0, d_in_1, ...)
        n = len(dims_in)
        out_axes = list(range(0, 2 * n, 2))  # [0, 2, 4, ...]
        in_axes = list(range(1, 2 * n, 2))  # [1, 3, 5, ...]
        result = jnp.transpose(result, out_axes + in_axes)

        # Reshape to (N_out, N_in)
        N_out = 1
        for d in dims_out:
            N_out *= d
        N_in = 1
        for d in dims_in:
            N_in *= d
        return result.reshape(N_out, N_in)
