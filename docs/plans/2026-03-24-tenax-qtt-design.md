# tenax-qtt: Quantic Tensor Train Library

**Date:** 2026-03-24
**Status:** Reviewed
**Repository:** `tenax-lab/tenax-qtt`

## Overview

`tenax-qtt` is a standalone Python package for quantic tensor train (QTT) algorithms, targeting applied math use cases: function approximation on fine grids, operator arithmetic, Fourier transforms, integration, and linear algebra — all in compressed QTT format.

It builds on `tenax` as a hard dependency, wrapping `FiniteMPS` (composition) for the QTT representation and reusing `tenax`'s SVD, QR, contraction engine, and both `DenseTensor`/`SymmetricTensor` backends.

**Reference implementation:** [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl) (Julia). This design covers feature parity for the high-priority subset, with an API ready to accommodate the remaining features in future releases.

**Key paper:** Fernandez et al., "Learning tensor networks with tensor cross interpolation: new algorithms and libraries" ([arXiv:2407.02454](https://arxiv.org/abs/2407.02454)).

## Package Metadata

- **PyPI name:** `tenax-qtt`
- **Import:** `import tenax_qtt`
- **Dependency:** `tenax` (hard), `jax` (transitive via tenax)
- **Python:** 3.11+
- **License:** same as tenax

## Package Structure

```
tenax-qtt/
├── pyproject.toml
├── CLAUDE.md
├── README.md
├── LICENSE
├── .github/workflows/tests.yml    # 3.11+3.12 Linux, 3.12 macOS
├── src/tenax_qtt/
│   ├── __init__.py                # Public API re-exports
│   ├── qtt.py                     # QTT class (wraps FiniteMPS via composition)
│   ├── matrix.py                  # QTTMatrix class (operator in QTT format)
│   ├── grid.py                    # Grid specs, bit layouts, coordinate conversions
│   ├── cross.py                   # TCI: prrLU + TCI2, error estimation
│   ├── arithmetic.py              # QTT arithmetic and recompression
│   ├── fourier.py                 # Analytic DFT MPO (Chen-Lindsey construction)
│   └── folding.py                 # SVD-based construction from dense data
├── tests/
│   ├── conftest.py
│   ├── test_qtt.py
│   ├── test_matrix.py
│   ├── test_grid.py
│   ├── test_cross.py
│   ├── test_arithmetic.py
│   ├── test_fourier.py
│   └── test_folding.py
└── examples/
    ├── function_approximation.py
    └── laplacian_solve.py
```

## Module Designs

### 1. Grid System (`grid.py`)

Handles the mapping between continuous coordinates and QTT site indices.

#### Types

```python
@dataclass(frozen=True)
class UniformGrid:
    """One variable on [a, b] discretized into 2^n_bits points."""
    a: float          # left endpoint
    b: float          # right endpoint
    n_bits: int       # number of qubits for this variable
    include_endpoint: bool = False  # whether grid includes b

@dataclass(frozen=True)
class GridSpec:
    """Multi-dimensional grid with configurable layout."""
    variables: tuple[UniformGrid, ...]
    layout: Literal["interleaved", "fused", "grouped"]
```

#### Layouts

- **`"interleaved"`**: One bit per site, alternating variables at each scale level. Sites = `[x₁_bit0, x₂_bit0, ..., x₁_bit1, x₂_bit1, ...]`. Local dimension 2. Requires all variables to have the same `n_bits`. Best compression for smooth multivariate functions.
- **`"fused"`**: One site per scale level, all variables' bits merged at that level. Local dimension `2^d` where `d` = number of variables. Allows non-uniform `n_bits` (shorter variables padded or truncated at coarser levels).
- **`"grouped"`**: All bits of variable 1 first, then variable 2, etc. Local dimension 2 throughout. Allows non-uniform `n_bits`. Natural for separable functions or per-variable operators.

#### Functions

```python
def grid_to_sites(grid: GridSpec, x: tuple[float, ...]) -> tuple[int, ...]
def sites_to_grid(grid: GridSpec, sites: tuple[int, ...]) -> tuple[float, ...]
def batch_grid_to_sites(grid: GridSpec, xs: jax.Array) -> jax.Array
def batch_sites_to_grid(grid: GridSpec, sites: jax.Array) -> jax.Array
def num_sites(grid: GridSpec) -> int
def local_dim(grid: GridSpec, site: int) -> int
def site_permutation(source: GridSpec, target_layout: str) -> tuple[int, ...]
    """Permutation mapping sites from source layout to target layout.
    Useful for reordering an existing QTT between layouts, e.g.,
    converting a grouped QTT to interleaved ordering.
    """
```

**Grid spacing note**: With `include_endpoint=False`, the grid has `2^n_bits` points with spacing `(b-a) / 2^n_bits`. With `include_endpoint=True`, spacing is `(b-a) / (2^n_bits - 1)`. The `integrate()` method's trapezoidal weights account for this: half-weight at boundary points when `include_endpoint=True`.

### 2. QTT Class (`qtt.py`)

Wraps `tenax.FiniteMPS` via composition (not inheritance). This avoids `@dataclass` field-ordering issues and decouples from `FiniteMPS`'s internal label conventions (`"v{i-1}_{i}"`, `"p{i}"`), which QTT construction code must match when building site tensors. MPS methods are forwarded explicitly.

#### Data Model

```python
@dataclass(frozen=True)
class QTT:
    mps: FiniteMPS    # the underlying tensor train
    grid: GridSpec    # grid semantics
```

#### Construction

```python
    @classmethod
    def from_cross(
        cls,
        f: Callable[[jax.Array], jax.Array] | Callable[[tuple[float, ...]], complex],
        grid: GridSpec,
        tol: float = 1e-8,
        max_bond_dim: int = 64,
        batch: bool = False,
        **kwargs,
    ) -> "QTTResult":
        """Build via TCI (delegates to cross.py).

        f signature:
          - scalar (batch=False): f((x0, x1, ..., xd)) -> complex
          - batched (batch=True): f(xs: jax.Array[batch_size, d]) -> jax.Array[batch_size]
        Scalar callables are auto-wrapped with jax.vmap for the batched path.
        """

    @classmethod
    def from_dense(cls, data: jax.Array, grid: GridSpec, max_bond_dim: int | None = None, tol: float = 1e-8) -> "QTT":
        """Build via SVD folding (delegates to folding.py)."""

    @classmethod
    def from_mps(cls, mps: FiniteMPS, grid: GridSpec) -> "QTT":
        """Wrap an existing FiniteMPS with grid metadata.

        The MPS must follow tenax's label conventions: site i has legs
        "v{i-1}_{i}" (left bond), "p{i}" (physical), "v{i}_{i+1}" (right bond),
        with boundary sites having dim-1 "v_-1_0" and "v{L-1}_{L}" bonds.
        """

    @classmethod
    def zeros(cls, grid: GridSpec) -> "QTT":
        """Bond-dim-1 QTT representing the zero function.
        All site tensors are zero-valued with shape (1, d_phys, 1).
        """

    @classmethod
    def ones(cls, grid: GridSpec) -> "QTT":
        """Bond-dim-1 QTT representing the constant function f(x) = 1.
        Site tensors are all-ones with shape (1, d_phys, 1), normalized
        so that evaluate(x) = 1 for all grid points.
        """
```

#### MPS Delegation

The following `FiniteMPS` attributes and methods are forwarded via properties/methods on `QTT`:

- **Properties**: `tensors`, `bond_dims`, `orth_center`, `singular_values`, `log_norm`
- **Methods**: `canonicalize(center)`, `left_canonicalize()`, `right_canonicalize()`, `overlap(other.mps)`, `norm()`, `entanglement_entropy(bond)`, `compute_singular_values()`

Users can also access `qtt.mps` directly for full `FiniteMPS` interop (e.g., passing to `tenax.dmrg`).

#### Function-Space Methods

```python
    def evaluate(self, x: tuple[float, ...]) -> complex:
        """Evaluate QTT at a single continuous-domain point.
        Maps x to site indices via grid, then contracts MPS.
        """

    def evaluate_batch(self, xs: jax.Array) -> jax.Array:
        """Vectorized evaluation at multiple points.
        xs: shape (n_points, d) of continuous coordinates.
        Returns: shape (n_points,) complex array.
        """

    def integrate(self, variables: list[int] | None = None) -> "QTT | complex":
        """Integrate over specified variables using trapezoidal quadrature.

        Uses continuous measure: weight = dx for interior points,
        dx/2 for boundary points when include_endpoint=True.

        variables=None: integrate all → scalar.
        variables=[0, 2]: integrate out x₀, x₂ → QTT on remaining variables
                          with a reduced GridSpec.

        Implemented by contracting selected sites with quadrature weight vectors.
        """

    def sum(self, variables: list[int] | None = None) -> "QTT | complex":
        """Sum over specified variables without grid spacing weights.
        Contracts selected sites with all-ones vectors.
        Same partial/full semantics as integrate().
        """

    def norm_l2(self) -> float:
        """Continuous L² norm: sqrt(∫|f(x)|² dx).
        Computed as sqrt(hadamard(self, self.conj()).integrate()).real,
        where the integral uses trapezoidal quadrature weights.
        """

    def to_dense(self) -> jax.Array:
        """Expand QTT to full dense array.
        Shape depends on layout: (2^n1, 2^n2, ...) for grouped,
        (2^n_total,) for 1D.
        Warning: exponential memory — use only for small grids or debugging.
        """
```

### 3. QTTMatrix Class (`matrix.py`)

A linear operator in QTT format. Uses composition (owns an MPO) rather than subclassing `TensorNetwork`. The MPO apply/compose/zipup algorithms are implemented in this module — tenax's `TensorNetwork` is a generic graph container without operator-specific infrastructure, so these are new code in tenax-qtt.

#### Data Model

```python
@dataclass(frozen=True)
class QTTMatrix:
    mpo: TensorNetwork    # MPO with 4-leg site tensors
    grid_in: GridSpec
    grid_out: GridSpec
```

**MPO site tensor convention**: Site `i` has legs labeled `"w{i-1}_{i}"` (left bond), `"mpo_top_{i}"` (output/row physical), `"mpo_bot_{i}"` (input/column physical), `"w{i}_{i+1}"` (right bond). This matches tenax's existing MPO convention used by `build_mpo_heisenberg`.

#### Construction

```python
    @classmethod
    def identity(cls, grid: GridSpec) -> "QTTMatrix": ...

    @classmethod
    def from_cross(
        cls,
        f: Callable[[tuple[float, ...], tuple[float, ...]], complex],
        grid_in: GridSpec,
        grid_out: GridSpec,
        tol: float = 1e-8,
        **kwargs,
    ) -> "QTTMatrix":
        """TCI on a matrix-valued function.

        f signature: f(x_out, x_in) -> complex, where x_out and x_in are
        tuples of continuous coordinates on grid_out and grid_in respectively.
        Internally, the operator is treated as a function of interleaved
        output/input site indices for TCI construction.
        """

    @classmethod
    def from_dense(cls, matrix: jax.Array, grid_in: GridSpec, grid_out: GridSpec, max_bond_dim: int | None = None) -> "QTTMatrix": ...

    @classmethod
    def laplacian_1d(cls, grid: GridSpec) -> "QTTMatrix":
        """Finite-difference Laplacian with known analytical QTT form."""

    @classmethod
    def derivative_1d(cls, grid: GridSpec) -> "QTTMatrix":
        """First derivative operator."""
```

#### Operations

```python
    def apply(
        self, qtt: QTT,
        method: Literal["tci", "naive", "zipup"] = "tci",
        tol: float = 1e-8,
        max_bond_dim: int = 64,
    ) -> QTT:
        """MPO × MPS with recompression."""

    def compose(
        self, other: "QTTMatrix",
        method: Literal["tci", "naive", "zipup"] = "tci",
        tol: float = 1e-8,
        max_bond_dim: int = 64,
    ) -> "QTTMatrix":
        """MPO × MPO."""

    def transpose(self) -> "QTTMatrix": ...

    def __add__(self, other): ...
    def __sub__(self, other): ...
    def __mul__(self, scalar): ...
```

#### Contraction Methods

These are implemented from scratch in tenax-qtt (not delegated to tenax):

- **`"naive"`**: Exact contraction (bond dim = χ_A × χ_B) followed by SVD recompression. Simple and correct. Bond dim can blow up for intermediate steps.
- **`"zipup"`**: Sweep left→right, contracting and LU/SVD-compressing on the fly. Bond dim stays bounded throughout. Faster and more memory-efficient than naive.
- **`"tci"`** (default): Re-interpolate the result via cross-interpolation. Treats the output as a black-box function evaluated by contracting MPO with input MPS at specific indices. Best for keeping bond dims low when the result is compressible.

### 4. Cross-Interpolation (`cross.py`)

Two algorithms for building QTTs from black-box function evaluators. Returns a `QTTResult` with convergence diagnostics.

```python
@dataclass(frozen=True)
class QTTResult:
    """Result of cross-interpolation with convergence diagnostics."""
    qtt: QTT
    n_iter: int                # number of sweeps performed
    converged: bool            # whether tolerance was reached
    estimated_error: float     # final pivot-based error estimate
    n_function_evals: int      # total number of f evaluations

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
```

#### prrLU (default)

Iterative Schur complement elimination. Advantages over TCI2:
- **Pivot removal**: can drop suboptimal pivots mid-run
- **Configurable pivot search**: `"rook"` (fast), `"full"` (exhaustive), `"block_rook"` (compromise)
- **Recompression**: can also compress an existing QTT
- **Numerical stability**: avoids direct pivot-matrix inversion

#### TCI2 (fallback)

Standard alternating half-sweeps with greedy rook pivoting. Simpler, kept as baseline.

#### Shared Infrastructure

- Scalar callables auto-wrapped with `jax.vmap` for the batched path
- Batched callables called directly with shape `(batch_size, d)`
- All pivot evaluations grouped into batches

#### Error Estimation

```python
def estimate_error(
    qtt: QTT,
    f: Callable,
    n_samples: int = 1000,
    seed: int = 0,
) -> float:
    """Estimate max |f(x) - qtt(x)| over the grid.

    Two-phase algorithm:
    1. Random sampling: evaluate |f(x) - qtt(x)| at n_samples random grid points
       to find candidate regions of large error.
    2. Greedy refinement: starting from the worst random samples, greedily search
       neighboring grid points to find local error maxima.

    Returns the largest |f(x) - qtt(x)| found.
    """
```

#### Recompression

```python
def recompress(
    qtt: QTT,
    tol: float = 1e-8,
    max_bond_dim: int | None = None,
    method: Literal["svd", "prrlu"] = "svd",
) -> QTT:
    """Recompress a QTT to lower bond dimension."""
```

### 5. Arithmetic (`arithmetic.py`)

Operations on QTT objects that produce new QTTs. All binary operations require compatible grids (identical `GridSpec`); a `ValueError` is raised for mismatched grids (different domains, resolutions, or layouts).

```python
def add(a: QTT, b: QTT, tol: float = 1e-8, max_bond_dim: int | None = None) -> QTT:
    """Sum of two QTTs. Direct-sum bond dimensions then recompress.
    Raises ValueError if a.grid != b.grid.
    """

def subtract(a: QTT, b: QTT, tol: float = 1e-8, max_bond_dim: int | None = None) -> QTT:
    """Difference of two QTTs.
    Raises ValueError if a.grid != b.grid.
    """

def scalar_multiply(a: QTT, c: complex) -> QTT:
    """Scale a QTT by a constant (multiply first site tensor)."""

def hadamard(a: QTT, b: QTT, tol: float = 1e-8, max_bond_dim: int | None = None) -> QTT:
    """Element-wise (Hadamard) product. Bond dim = χ_a × χ_b before recompression.
    Raises ValueError if a.grid != b.grid.
    """
```

`QTT` will have `__add__`, `__sub__`, `__mul__` dunder methods that delegate to these with default tolerances.

### 6. Fourier Transform (`fourier.py`)

Analytic construction of the discrete Fourier transform as a QTT-format MPO, following the Chen-Lindsey method.

```python
def fourier_mpo(
    grid: GridSpec,
    inverse: bool = False,
    max_bond_dim: int = 64,
    tol: float = 1e-12,
) -> QTTMatrix:
    """DFT operator in QTT format.

    Builds twiddle factors exp(±2πi·j·k/N) as an MPO using
    Chebyshev-Lagrange polynomial interpolation of the phase.

    Bond dimension controlled by tol on the Chebyshev interpolation,
    typically O(log(1/tol)) ≈ 40-50 for double precision.

    For multi-D grids with grouped layout: returns a single composed QTTMatrix
    that applies per-variable DFT MPOs in variable order (x₁ first, then x₂, etc.).
    For interleaved/fused layouts: the DFT MPO operates on the interleaved site
    ordering directly. Variables must have the same n_bits for interleaved layout.
    """
```

### 7. SVD Folding (`folding.py`)

Utility for constructing QTTs from known dense data.

```python
def fold_to_qtt(
    data: jax.Array,
    grid: GridSpec,
    max_bond_dim: int | None = None,
    tol: float = 1e-8,
) -> QTT:
    """Reshape dense array into high-dimensional tensor, compress via successive SVDs.

    For 1D: data has shape (2^n_bits,), reshaped to (2, 2, ..., 2), then
    left-to-right SVD sweep produces MPS with truncation.

    For multi-D: reshape according to grid layout, then same SVD sweep.
    """
```

## Prerequisites (tenax changes)

- **Export `FiniteMPS`**: Add `FiniteMPS` (and `InfiniteMPS`) to `tenax/__init__.py`'s `__all__` so tenax-qtt can import it as part of the public API rather than reaching into `tenax.core.mps`.

## Polymorphism

All modules are written to be tensor-type agnostic. Since `tenax`'s `svd`, `qr`, `contract`, and `truncated_svd` dispatch on `DenseTensor` vs `SymmetricTensor`, QTT operations inherit this polymorphism automatically. Construction methods (`from_cross`, `from_dense`) produce `DenseTensor`-based QTTs by default. Users can construct symmetric QTTs by passing `SymmetricTensor` site tensors via `QTT.from_mps()`.

## CI / Tooling

- **Build:** `uv` + `pyproject.toml`, mirroring tenax
- **CI matrix:** Python 3.11 + 3.12 on Linux (required), Python 3.12 on macOS
- **Testing:** pytest with same marker conventions as tenax (`core`, `algorithm`, `slow`)
- **Dependencies:** `tenax`, `jax`, `jaxlib` (CPU sufficient for CI)

## Future Extensions (API-ready, not in v1)

These features are **not implemented** in v1 but the API is designed to accommodate them without breaking changes:

- **Non-binary bases**: `UniformGrid.base` field (default 2), affects `local_dim`
- **Custom index tables**: `GridSpec.index_table` field overriding the layout-derived mapping
- **Named variables**: `UniformGrid.name` field for ergonomic keyword access
- **Function evaluation cache**: `CachedFunction` wrapper with large-integer keys for `n_bits > 63`
- **TCI1 algorithm**: additional `method="tci1"` option in `cross_interpolation`
- **TT fitting from noisy data**: `QTT.from_fit(data, grid, noise_level)`
- **AMEN solver**: `QTTMatrix.solve(rhs, method="amen")` for Ax=b in QTT format
- **Global pivot search**: pluggable `PivotFinder` strategy in `cross_interpolation`
