# tenax-qtt

Quantic Tensor Train (QTT) algorithms built on [Tenax](https://github.com/tenax-lab/tenax).

[![CI](https://github.com/tenax-lab/tenax-qtt/actions/workflows/ci.yml/badge.svg)](https://github.com/tenax-lab/tenax-qtt/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

## Overview

The Quantic Tensor Train (QTT) format represents functions and operators on
exponentially fine grids using tensor trains (matrix product states) with
logarithmic complexity in the number of grid points. A function on a grid of
$N = 2^R$ points is stored as an MPS with $R$ sites and local dimension 2,
compressing smooth functions to bond dimensions that depend only on the
function's complexity -- not on $N$.

**tenax-qtt** provides QTT construction (SVD folding and tensor cross
interpolation), arithmetic, operator application, and Fourier transforms, all
built on the [Tenax](https://github.com/tenax-lab/tenax) tensor network library
and JAX.

## Installation

Development install (requires [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/tenax-lab/tenax-qtt.git
cd tenax-qtt
uv sync
```

## Quick Start

```python
import math
from tenax_qtt import UniformGrid, GridSpec, cross_interpolation

# Define a 1D grid: 2^10 = 1024 points on [0, 2pi]
grid = GridSpec(variables=(UniformGrid(0, 2 * math.pi, 10),), layout="grouped")

# Build a QTT from sin(x) via tensor cross interpolation
result = cross_interpolation(lambda x: math.sin(x[0]), grid, tol=1e-8)
qtt = result.qtt

# Evaluate at a point
val = qtt.evaluate((1.0,))
print(f"sin(1.0) ~ {val.real:.6f}  (exact: {math.sin(1.0):.6f})")

# Integrate sin(x) over [0, 2pi] (exact answer: 0)
integral = qtt.integrate()
print(f"integral ~ {integral.real:.6e}")

# Bond dimensions (measures compression)
print(f"bond dims: {qtt.bond_dims}")
```

## Features

- **QTT construction** -- SVD folding from dense data (`fold_to_qtt`) or
  black-box functions via tensor cross interpolation (`cross_interpolation`)
- **Two TCI algorithms** -- prrLU (partial rank-revealing LU) and TCI2
  (alternating half-sweep), with pivot enrichment and convergence diagnostics
- **Grid system** -- Uniform grids with grouped, interleaved, or fused site
  layouts for multi-dimensional problems
- **Arithmetic** -- addition, subtraction, scalar multiplication, Hadamard
  (element-wise) product, and SVD recompression
- **Operators** -- `QTTMatrix` (MPO format) with identity, derivative, and
  Laplacian constructors; naive, zip-up, and TCI-based application methods
- **Fourier transform** -- Analytical DFT MPO via binary phase decomposition,
  with SVD compression (no dense $N \times N$ matrix formed)
- **Evaluation** -- pointwise and batched evaluation, dense expansion,
  summation, and integration (full or partial, with trapezoidal quadrature)
- **MPS interop** -- `QTT` wraps Tenax `FiniteMPS`, exposing canonicalization,
  overlap, entanglement entropy, and singular values

## API Overview

| Module | Key exports | Description |
|--------|------------|-------------|
| `tenax_qtt.qtt` | `QTT` | Core QTT class wrapping `FiniteMPS` with grid semantics |
| `tenax_qtt.grid` | `UniformGrid`, `GridSpec` | Grid specification and coordinate-to-site mappings |
| `tenax_qtt.cross` | `cross_interpolation`, `QTTResult`, `estimate_error` | Tensor cross interpolation (prrLU / TCI2) |
| `tenax_qtt.folding` | `fold_to_qtt` | SVD-based QTT construction from dense arrays |
| `tenax_qtt.matrix` | `QTTMatrix` | Linear operators in MPO format with application methods |
| `tenax_qtt.fourier` | `fourier_mpo` | Analytical DFT operator in QTT format |
| `tenax_qtt.arithmetic` | `add`, `subtract`, `hadamard`, `recompress` | QTT arithmetic and compression |

## Examples

See the [`examples/`](examples/) directory:

- **`function_approximation.py`** -- Build QTTs from functions via SVD folding
  and TCI; evaluate, integrate, and estimate errors.
- **`laplacian_solve.py`** -- Build the Laplacian and Fourier operators in QTT
  format; apply them to verify eigenvalues and compute spectra.

Run an example:

```bash
uv run python examples/function_approximation.py
```

## Citation

If you use tenax-qtt in your research, please cite the prrLU paper:

```bibtex
@article{fernandez2024prrlu,
  title={Speeding up tensor network contraction with prrLU decomposition},
  author={Fernandez, Yuriel Nunez and others},
  journal={arXiv preprint arXiv:2407.02454},
  year={2024}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
