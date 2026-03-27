"""Function approximation with QTT.

Demonstrates building QTT representations of 1D functions via:
1. SVD folding from dense data
2. Tensor cross interpolation (TCI)

Shows evaluation, integration, and error estimation.
"""

import math

import jax.numpy as jnp

from tenax_qtt import (
    GridSpec,
    UniformGrid,
    cross_interpolation,
    estimate_error,
    fold_to_qtt,
)

# ---------------------------------------------------------------------------
# 1. SVD folding: sin(x) on [0, 2*pi]
# ---------------------------------------------------------------------------

n_bits = 10  # 2^10 = 1024 grid points
grid_sin = GridSpec(variables=(UniformGrid(0, 2 * math.pi, n_bits),), layout="grouped")
N = 1 << n_bits

# Build dense function values on the grid
dx = 2 * math.pi / N
x_vals = jnp.arange(N) * dx
f_dense = jnp.sin(x_vals)

# Fold into QTT via left-to-right SVD
qtt_sin = fold_to_qtt(f_dense, grid_sin, tol=1e-10)

print("=== sin(x) via SVD folding ===")
print(f"Grid: {N} points on [0, 2*pi]")
print(f"Bond dimensions: {qtt_sin.bond_dims}")
print(f"Max bond dim:    {max(qtt_sin.bond_dims)}")

# Evaluate at a test point
x_test = 1.0
val = qtt_sin.evaluate((x_test,))
print(f"sin({x_test}) = {val.real:.10f}  (exact: {math.sin(x_test):.10f})")

# Integrate sin(x) over [0, 2*pi] -- exact answer is 0
integral = qtt_sin.integrate()
print(f"Integral of sin(x) on [0, 2*pi]: {integral.real:.2e}")
print()

# ---------------------------------------------------------------------------
# 2. TCI: exp(-x^2) on [-3, 3]
# ---------------------------------------------------------------------------

grid_gauss = GridSpec(variables=(UniformGrid(-3, 3, n_bits),), layout="grouped")


def gaussian(x):
    return math.exp(-(x[0] ** 2))


result = cross_interpolation(
    gaussian,
    grid_gauss,
    tol=1e-8,
    max_bond_dim=32,
    method="prrlu",
)
qtt_gauss = result.qtt

print("=== exp(-x^2) via TCI (prrLU) ===")
print(f"Grid: {N} points on [-3, 3]")
print(f"Converged:       {result.converged}")
print(f"Iterations:      {result.n_iter}")
print(f"Function evals:  {result.n_function_evals}")
print(f"Estimated error: {result.estimated_error:.2e}")
print(f"Bond dimensions: {qtt_gauss.bond_dims}")
print(f"Max bond dim:    {max(qtt_gauss.bond_dims)}")

# Evaluate
x_test = 0.5
val = qtt_gauss.evaluate((x_test,))
print(f"exp(-{x_test}^2) = {val.real:.10f}  (exact: {math.exp(-(x_test**2)):.10f})")

# Integrate exp(-x^2) over [-3, 3] -- exact: sqrt(pi) * erf(3) ~ 1.7725
integral = qtt_gauss.integrate()
exact_integral = math.sqrt(math.pi) * math.erf(3)
print(
    f"Integral of exp(-x^2) on [-3,3]: {integral.real:.6f}  (exact: {exact_integral:.6f})"
)
print()

# ---------------------------------------------------------------------------
# 3. Compare bond dimensions
# ---------------------------------------------------------------------------

print("=== Bond dimension comparison ===")
print(f"sin(x):      max chi = {max(qtt_sin.bond_dims):3d}  (SVD folding)")
print(f"exp(-x^2):   max chi = {max(qtt_gauss.bond_dims):3d}  (TCI)")
print()

# ---------------------------------------------------------------------------
# 4. Error estimation for TCI result
# ---------------------------------------------------------------------------

err = estimate_error(qtt_gauss, gaussian, n_samples=500, seed=42)
print("=== TCI error estimation ===")
print(f"Max pointwise error (500 random samples): {err:.2e}")
