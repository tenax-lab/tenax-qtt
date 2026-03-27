"""Solving -u'' = f with QTT operators.

Demonstrates:
1. Building the Laplacian operator in QTT format
2. Applying it to a known function
3. Fourier transform of a signal
"""

import math

import jax.numpy as jnp

from tenax_qtt import GridSpec, UniformGrid, fold_to_qtt, fourier_mpo
from tenax_qtt.matrix import QTTMatrix

# ---------------------------------------------------------------------------
# 1. Build the Laplacian on [0, 1] and apply to sin(pi*x)
# ---------------------------------------------------------------------------

n_bits = 8  # 2^8 = 256 grid points
grid = GridSpec(
    variables=(UniformGrid(0, 1, n_bits, include_endpoint=True),),
    layout="grouped",
)
N = 1 << n_bits

print("=== Laplacian operator in QTT format ===")
print(f"Grid: {N} points on [0, 1]")

# Build the Laplacian MPO: d^2/dx^2
lap = QTTMatrix.laplacian_1d(grid)
print(f"Laplacian MPO bond dims: {[t.shape[0] for t in lap.site_tensors]}")

# Build sin(pi*x) as a QTT via SVD folding
x_vals = jnp.linspace(0, 1, N)
f_sin = jnp.sin(math.pi * x_vals)
qtt_sin = fold_to_qtt(f_sin, grid, tol=1e-10)

print(f"sin(pi*x) bond dims: {qtt_sin.bond_dims}")

# Apply Laplacian: d^2/dx^2 sin(pi*x) = -pi^2 sin(pi*x)
lap_sin = lap.apply(qtt_sin, method="naive", tol=1e-8, max_bond_dim=64)

print(f"Lap(sin) bond dims:  {lap_sin.bond_dims}")

# Verify at an interior point (avoid boundaries where FD stencil is inaccurate)
x_test = 0.5
val_lap = lap_sin.evaluate((x_test,))
val_sin = qtt_sin.evaluate((x_test,))

if abs(val_sin) > 1e-12:
    ratio = val_lap / val_sin
    print(f"\nAt x = {x_test}:")
    print(f"  sin(pi*x)           = {val_sin.real:.6f}")
    print(f"  Lap[sin(pi*x)]      = {val_lap.real:.6f}")
    print(f"  ratio Lap/sin       = {ratio.real:.4f}")
    print(f"  expected (-pi^2)    = {-(math.pi**2):.4f}")
    print(f"  relative error      = {abs(ratio.real + math.pi**2) / math.pi**2:.2e}")
print()

# ---------------------------------------------------------------------------
# 2. Fourier transform of a signal
# ---------------------------------------------------------------------------

# Use n_bits=5 (32 points) where the DFT MPO is exact at max_bond_dim=16
n_bits_fft = 5
grid_fft = GridSpec(
    variables=(UniformGrid(0, 1, n_bits_fft),),
    layout="grouped",
)
N_fft = 1 << n_bits_fft

print("=== Fourier transform in QTT format ===")
print(f"Grid: {N_fft} points on [0, 1)")

# Build a signal: f(t) = sin(2*pi*3*t) + 0.5*cos(2*pi*7*t)
# This has frequency peaks at k=3 (magnitude N/2) and k=7 (magnitude N/4)
dt = 1.0 / N_fft
t_vals = jnp.arange(N_fft) * dt
signal = jnp.sin(2 * math.pi * 3 * t_vals) + 0.5 * jnp.cos(2 * math.pi * 7 * t_vals)

qtt_signal = fold_to_qtt(signal, grid_fft, tol=1e-12)
print(f"Signal bond dims: {qtt_signal.bond_dims}")

# Build DFT MPO (exact at this grid size)
F = fourier_mpo(grid_fft, max_bond_dim=512, tol=1e-14)
print(f"DFT MPO bond dims: {[t.shape[0] for t in F.site_tensors]}")

# Apply DFT
qtt_spectrum = F.apply(qtt_signal, method="naive", tol=1e-10, max_bond_dim=256)
print(f"Spectrum bond dims: {qtt_spectrum.bond_dims}")

# Extract the spectrum and find peaks
spectrum_dense = qtt_spectrum.to_dense()
magnitudes = jnp.abs(spectrum_dense)

# Show nonzero frequency components
print("\nFrequency spectrum (nonzero components):")
for k in range(N_fft):
    mag = float(magnitudes[k])
    if mag > 0.1:
        freq = k if k <= N_fft // 2 else k - N_fft
        print(f"  k = {freq:+3d},  |F[k]| = {mag:.4f}")

print("\nExpected peaks at k = +/-3 (sin) and k = +/-7 (cos)")
