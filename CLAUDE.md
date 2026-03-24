# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

`tenax-qtt` is a Python package for Quantic Tensor Train (QTT) algorithms, targeting applied math: function approximation on fine grids, operator arithmetic, Fourier transforms, integration, and linear algebra in compressed QTT format.

It depends on `tenax` (the tensor network library) and wraps `tenax.FiniteMPS` for QTT storage.

## Build & Test

- **Package manager**: `uv`
- **Install**: `uv pip install -e ".[dev]"` (requires tenax installed, e.g., `uv pip install -e ../tenax`)
- **Test**: `uv run pytest -m core` (fast), `uv run pytest` (all)
- **Lint**: `uv run ruff check src/ tests/`

## Git Workflow

- Always open a PR instead of pushing directly to `main`.
- Merge PRs with `gh pr merge <number> --squash --delete-branch --auto`.
- **Pytest markers**: `core` (fast), `algorithm` (full runs), `slow` (expensive). CI required checks run `pytest -m core`.

## Coding Rules

- Follow tenax conventions: label-based tensor operations, polymorphic across DenseTensor/SymmetricTensor.
- QTT site tensors must follow tenax's MPS label convention: `"v{i-1}_{i}"` (left bond), `"p{i}"` (physical), `"v{i}_{i+1}"` (right bond).
- Keep modules focused and under ~400 lines each.

## Design Spec

See `docs/plans/2026-03-24-tenax-qtt-design.md` for the full design document.
