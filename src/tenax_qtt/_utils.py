"""Shared utilities for tenax-qtt."""

from __future__ import annotations

import numpy as np
from tenax import FlowDirection, TensorIndex, U1Symmetry

_sym = U1Symmetry()


def trivial_index(dim: int, flow: FlowDirection, label: str) -> TensorIndex:
    """Create a TensorIndex with trivial (all-zero) U(1) charges."""
    return TensorIndex(_sym, np.zeros(dim, dtype=np.int32), flow, label=label)
