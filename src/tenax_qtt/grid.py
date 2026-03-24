"""Grid specifications and coordinate-to-site mappings for QTT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class UniformGrid:
    """One variable on [a, b] discretized into 2^n_bits points."""

    a: float
    b: float
    n_bits: int
    include_endpoint: bool = False

    def __post_init__(self) -> None:
        if self.n_bits < 1:
            raise ValueError(f"n_bits must be >= 1, got {self.n_bits}")
        if self.a >= self.b:
            raise ValueError(f"a must be < b, got a={self.a}, b={self.b}")

    @property
    def n_points(self) -> int:
        return 1 << self.n_bits

    @property
    def dx(self) -> float:
        if self.include_endpoint:
            return (self.b - self.a) / (self.n_points - 1)
        return (self.b - self.a) / self.n_points


@dataclass(frozen=True)
class GridSpec:
    """Multi-dimensional grid with configurable layout."""

    variables: tuple[UniformGrid, ...]
    layout: Literal["interleaved", "fused", "grouped"]

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("At least one variable required")
        if self.layout == "interleaved":
            nbits = {v.n_bits for v in self.variables}
            if len(nbits) > 1:
                raise ValueError(
                    "Interleaved layout requires all variables to have the "
                    f"same n_bits, got {sorted(nbits)}"
                )
