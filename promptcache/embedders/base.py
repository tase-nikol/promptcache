from __future__ import annotations
from typing import Protocol, Sequence


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> Sequence[float]:
        """Return a vector of length dim."""
        ...
