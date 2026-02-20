from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class Candidate:
    entry_id: str
    text: str
    similarity: float
    meta: Dict[str, Any]
    tokens_in: Optional[int]
    tokens_out: Optional[int]
    cost_usd: Optional[float]
    created_at_unix: Optional[int]


class VectorBackend(Protocol):
    """
    A backend that can:
    - upsert entries with embedding vector
    - query nearest neighbor candidates with filters
    """
    def upsert(
        self,
        *,
        namespace: str,
        entry_id: str,
        vector: Sequence[float],
        text: str,
        ttl_seconds: int,
        tags: Dict[str, str],
        meta: Dict[str, Any],
        tokens_in: Optional[int],
        tokens_out: Optional[int],
        cost_usd: Optional[float],
        created_at_unix: int,
    ) -> None:
        ...

    def query(
        self,
        *,
        namespace: str,
        vector: Sequence[float],
        k: int,
        tags: Dict[str, str],
    ) -> Tuple[Optional[Candidate], Sequence[Candidate]]:
        ...
