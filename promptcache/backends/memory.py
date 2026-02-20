from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple, List
from dataclasses import dataclass
import numpy as np
from .base import VectorBackend, Candidate


@dataclass
class _MemEntry:
    entry_id: str
    vec: np.ndarray
    text: str
    tags: Dict[str, str]
    meta: Dict[str, Any]
    tokens_in: Optional[int]
    tokens_out: Optional[int]
    cost_usd: Optional[float]
    created_at_unix: Optional[int]


class MemoryBackend(VectorBackend):
    def __init__(self):
        self._store: Dict[str, List[_MemEntry]] = {}

    def upsert(
        self,
        *,
        namespace: str,
        entry_id: str,
        vector: Sequence[float],
        text: str,
        ttl_seconds: int,  # ignored in memory backend
        tags: Dict[str, str],
        meta: Dict[str, Any],
        tokens_in: Optional[int],
        tokens_out: Optional[int],
        cost_usd: Optional[float],
        created_at_unix: int,
    ) -> None:
        v = np.asarray(vector, dtype=np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        self._store.setdefault(namespace, [])
        # overwrite if exists
        self._store[namespace] = [e for e in self._store[namespace] if e.entry_id != entry_id]
        self._store[namespace].append(
            _MemEntry(entry_id, v, text, dict(tags), dict(meta), tokens_in, tokens_out, cost_usd, created_at_unix)
        )

    def query(
        self,
        *,
        namespace: str,
        vector: Sequence[float],
        k: int,
        tags: Dict[str, str],
    ) -> Tuple[Optional[Candidate], Sequence[Candidate]]:

        entries = self._store.get(namespace, [])
        if not entries:
            return None, []

        q = np.asarray(vector, dtype=np.float32)
        q /= (np.linalg.norm(q) + 1e-12)

        cands: List[Candidate] = []
        for e in entries:
            if any(e.tags.get(k_) != v_ for k_, v_ in tags.items()):
                continue
            sim = float(np.dot(q, e.vec))
            cands.append(
                Candidate(
                    entry_id=e.entry_id,
                    text=e.text,
                    similarity=sim,
                    meta=e.meta,
                    tokens_in=e.tokens_in,
                    tokens_out=e.tokens_out,
                    cost_usd=e.cost_usd,
                    created_at_unix=e.created_at_unix,
                )
            )
        cands.sort(key=lambda c: c.similarity, reverse=True)
        top = cands[0] if cands else None
        return top, cands[:k]
