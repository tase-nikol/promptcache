from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Optional

from .types import CacheMeta, CacheResult
from .normalize import normalize_text
from .hashing import stable_context_hash, sha256_hex
from .cost import CostTracker
from .embedders.base import Embedder
from .backends.base import VectorBackend

import redis


class SemanticCache:
    def __init__(
        self,
        *,
        backend: VectorBackend,
        embedder: Embedder,
        namespace: str,
        threshold: float = 0.92,
        ttl_seconds: int = 60 * 60 * 24 * 7,
        k: int = 8,
        cost_tracker: Optional[CostTracker] = None,
        min_prompt_length: int = 16,
        max_answer_chars: int = 50_000,
    ):
        self.backend = backend
        self.embedder = embedder
        self.namespace = namespace
        self.threshold = float(threshold)
        self.ttl_seconds = int(ttl_seconds)
        self.k = int(k)
        self.cost_tracker = cost_tracker or CostTracker(pricing=None)
        self.min_prompt_length = int(min_prompt_length)
        self.max_answer_chars = int(max_answer_chars)
        self._hits = 0
        self._misses = 0

    def _build_cache_text(self, prompt: str, meta: CacheMeta) -> str:
        # Include system prompt into semantic key by default (safer)
        parts = []
        if meta.system_prompt:
            parts.append("SYSTEM:\n" + normalize_text(meta.system_prompt))
        parts.append("USER:\n" + normalize_text(prompt))
        if meta.tools_schema:
            parts.append("TOOLS:\n" + normalize_text(meta.tools_schema))
        return "\n\n".join(parts)

    def get_or_set(
        self,
        *,
        prompt: str,
        llm_call: Callable[[], Any],
        extract_text: Callable[[Any], str],
        meta: Optional[CacheMeta] = None,
        tokens_in: Optional[Callable[[Any], int]] = None,
        tokens_out: Optional[Callable[[Any], int]] = None,
        do_not_cache: Optional[Callable[[str, CacheMeta, str], bool]] = None,
    ) -> CacheResult:
        meta = meta or CacheMeta()
        prompt_n = normalize_text(prompt)

        # tiny prompts often too risky/noisy
        if len(prompt_n) < self.min_prompt_length:
            resp = llm_call()
            text = extract_text(resp)
            return CacheResult(text=text, cache_hit=False)

        cache_text = self._build_cache_text(prompt_n, meta)
        vec = self.embedder.embed(cache_text)

        ctx_hash = stable_context_hash(meta.system_prompt, meta.tools_schema)

        tags = {
            "model": meta.model or "unknown",
            "ctx_hash": ctx_hash,
            "embedder": getattr(self.embedder, "name", "unknown"),
        }

        best, _ = self.backend.query(namespace=self.namespace, vector=vec, k=self.k, tags=tags) # nearest neighbors
        if best is not None and best.similarity >= self.threshold:
            self._hits += 1

            return CacheResult(
                text=best.text,
                cache_hit=True,
                similarity=best.similarity,
                entry_id=best.entry_id,
                tokens_in=best.tokens_in,
                tokens_out=best.tokens_out,
                cost_usd=best.cost_usd,
                created_at_unix=best.created_at_unix,
            )

        # Miss: call LLM
        resp = llm_call()
        text = extract_text(resp)
        text = text[: self.max_answer_chars]

        # Optional token extraction
        tin = tokens_in(resp) if tokens_in else None
        tout = tokens_out(resp) if tokens_out else None
        cost = self.cost_tracker.compute(meta.model, tin, tout) if (tin is not None and tout is not None) else None

        if do_not_cache and do_not_cache(prompt_n, meta, text):
            return CacheResult(
                text=text,
                cache_hit=False,
                tokens_in=tin,
                tokens_out=tout,
                cost_usd=cost,
            )

        created = int(time.time())
        entry_id = uuid.uuid4().hex
        self.backend.upsert(
            namespace=self.namespace,
            entry_id=entry_id,
            vector=vec,
            text=text,
            ttl_seconds=self.ttl_seconds,
            tags=tags,
            meta={
                "prompt_hash": sha256_hex(cache_text),
                "meta": meta.model_dump(),
            },
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost,
            created_at_unix=created,
        )
        self._misses += 1

        return CacheResult(
            text=text,
            cache_hit=False,
            entry_id=entry_id,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost,
            created_at_unix=created,
        )

    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": round(hit_rate, 4),
            "hit_rate_percent": round(hit_rate * 100, 2),
        }

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0