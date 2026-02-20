from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, List
import json

import numpy as np
import redis

from .base import VectorBackend, Candidate


def _vec_to_bytes(vec: Sequence[float]) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    arr /= (np.linalg.norm(arr) + 1e-12)
    return arr.tobytes()


class RedisVectorBackend(VectorBackend):
    """
    Requires Redis Stack / RediSearch with VECTOR similarity search.
    Uses HNSW + COSINE.
    """

    def __init__(
        self,
        url: str,
        *,
        index_prefix: str = "llmcache",
        dim: int,
        metric: str = "COSINE",
    ):
        self._r = redis.Redis.from_url(url, decode_responses=False)
        self._index_prefix = index_prefix
        self._dim = dim
        self._metric = metric

    def _index_name(self, namespace: str) -> str:
        return f"{self._index_prefix}:{namespace}:idx"

    def _key_prefix(self, namespace: str) -> str:
        return f"{self._index_prefix}:{namespace}:entry:"

    def ensure_index(self, namespace: str) -> None:
        """
        Create RediSearch index if missing.
        """
        idx = self._index_name(namespace)
        prefix = self._key_prefix(namespace)

        # Detect index existence
        try:
            self._r.execute_command("FT.INFO", idx)
            return
        except redis.ResponseError:
            pass

        # Create index
        # Fields:
        # - embedding VECTOR HNSW
        # - model TAG
        # - ctx_hash TAG  (system+tools hash)
        # - embedder TAG
        schema = [
            "SCHEMA",
            "embedding", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32",
            "DIM", str(self._dim),
            "DISTANCE_METRIC", self._metric,
            "model", "TAG",
            "ctx_hash", "TAG",
            "embedder", "TAG",
        ]

        self._r.execute_command(
            "FT.CREATE", idx,
            "ON", "HASH",
            "PREFIX", "1", prefix,
            *schema
        )

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
        self.ensure_index(namespace)

        key = self._key_prefix(namespace) + entry_id
        payload = {
            "text": text,
            "meta_json": json.dumps(meta, ensure_ascii=False),
            "tokens_in": str(tokens_in) if tokens_in is not None else "",
            "tokens_out": str(tokens_out) if tokens_out is not None else "",
            "cost_usd": str(cost_usd) if cost_usd is not None else "",
            "created_at_unix": str(created_at_unix),
            "embedding": _vec_to_bytes(vector),
        }
        # tags are indexed fields
        payload["model"] = tags.get("model", "")
        payload["ctx_hash"] = tags.get("ctx_hash", "")
        payload["embedder"] = tags.get("embedder", "")

        # HSET + EXPIRE
        pipe = self._r.pipeline()
        pipe.hset(key, mapping=payload)
        pipe.expire(key, int(ttl_seconds))
        pipe.execute()

    def query(
        self,
        *,
        namespace: str,
        vector: Sequence[float],
        k: int,
        tags: Dict[str, str],
    ) -> Tuple[Optional[Candidate], Sequence[Candidate]]:
        self.ensure_index(namespace)

        idx = self._index_name(namespace)
        qvec = _vec_to_bytes(vector)
        print(1, idx, qvec)
        print(2, tags)
        # Filter by tags (exact match)
        filters = []
        for field, val in tags.items():
            if not val:
                continue
            # TAG query needs braces
            filters.append(f"@{field}:{{{self._escape_tag(val)}}}")
        base = " ".join(filters) if filters else "*"

        # KNN query
        # Returns __embedding_score as distance; for COSINE distance lower is better.
        # query = f"{base})=>[KNN {k} @embedding $vec AS dist]"
        query = f"(*)=>[KNN {k} @embedding $vec AS dist]"

        res = self._r.execute_command(
            "FT.SEARCH", idx,
            query,
            "PARAMS", "2", "vec", qvec,
            "SORTBY", "dist", "ASC",
            "RETURN", "7", "text", "meta_json", "tokens_in", "tokens_out", "cost_usd", "created_at_unix", "dist",
            "DIALECT", "2",
        )
        print(3, res)
        # res format: [count, key1, [field, value, ...], key2, [...], ...]
        if not res or res[0] == 0:
            return None, []
        print(res)
        cands: List[Candidate] = []
        for i in range(1, len(res), 2):
            raw_key = res[i]
            fields = res[i + 1]
            d = {fields[j].decode("utf-8"): fields[j + 1] for j in range(0, len(fields), 2)}

            dist = float(d["dist"].decode("utf-8"))
            # Convert COSINE distance to cosine similarity approx:
            similarity = 1.0 - dist

            entry_id = raw_key.decode("utf-8").split(":")[-1]
            text = d["text"].decode("utf-8")
            meta_json = d["meta_json"].decode("utf-8")
            meta = json.loads(meta_json) if meta_json else {}

            def _parse_int(b: bytes) -> Optional[int]:
                s = b.decode("utf-8")
                return int(s) if s else None

            def _parse_float(b: bytes) -> Optional[float]:
                s = b.decode("utf-8")
                return float(s) if s else None

            cand = Candidate(
                entry_id=entry_id,
                text=text,
                similarity=similarity,
                meta=meta,
                tokens_in=_parse_int(d["tokens_in"]),
                tokens_out=_parse_int(d["tokens_out"]),
                cost_usd=_parse_float(d["cost_usd"]),
                created_at_unix=_parse_int(d["created_at_unix"]),
            )
            cands.append(cand)
        print("REDIS CANDIDATES:", cands)

        filtered = []
        for cand in cands:
            stored_meta = cand.meta.get("meta", {})

            if (
                    stored_meta.get("model") == tags["model"]
                    and cand.meta.get("prompt_hash")  # optional check
            ):
                filtered.append(cand)

        best = filtered[0] if filtered else None
        print("BEST:", best)
        return best, filtered

    def _escape_tag(self, value: str) -> str:
        # Only escape backslash and curly braces
        return (
            value.replace("\\", "\\\\")
            .replace("{", "\\{")
            .replace("}", "\\}")
        )