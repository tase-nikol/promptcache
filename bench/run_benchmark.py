from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import tiktoken
from sentence_transformers import SentenceTransformer

from promptcache.cache import SemanticCache
from promptcache.backends.memory import MemoryBackend
from promptcache.backends.redis_vector import RedisVectorBackend
from promptcache.types import CacheMeta


# ---------- Embedding ----------
class SentenceTransformersEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        # dim from a dummy run
        self.dim = len(self._model.encode(["dim-check"], normalize_embeddings=True)[0])
        self.name = f"st:{model_name}"

    def embed(self, text: str):
        # promptcache expects list[float]
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()


# ---------- Token / cost estimation ----------
@dataclass
class Pricing:
    # dollars per 1M tokens
    input_per_1m: float = 0.50
    output_per_1m: float = 1.50


class TokenEstimator:
    def __init__(self, model: str = "gpt-4o-mini"):
        # any encoding works for consistent estimation; this is fine for relative comparisons
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except Exception:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


def estimate_cost_usd(tokens_in: int, tokens_out: int, pricing: Pricing) -> float:
    return (tokens_in / 1_000_000) * pricing.input_per_1m + (tokens_out / 1_000_000) * pricing.output_per_1m


# ---------- Workload loading ----------
def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ---------- LLM stub ----------
def make_llm_stub(intent_to_answer: dict[str, str]) -> Callable[[str], dict[str, Any]]:
    def llm_call(prompt: str) -> dict[str, Any]:
        # Return an answer based on intent_id embedded in the prompt mapping
        # The benchmark driver passes the intent_id separately; we’ll inject it via closure.
        raise RuntimeError("llm_call should be wrapped per item with the intent answer")
    return llm_call


# ---------- Benchmark ----------
@dataclass
class ItemResult:
    workload: str
    intent_id: str
    prompt: str
    cache_hit: bool
    similarity: float | None
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    unsafe_hit: bool


def run_workload(
    *,
    name: str,
    rows: list[dict],
    cache: SemanticCache,
    meta: CacheMeta,
    token_estimator: TokenEstimator,
    pricing: Pricing,
    fixed_output_tokens: int = 180,
) -> list[ItemResult]:
    # Deterministic canned answer per intent_id
    intent_answer = {r["intent_id"]: f"INTENT={r['intent_id']}\n(workload={name})" for r in rows}

    results: list[ItemResult] = []
    for r in rows:
        prompt = r["prompt"]
        intent_id = r["intent_id"]

        # token estimates
        tokens_in = token_estimator.count(prompt)
        tokens_out = fixed_output_tokens
        cost = estimate_cost_usd(tokens_in, tokens_out, pricing)

        called = {"ran": False}

        def llm_call():
            called["ran"] = True
            return {"text": intent_answer[intent_id], "intent_id": intent_id}

        def extract(resp: dict[str, Any]) -> str:
            return resp["text"]

        t0 = time.perf_counter()
        meta_with_intent = CacheMeta(
            model=meta.model,
            system_prompt=meta.system_prompt,
            tools_schema=meta.tools_schema,
            temperature=meta.temperature,
            top_p=meta.top_p,
            extra={"intent_id": intent_id},
        )
        out = cache.get_or_set(
            prompt=prompt,
            llm_call=llm_call,
            extract_text=extract,
            meta=meta_with_intent,
        )
        dt = (time.perf_counter() - t0) * 1000.0

        unsafe_hit = False

        if out.cache_hit:
            cached_intent = None

            if out.matched_meta:
                try:
                    cached_intent = out.matched_meta["meta"]["extra"]["intent_id"]
                except Exception:
                    cached_intent = None

            # Only mark unsafe if we successfully retrieved a cached intent
            if cached_intent is not None:
                unsafe_hit = cached_intent != intent_id
            else:
                # If we can't retrieve intent metadata, consider it unsafe
                unsafe_hit = True

        results.append(
            ItemResult(
                workload=name,
                intent_id=intent_id,
                prompt=prompt,
                cache_hit=bool(out.cache_hit),
                similarity=out.similarity,
                latency_ms=dt,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
                unsafe_hit=unsafe_hit,
            )
        )

    return results


def summarize(results: list[ItemResult]) -> dict[str, Any]:
    n = len(results)
    hits = sum(1 for r in results if r.cache_hit)
    misses = n - hits
    unsafe_hits = sum(1 for r in results if r.unsafe_hit)
    hit_rate = hits / n if n else 0.0
    unsafe_hit_rate = (unsafe_hits / hits) if hits else 0.0

    baseline_tokens = sum(r.tokens_in + r.tokens_out for r in results)  # no cache: pay always
    cached_tokens = sum((0 if r.cache_hit else (r.tokens_in + r.tokens_out)) for r in results)  # pay only on misses
    token_savings = 1.0 - (cached_tokens / baseline_tokens) if baseline_tokens else 0.0

    baseline_cost = sum(r.cost_usd for r in results)
    cached_cost = sum((0.0 if r.cache_hit else r.cost_usd) for r in results)
    cost_savings = 1.0 - (cached_cost / baseline_cost) if baseline_cost else 0.0

    lat_p50 = _percentile([r.latency_ms for r in results], 50)
    lat_p95 = _percentile([r.latency_ms for r in results], 95)

    return {
        "n": n,
        "hits": hits,
        "misses": misses,
        "hit_rate": hit_rate,
        "unsafe_hits": unsafe_hits,
        "unsafe_hit_rate": unsafe_hit_rate,
        "baseline_tokens": baseline_tokens,
        "cached_tokens": cached_tokens,
        "token_savings": token_savings,
        "baseline_cost_usd": baseline_cost,
        "cached_cost_usd": cached_cost,
        "cost_savings": cost_savings,
        "latency_p50_ms": lat_p50,
        "latency_p95_ms": lat_p95,
    }


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs2 = sorted(xs)
    k = int(round((p / 100.0) * (len(xs2) - 1)))
    k = max(0, min(k, len(xs2) - 1))
    return xs2[k]


def main() -> None:
    root = Path(__file__).parent
    support = load_jsonl(root / "workloads" / "support.jsonl")
    rag = load_jsonl(root / "workloads" / "rag.jsonl")
    creative = load_jsonl(root / "workloads" / "creative.jsonl")

    backend_kind = os.environ.get("BENCH_BACKEND", "memory").lower()

    thresholds = [0.82, 0.85, 0.88, 0.90, 0.92]

    embedder = SentenceTransformersEmbedder(os.environ.get("BENCH_EMBEDDER", "all-MiniLM-L6-v2"))
    meta = CacheMeta(
        model=os.environ.get("BENCH_MODEL", "gpt_bench"),
        system_prompt="You are a helpful assistant.",
        tools_schema="",
    )

    token_estimator = TokenEstimator()
    pricing = Pricing(
        input_per_1m=float(os.environ.get("BENCH_PRICE_IN", "0.50")),
        output_per_1m=float(os.environ.get("BENCH_PRICE_OUT", "1.50")),
    )

    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []

    for thr in thresholds:
        for name, rows in [("support", support), ("rag", rag), ("creative", creative)]:
            # fresh backend per workload to avoid cross-contamination
            if backend_kind == "redis":
                url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
                backend = RedisVectorBackend(url=url, dim=embedder.dim)
            else:
                backend = MemoryBackend()

            cache = SemanticCache(
                backend=backend,
                embedder=embedder,
                namespace=f"bench_{backend_kind}_{name}_{thr}",
                threshold=thr,
                ttl_seconds=60 * 60 * 24,
                min_prompt_length=0,
            )

            results = run_workload(
                name=name,
                rows=rows,
                cache=cache,
                meta=meta,
                token_estimator=token_estimator,
                pricing=pricing,
            )

            s = summarize(results)
            record = {
                "backend": backend_kind,
                "embedder": embedder.name,
                "threshold": thr,
                "workload": name,
                "summary": s,
            }
            all_runs.append(record)
            print(json.dumps(record, indent=2))

    (out_dir / "results.json").write_text(json.dumps(all_runs, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
