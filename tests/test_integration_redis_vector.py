import os
import pytest

from promptcache.cache import SemanticCache
from promptcache.backends.redis_vector import RedisVectorBackend
from promptcache.embedders.fake import FakeSemanticEmbedder
from promptcache.types import CacheMeta


@pytest.mark.skipif(os.environ.get("REDIS_URL") is None, reason="Set REDIS_URL to run integration test")
@pytest.mark.integration
def test_redis_vector_backend_hit():
    embedder = FakeSemanticEmbedder()
    backend = RedisVectorBackend(url=os.environ["REDIS_URL"], dim=embedder.dim)
    cache = SemanticCache(backend=backend, embedder=embedder, namespace="itest", threshold=0.0, min_prompt_length=0)

    meta = CacheMeta(model="gpt-test", system_prompt="You are helpful.")
    called = {"n": 0}

    def llm_call():
        called["n"] += 1
        return {"text": "cached answer"}

    def extract(r):
        return r["text"]

    r1 = cache.get_or_set(prompt="Say hello about Redis", llm_call=llm_call, extract_text=extract, meta=meta)
    r2 = cache.get_or_set(prompt="Say hello about Redis briefly", llm_call=llm_call, extract_text=extract, meta=meta)

    assert r1.cache_hit is False
    assert r2.cache_hit is True
    assert called["n"] == 1
