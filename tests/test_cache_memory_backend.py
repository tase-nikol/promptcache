from promptcache.cache import SemanticCache
from promptcache.backends.memory import MemoryBackend
from promptcache.embedders.fake import FakeSemanticEmbedder
from promptcache.types import CacheMeta

def test_cache_hit_on_similar_prompt():
    backend = MemoryBackend()
    embedder = FakeSemanticEmbedder()

    cache = SemanticCache(
        backend=backend,
        embedder=embedder,
        namespace="test",
        threshold=0.0,
        ttl_seconds=3600,
        min_prompt_length=0
    )

    meta = CacheMeta(model="gpt-test", system_prompt="You are helpful.")
    called = {"n": 0}

    def llm_call():
        called["n"] += 1
        return {"text": "Hello!"}

    def extract(resp):
        return resp["text"]

    r1 = cache.get_or_set(prompt="Say hello", llm_call=llm_call, extract_text=extract, meta=meta)
    r2 = cache.get_or_set(prompt="Say hello please", llm_call=llm_call, extract_text=extract, meta=meta)

    assert r1.cache_hit is False
    assert r2.cache_hit is True
    assert called["n"] == 1
