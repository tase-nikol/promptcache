from promptcache.normalize import normalize_text
from promptcache.hashing import stable_context_hash

def test_normalize_text():
    assert normalize_text("  hello   world \n") == "hello world"

def test_stable_context_hash_changes():
    h1 = stable_context_hash("sys", "tools")
    h2 = stable_context_hash("sys2", "tools")
    assert h1 != h2
