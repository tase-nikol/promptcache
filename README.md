PromptCache
===========

> Reduce your LLM API costs by 30--70% with semantic caching.

PromptCache reuses LLM responses for **semantically similar prompts**, not just exact string matches.

If two users ask:

-   "Explain Redis in simple terms"

-   "Can you explain Redis simply?"

You shouldn't pay twice.

PromptCache makes sure you don't.

* * * * * 
![License](https://img.shields.io/badge/license-MIT-green)

* * * * *

The Problem
--------------

If you're using OpenAI or any LLM API in production, you're likely paying repeatedly for:

-   The same question phrased differently

-   Similar support requests across users

-   Slight variations in prompts

-   Background job retries

-   RAG pipelines returning near-identical queries

Traditional caching only works for **exact matches**.

LLMs need **semantic caching**.

* * * * *

What PromptCache Does
-----------------------

1.  Embeds your prompt into a vector

2.  Searches Redis for similar past prompts

3.  If similarity ≥ threshold → returns cached response

4.  Otherwise → calls the LLM and stores the result

```sql
User Prompt
     ↓
Embed → Redis Vector Search
     ↓
Hit? → Return cached answer
Miss? → Call LLM → Store result
```

* * * * *

10-Second Example
--------------------

```python
from promptcache import SemanticCache
from promptcache.backends.redis_vector import RedisVectorBackend
from promptcache.embedders.openai import OpenAIEmbedder
from promptcache.types import CacheMeta

embedder = OpenAIEmbedder(model="text-embedding-3-small")

backend = RedisVectorBackend(
    url="redis://localhost:6379/0",
    dim=embedder.dim,
)

cache = SemanticCache(
    backend=backend,
    embedder=embedder,
    namespace="support-bot",
    threshold=0.92,
)

meta = CacheMeta(
    model="gpt-4.1-mini",
    system_prompt="You are a helpful support assistant.",
)

result = cache.get_or_set(
    prompt="How do I reset my password?",
    llm_call=my_llm_call,
    extract_text=lambda r: r.output_text,
    meta=meta,
)

print(result.cache_hit)  # True or False`
```
That's it.

* * * * *

Example Impact
-----------------

In a SaaS support assistant:

-   62% cache hit rate

-   48% reduction in token usage

-   44% reduction in API spend

Your mileage depends on workload --- but high-volume, repetitive systems benefit the most.

* * * * *

Production-Ready Design
--------------------------

PromptCache isolates cache entries by:

-   `namespace`

-   `model`

-   `system_prompt`

-   `tools_schema`

-   `embedder`

This prevents cross-context contamination.

Additional features:

-   ✅ Redis HNSW vector search (cosine similarity)

-   ✅ TTL support

-   ✅ Hit-rate statistics

-   ✅ Optional cost tracking

-   ✅ In-memory backend (for testing)

-   ✅ Framework-agnostic (no LangChain dependency)

* * * * *

Installation
---------------

```bash
pip install promptcache-ai
```

Optional OpenAI embedder:
```bash
pip install promptcache-ai[openai]
```

* * * * *

Redis Setup
--------------

PromptCache requires **Redis Stack** (RediSearch with vector support).

Run locally:

```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```
Verify:

```bash
redis-cli MODULE LIST
```

You should see:

```sql
search
```

* * * * *

Stats
--------

Measure impact:

```python
print(cache.stats())
```

Example:

```json
{
    "hits": 1240,
    "misses": 860,
    "total": 2100,
    "hit_rate_percent": 59.05
}
```

* * * * *

When It Helps Most
---------------------

-   Customer support bots

-   Internal copilots

-   FAQ systems

-   Knowledge assistants

-   Deterministic / low-temperature tasks

-   High-volume similar prompts

* * * * *

When It May Not Help
-----------------------

-   Highly personalized prompts

-   Creative high-temperature tasks

-   Frequently changing context

* * * * *

Testing
----------

Run unit tests:

```python
pytest
```

Run Redis integration tests:

```bash
export REDIS_URL="redis://localhost:6379/0"
pytest
```