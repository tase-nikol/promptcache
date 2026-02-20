from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class CacheMeta(BaseModel):
    model: str = "unknown"
    system_prompt: str = ""
    tools_schema: str = ""  # JSON schema or tool/function signature string
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class CacheResult(BaseModel):
    text: str
    cache_hit: bool
    similarity: Optional[float] = None
    entry_id: Optional[str] = None

    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None

    created_at_unix: Optional[int] = None
    matched_meta: Optional[Dict[str, Any]] = None