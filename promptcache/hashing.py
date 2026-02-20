from __future__ import annotations

import hashlib
from .normalize import normalize_text


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stable_context_hash(system_prompt: str, tools_schema: str) -> str:
    sys_n = normalize_text(system_prompt)
    tools_n = normalize_text(tools_schema)
    return sha256_hex(f"sys:{sys_n}\ntools:{tools_n}")
