from __future__ import annotations

import re


_ws_re = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    """
    Normalize for embedding + hashing:
    - strip
    - collapse whitespace
    """
    s = s.strip()
    s = _ws_re.sub(" ", s)
    return s
