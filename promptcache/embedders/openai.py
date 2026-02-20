from __future__ import annotations
from typing import Sequence, Optional

import numpy as np

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class OpenAIEmbedder:
    def __init__(self, model: str, api_key: Optional[str] = None, dim: Optional[int] = None):
        if OpenAI is None:
            raise RuntimeError("openai package not installed. Install with: pip install promptcache[openai]")
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.name = f"openai:{model}"
        self.dim = dim or 1536  # common default; actual dim depends on model

    def embed(self, text: str) -> Sequence[float]:
        resp = self._client.embeddings.create(model=self.model, input=text)
        vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
        vec /= (np.linalg.norm(vec) + 1e-12)
        self.dim = int(vec.shape[0])
        return vec.tolist()
