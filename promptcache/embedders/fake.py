from __future__ import annotations
import numpy as np


class FakeSemanticEmbedder:
    name = "fake-semantic"
    dim = 8

    def embed(self, text: str):
        # Extremely naive semantic approximation:
        # vector based on word presence
        words = set(text.lower().split())

        base = {
            "say": 1.0,
            "hello": 1.0,
            "please": 0.2,
        }

        vec = []
        for key in ["say", "hello", "please", "hi", "thanks", "help", "you", "are"]:
            vec.append(base.get(key, 0.0) if key in words else 0.0)

        # pad to fixed dim
        while len(vec) < self.dim:
            vec.append(0.0)

        return vec[:self.dim]
