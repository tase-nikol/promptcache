from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional


class Pricing(Protocol):
    def cost_usd(self, model: str, tokens_in: int, tokens_out: int) -> float: ...


@dataclass(frozen=True)
class SimplePricing(Pricing):
    """
    Provide per-1M token prices. Users can replace this with their own.
    """
    input_per_1m: float
    output_per_1m: float

    def cost_usd(self, model: str, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.input_per_1m + tokens_out * self.output_per_1m) / 1_000_000.0


@dataclass
class CostTracker:
    pricing: Optional[Pricing] = None

    def compute(self, model: str, tokens_in: int, tokens_out: int) -> Optional[float]:
        if self.pricing is None:
            return None
        return float(self.pricing.cost_usd(model=model, tokens_in=tokens_in, tokens_out=tokens_out))
