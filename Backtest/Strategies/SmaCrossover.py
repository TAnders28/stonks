"""SMA fast/slow crossover. Signal uses closes[0..i] only."""
from __future__ import annotations

NAME = "sma_crossover"
DESCRIPTION = "Long when SMA(fast) > SMA(slow); flat otherwise. Decision on close i, fill next open."


def make_signal(fast: int = 20, slow: int = 50):
    assert fast < slow

    def signal(bars: list[dict], i: int) -> int:
        if i + 1 < slow:
            return 0
        closes = [bars[j]["close"] for j in range(i - slow + 1, i + 1)]
        sma_slow = sum(closes) / slow
        sma_fast = sum(closes[-fast:]) / fast
        return 1 if sma_fast > sma_slow else 0

    signal.fast = fast
    signal.slow = slow
    signal.name = f"sma_{fast}_{slow}"
    signal.description = DESCRIPTION
    return signal
