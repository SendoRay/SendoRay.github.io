"""Pattern 5: Odds manipulation detection.

Detects large bets that cause significant probability shifts followed by
quick reversions, which can indicate wash trading or market manipulation
rather than genuine informed trading.
"""
import math
from dataclasses import dataclass


@dataclass
class OddsManipulationResult:
    price_impact_score: float    # 0-1, how much the price moved
    reversal_score: float        # 0-1, how much it reverted after
    composite_score: float


def evaluate(trade: dict, price_before: float | None,
             price_after: float | None, price_later: float | None,
             trade_amount_usd: float | None = None,
             market_liquidity: float | None = None) -> OddsManipulationResult:
    amount = trade_amount_usd or trade.get("amount_usd") or 0

    # --- Price impact score ---
    if price_before is not None and price_after is not None and price_before > 0:
        raw_impact = abs(price_after - price_before) / price_before
        # Normalize: 20% move → 1.0, 2% move → 0.1
        price_impact_score = min(1.0, raw_impact / 0.20)
    else:
        price_impact_score = 0.0

    # --- Reversal score ---
    # How much did the price revert from the post-trade level?
    if price_after is not None and price_later is not None and price_before is not None:
        move = abs(price_after - price_before)
        reversion = abs(price_later - price_before)
        if move > 0:
            reversal_ratio = reversion / move  # 1.0 = full revert, 0.0 = no revert
            reversal_score = max(0.0, min(1.0, (reversal_ratio - 0.5) * 2))
        else:
            reversal_score = 0.0
    else:
        reversal_score = 0.0

    # Composite: both impact and reversal needed for manipulation signal
    composite = price_impact_score * 0.4 + reversal_score * 0.6

    return OddsManipulationResult(
        price_impact_score=round(price_impact_score, 4),
        reversal_score=round(reversal_score, 4),
        composite_score=round(composite, 4),
    )
