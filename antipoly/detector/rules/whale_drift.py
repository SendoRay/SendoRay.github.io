"""Pattern 4: Whale behavior drift detection.

Detects when a known large-volume wallet suddenly trades in a new market
category or makes an atypical bet relative to its historical pattern.
"""
from dataclasses import dataclass


@dataclass
class WhaleDriftResult:
    behavior_drift_score: float   # 0-1, how different from historical pattern
    new_market_score: float       # 0-1, whether the market category is new
    composite_score: float


def evaluate(trade: dict, wallet_total_volume_usd: float,
             wallet_markets_traded: int, wallet_total_trades: int,
             market_category: str | None, wallet_known_categories: list[str] | None,
             min_whale_volume: float = 50000) -> WhaleDriftResult:
    # Only evaluate for known whales
    is_whale = wallet_total_volume_usd >= min_whale_volume

    if not is_whale or wallet_total_trades < 10:
        return WhaleDriftResult(
            behavior_drift_score=0.0,
            new_market_score=0.0,
            composite_score=0.0,
        )

    # --- New market category score ---
    known_cats = set(wallet_known_categories or [])
    if market_category and market_category not in known_cats:
        new_market_score = 1.0
    elif market_category and market_category in known_cats:
        new_market_score = 0.0
    else:
        new_market_score = 0.5  # unknown category

    # --- Behavior drift score ---
    # How unusual is this trade size for this wallet?
    amount = trade.get("amount_usd") or 0
    avg_trade_size = wallet_total_volume_usd / max(wallet_total_trades, 1)

    if avg_trade_size > 0:
        size_ratio = amount / avg_trade_size
        # 10x their average → drift 1.0, 1x → drift 0.0
        behavior_drift_score = min(1.0, max(0.0, (size_ratio - 1.0) / 9.0))
    else:
        behavior_drift_score = 0.5

    composite = (behavior_drift_score * 0.5 + new_market_score * 0.5)

    return WhaleDriftResult(
        behavior_drift_score=round(behavior_drift_score, 4),
        new_market_score=round(new_market_score, 4),
        composite_score=round(composite, 4),
    )
