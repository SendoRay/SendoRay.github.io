"""Pattern 3: Coordinated betting detection.

Detects multiple wallets betting on the same outcome in the same market
within a short time window, where those wallets share funding sources
(via on-chain fund flow analysis through the subgraph).
"""
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass


@dataclass
class CoordinatedResult:
    coordination_score: float       # 0-1, time clustering
    fund_correlation_score: float   # 0-1, shared funding sources
    related_wallets: list[str]      # addresses of related wallets
    composite_score: float


async def evaluate(trade: dict, recent_same_direction_trades: list[dict],
                   related_wallets: list[dict] | None) -> CoordinatedResult:
    now = trade.get("trade_timestamp", datetime.now(timezone.utc))
    if isinstance(now, str):
        now = datetime.fromisoformat(now.replace("Z", "+00:00"))

    # --- Time clustering score ---
    # How many wallets bet the same direction in the last 2 hours?
    window = timedelta(hours=2)
    recent_wallets = set()
    for t in recent_same_direction_trades:
        t_time = t.get("trade_timestamp")
        if isinstance(t_time, str):
            t_time = datetime.fromisoformat(t_time.replace("Z", "+00:00"))
        if t_time and (now - t_time) <= window:
            addr = t.get("taker_address") or t.get("maker_address")
            if addr:
                recent_wallets.add(addr)

    # 5+ wallets in 2h → score 1.0, 1 wallet → score 0.0
    coordination_score = min(1.0, max(0.0, (len(recent_wallets) - 1) / 4.0))

    # --- Fund correlation score ---
    if related_wallets:
        fund_correlation_score = min(1.0, len(related_wallets) / 3.0)
    else:
        fund_correlation_score = 0.0

    related_addrs = [r.get("address", "") for r in (related_wallets or [])]

    composite = (coordination_score * 0.5 + fund_correlation_score * 0.5)

    return CoordinatedResult(
        coordination_score=round(coordination_score, 4),
        fund_correlation_score=round(fund_correlation_score, 4),
        related_wallets=related_addrs,
        composite_score=round(composite, 4),
    )
