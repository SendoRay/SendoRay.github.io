"""Pattern 1: New wallet + low probability + large bet (Maduro type).

Detects trades where a wallet with no/short history bets on a low-probability
outcome with a disproportionately large amount. This is the signature pattern
of insider trading on prediction markets.
"""
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass


@dataclass
class NewWalletResult:
    wallet_age_score: float       # 0-1, newer = higher
    bet_size_score: float         # 0-1, larger relative to market avg = higher
    low_probability_score: float  # 0-1, lower probability = higher
    composite_score: float        # weighted combination
    is_hard_trigger: bool         # true if all three conditions met strongly


async def evaluate(trade: dict, wallet_first_seen: datetime | None,
                   wallet_trade_count: int, market_probability: float | None,
                   market_avg_trade_usd: float | None,
                   max_age_days: int = 7, low_prob_max: float = 0.10,
                   min_amount_usd: float = 10000) -> NewWalletResult:
    now = datetime.now(timezone.utc)

    # --- Wallet age score ---
    if wallet_first_seen is None or wallet_trade_count == 0:
        wallet_age_score = 1.0  # completely new wallet
    else:
        age_days = (now - wallet_first_seen).total_seconds() / 86400
        if age_days <= 0:
            wallet_age_score = 1.0
        elif age_days >= 30:
            wallet_age_score = 0.0
        else:
            wallet_age_score = max(0.0, 1.0 - (age_days / 30.0))

    # --- Low probability score ---
    if market_probability is None:
        low_probability_score = 0.5  # unknown, neutral
    elif market_probability <= 0:
        low_probability_score = 1.0
    elif market_probability >= 0.5:
        low_probability_score = 0.0
    else:
        # Linear: 0% prob → 1.0, 50% prob → 0.0
        low_probability_score = 1.0 - (market_probability / 0.5)

    # --- Bet size score ---
    amount = trade.get("amount_usd") or 0
    if market_avg_trade_usd and market_avg_trade_usd > 0:
        ratio = amount / market_avg_trade_usd
        # 10x average → score 1.0, 1x average → score 0.0
        bet_size_score = min(1.0, max(0.0, (ratio - 1.0) / 9.0))
    else:
        # Absolute scale: $50k+ → 1.0, $1k → 0.0
        bet_size_score = min(1.0, max(0.0, (amount - 1000) / 49000))

    # --- Composite ---
    composite = (wallet_age_score * 0.4 + low_probability_score * 0.35 + bet_size_score * 0.25)

    # --- Hard trigger ---
    is_hard = (
        wallet_age_score >= 0.7
        and low_probability_score >= 0.8
        and bet_size_score >= 0.7
        and amount >= min_amount_usd
    )

    return NewWalletResult(
        wallet_age_score=wallet_age_score,
        bet_size_score=bet_size_score,
        low_probability_score=low_probability_score,
        composite_score=composite,
        is_hard_trigger=is_hard,
    )
