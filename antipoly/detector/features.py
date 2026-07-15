"""Feature extraction: aggregates 5 pattern rule outputs into a feature vector."""
from dataclasses import dataclass, asdict

from detector.rules.new_wallet import NewWalletResult
from detector.rules.volume_spike import VolumeSpikeResult
from detector.rules.coordinated import CoordinatedResult
from detector.rules.whale_drift import WhaleDriftResult
from detector.rules.odds_manipulation import OddsManipulationResult

FEATURE_NAMES = [
    "wallet_age_score",
    "bet_size_score",
    "low_probability_score",
    "volume_spike_score",
    "volume_z_score",
    "coordination_score",
    "fund_correlation_score",
    "behavior_drift_score",
    "new_market_score",
    "price_impact_score",
    "reversal_score",
]


@dataclass
class FeatureVector:
    wallet_age_score: float = 0.0
    bet_size_score: float = 0.0
    low_probability_score: float = 0.0
    volume_spike_score: float = 0.0
    volume_z_score: float = 0.0
    coordination_score: float = 0.0
    fund_correlation_score: float = 0.0
    behavior_drift_score: float = 0.0
    new_market_score: float = 0.0
    price_impact_score: float = 0.0
    reversal_score: float = 0.0

    def to_array(self) -> list[float]:
        return [getattr(self, name) for name in FEATURE_NAMES]

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in FEATURE_NAMES}


def build_feature_vector(
    new_wallet: NewWalletResult | None = None,
    volume_spike: VolumeSpikeResult | None = None,
    coordinated: CoordinatedResult | None = None,
    whale_drift: WhaleDriftResult | None = None,
    odds_manip: OddsManipulationResult | None = None,
) -> FeatureVector:
    fv = FeatureVector()

    if new_wallet:
        fv.wallet_age_score = new_wallet.wallet_age_score
        fv.bet_size_score = new_wallet.bet_size_score
        fv.low_probability_score = new_wallet.low_probability_score

    if volume_spike:
        fv.volume_spike_score = volume_spike.volume_spike_score
        fv.volume_z_score = _normalize_z(volume_spike.volume_z_score)

    if coordinated:
        fv.coordination_score = coordinated.coordination_score
        fv.fund_correlation_score = coordinated.fund_correlation_score

    if whale_drift:
        fv.behavior_drift_score = whale_drift.behavior_drift_score
        fv.new_market_score = whale_drift.new_market_score

    if odds_manip:
        fv.price_impact_score = odds_manip.price_impact_score
        fv.reversal_score = odds_manip.reversal_score

    return fv


def _normalize_z(z: float) -> float:
    """Normalize z-score to 0-1 range: |z|>=6 → 1.0, z=0 → 0.0"""
    return min(1.0, abs(z) / 6.0)
