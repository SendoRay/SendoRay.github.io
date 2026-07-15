"""Pattern 2: Volume spike detection.

Detects markets where recent trading volume deviates significantly from
the historical baseline. A sudden spike can signal informed money entering
the market before a major event.
"""
import math
from dataclasses import dataclass


@dataclass
class VolumeSpikeResult:
    volume_z_score: float     # standard deviations from baseline
    volume_spike_score: float # 0-1, normalized
    recent_volume_usd: float
    baseline_volume_usd: float
    baseline_std: float


def evaluate(recent_volume_usd: float, baseline_volume_usd: float,
             baseline_std: float) -> VolumeSpikeResult:
    if baseline_std is None or baseline_std <= 0:
        z_score = 0.0 if recent_volume_usd <= baseline_volume_usd else 5.0
    else:
        z_score = (recent_volume_usd - baseline_volume_usd) / baseline_std

    # Sigmoid-like normalization: z=2 → ~0.5, z=4 → ~0.8, z=6 → ~0.95
    spike_score = 1.0 / (1.0 + math.exp(-(z_score - 2.5) * 0.8))

    return VolumeSpikeResult(
        volume_z_score=round(z_score, 2),
        volume_spike_score=round(spike_score, 4),
        recent_volume_usd=recent_volume_usd,
        baseline_volume_usd=baseline_volume_usd,
        baseline_std=baseline_std,
    )
