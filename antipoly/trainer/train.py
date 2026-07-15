"""Trainer: independent script that trains the Isolation Forest model.

Runs periodically (weekly via cron or Docker restart policy).
Pulls historical trade features from the DB, trains, saves, and activates.
"""
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy import select, func

from db.models import Trade, Market, Wallet, ModelVersion
from db.session import db_session
from utils.config import get_config
from detector.features import FeatureVector, FEATURE_NAMES, build_feature_vector
from detector.ml import AnomalyScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def extract_training_features(days: int = 90) -> np.ndarray:
    """Extract feature vectors from historical trades for training."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    with db_session() as session:
        trades = session.execute(
            select(Trade)
            .where(Trade.trade_timestamp >= since)
            .order_by(Trade.trade_timestamp.desc())
            .limit(50000)
        ).scalars().all()

        if not trades:
            logger.error("No trades found for training")
            return np.empty((0, len(FEATURE_NAMES)))

        logger.info(f"Extracting features from {len(trades)} trades...")

        # Batch load market data
        condition_ids = {t.condition_id for t in trades}
        markets = {}
        for cid in condition_ids:
            m = session.get(Market, cid)
            if m:
                markets[cid] = m

        # Batch load wallet data
        wallet_addrs = {t.taker_address for t in trades if t.taker_address}
        wallet_addrs.update({t.maker_address for t in trades if t.maker_address})
        wallets = {}
        for addr in wallet_addrs:
            w = session.get(Wallet, addr)
            if w:
                wallets[addr] = w

        # Compute market avg trade sizes
        market_avg_sizes = {}
        for cid in condition_ids:
            avg = session.execute(
                select(func.avg(Trade.amount_usd))
                .where(Trade.condition_id == cid)
            ).scalar()
            market_avg_sizes[cid] = float(avg) if avg else 0.0

        features = []
        for t in trades:
            market = markets.get(t.condition_id)
            wallet = wallets.get(t.taker_address or "") or wallets.get(t.maker_address or "")

            market_prob = market.yes_price if market else None
            market_avg = market_avg_sizes.get(t.condition_id, 0.0)

            # Simplified feature extraction for training
            # (no on-demand API calls, use DB data only)
            now = t.trade_timestamp

            # Wallet age
            if wallet and wallet.first_seen_at:
                age_days = (now - wallet.first_seen_at).total_seconds() / 86400
                wallet_age = max(0.0, 1.0 - (age_days / 30.0)) if age_days < 30 else 0.0
            else:
                wallet_age = 1.0

            # Low probability
            if market_prob is not None:
                low_prob = 1.0 - min(1.0, market_prob / 0.5) if market_prob < 0.5 else 0.0
            else:
                low_prob = 0.5

            # Bet size
            amount = t.amount_usd or 0
            if market_avg > 0:
                ratio = amount / market_avg
                bet_size = min(1.0, max(0.0, (ratio - 1.0) / 9.0))
            else:
                bet_size = min(1.0, max(0.0, (amount - 1000) / 49000))

            fv = FeatureVector(
                wallet_age_score=wallet_age,
                bet_size_score=bet_size,
                low_probability_score=low_prob,
            )
            features.append(fv.to_array())

        return np.array(features)


def train_and_save():
    """Main training routine."""
    cfg = get_config()
    window_days = cfg.trainer.training_window_days

    logger.info(f"Starting training with {window_days}-day window")
    features = extract_training_features(window_days)

    if features.shape[0] < 100:
        logger.error(f"Not enough training data: {features.shape[0]} samples (need 100+)")
        sys.exit(1)

    scorer = AnomalyScorer(contamination=cfg.trainer.contamination)
    metrics = scorer.train(features)

    # Save model
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"model_{version}.pkl"
    scorer.save(model_path)

    # Deactivate previous models and activate new one
    now = datetime.now(timezone.utc)
    with db_session() as session:
        session.execute(
            ModelVersion.__table__.update().values(is_active=False)
        )

        mv = ModelVersion(
            version=version,
            trained_at=now,
            training_data_start=now - timedelta(days=window_days),
            training_data_end=now,
            training_samples=int(features.shape[0]),
            contamination=cfg.trainer.contamination,
            model_path=str(model_path),
            metrics=metrics,
            is_active=True,
        )
        session.add(mv)

    logger.info(f"Training complete. Model v{version} activated. Path: {model_path}")
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    train_and_save()
