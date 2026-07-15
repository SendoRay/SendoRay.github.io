"""Detection pipeline: L1 SQL filter → L2 rule evaluation + ML scoring."""
import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, func, and_, desc

from db.models import Trade, Market, Wallet, Alert
from db.session import db_session
from utils.config import get_config
from collector.data_api import DataApiClient
from collector.subgraph import SubgraphClient
from detector.features import build_feature_vector
from detector.ml import AnomalyScorer
from detector.rules.new_wallet import evaluate as eval_new_wallet
from detector.rules.volume_spike import evaluate as eval_volume_spike
from detector.rules.coordinated import evaluate as eval_coordinated
from detector.rules.whale_drift import evaluate as eval_whale_drift
from detector.rules.odds_manipulation import evaluate as eval_odds_manip

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Orchestrates the two-tier detection pipeline.

    L1: Fast SQL filter (low probability + large amount)
    L2: Full rule evaluation + ML scoring for L1-flagged trades
    """

    def __init__(self):
        self.cfg = get_config()
        self.data_api = DataApiClient()
        self.subgraph = SubgraphClient()
        self.scorer = AnomalyScorer(contamination=self.cfg.trainer.contamination)
        self._model_loaded = False

    async def start(self):
        await self.data_api.start()
        await self.subgraph.start()
        self._load_latest_model()

    async def stop(self):
        await self.data_api.stop()
        await self.subgraph.stop()

    def _load_latest_model(self):
        from db.models import ModelVersion
        with db_session() as session:
            mv = session.execute(
                select(ModelVersion)
                .where(ModelVersion.is_active == True)
                .order_by(desc(ModelVersion.trained_at))
                .limit(1)
            ).scalar_one_or_none()

            if mv and mv.model_path:
                loaded = self.scorer.load(mv.model_path)
                if loaded:
                    self._model_loaded = True
                    logger.info(f"Loaded active model v{mv.version}")
                    return

        logger.warning("No active model found — running rules-only mode")

    async def run_l1_filter(self, since: datetime) -> list[dict]:
        """L1: SQL-level fast filter for candidate trades.

        Returns trades where amount >= threshold, regardless of probability
        (probability filter applied via market data if available).
        """
        with db_session() as session:
            trades = session.execute(
                select(Trade)
                .where(Trade.trade_timestamp >= since)
                .where(Trade.amount_usd >= self.cfg.detector.l1_min_amount_usd)
                .order_by(Trade.trade_timestamp.desc())
                .limit(500)
            ).scalars().all()

            results = []
            for t in trades:
                market = session.get(Market, t.condition_id)
                market_prob = market.yes_price if market else None

                # Apply probability filter if we have market data
                if market_prob is not None and market_prob > self.cfg.detector.l1_low_probability_threshold:
                    # Also check if probability is very high (>70%, could be "No" bet on low prob)
                    if market_prob < 0.70:
                        continue

                results.append({
                    "trade_id": t.id,
                    "trade_hash": t.trade_id,
                    "condition_id": t.condition_id,
                    "maker_address": t.maker_address,
                    "taker_address": t.taker_address,
                    "side": t.side,
                    "outcome": t.outcome,
                    "price": t.price,
                    "size": t.size,
                    "amount_usd": t.amount_usd,
                    "trade_timestamp": t.trade_timestamp,
                    "market_probability": market_prob,
                    "market_question": market.question if market else None,
                    "market_category": market.category if market else None,
                    "market_liquidity": market.liquidity if market else None,
                })

            logger.info(f"L1 filter: {len(results)} candidate trades from {since}")
            return results

    async def run_l2_detection(self, trade: dict) -> dict | None:
        """L2: Full rule evaluation + ML scoring for a single trade."""
        wallet_addr = trade.get("taker_address") or trade.get("maker_address")
        if not wallet_addr:
            return None

        # --- Gather context ---
        wallet_first_seen, wallet_trade_count, wallet_total_volume, wallet_markets_traded = (
            await self._get_wallet_context(wallet_addr)
        )

        market_avg_trade_usd = await self._get_market_avg_trade(trade["condition_id"])
        recent_volume, baseline_volume, baseline_std = await self._get_volume_baseline(trade["condition_id"])
        recent_same_direction = await self._get_recent_same_direction_trades(trade)
        related_wallets = await self._get_related_wallets(wallet_addr)
        wallet_categories = await self._get_wallet_categories(wallet_addr)
        price_data = await self._get_price_context(trade)

        # --- Run all 5 pattern rules ---
        nw_result = await eval_new_wallet(
            trade, wallet_first_seen, wallet_trade_count,
            trade.get("market_probability"), market_avg_trade_usd,
            max_age_days=self.cfg.detector.rule_new_wallet_max_age_days,
            low_prob_max=self.cfg.detector.rule_low_probability_max,
            min_amount_usd=self.cfg.detector.rule_min_amount_usd,
        )

        vs_result = eval_volume_spike(recent_volume, baseline_volume, baseline_std)

        coord_result = await eval_coordinated(trade, recent_same_direction, related_wallets)

        whale_result = eval_whale_drift(
            trade, wallet_total_volume, wallet_markets_traded,
            wallet_trade_count, trade.get("market_category"), wallet_categories,
        )

        odds_result = eval_odds_manip(
            trade, price_data.get("before"), price_data.get("after"),
            price_data.get("later"), trade.get("amount_usd"),
            trade.get("market_liquidity"),
        )

        # --- Build feature vector ---
        fv = build_feature_vector(nw_result, vs_result, coord_result, whale_result, odds_result)

        # --- ML scoring ---
        ml_score, shap_values = self.scorer.predict(fv) if self.scorer.is_ready else (0.0, {})

        # --- Rule hard-trigger ---
        rule_triggered = nw_result.is_hard_trigger

        # --- Determine severity ---
        if rule_triggered or ml_score >= self.cfg.detector.ml_score_high:
            severity = "high"
        elif ml_score >= self.cfg.detector.ml_score_low:
            severity = "low"
        else:
            return None  # Not anomalous enough

        return {
            "trade": trade,
            "wallet_address": wallet_addr,
            "severity": severity,
            "ml_score": ml_score,
            "rule_hard_trigger": rule_triggered,
            "triggered_rules": {
                "new_wallet": nw_result.composite_score,
                "volume_spike": vs_result.composite_score,
                "coordinated": coord_result.composite_score,
                "whale_drift": whale_result.composite_score,
                "odds_manipulation": odds_result.composite_score,
            },
            "shap_values": shap_values,
            "feature_vector": fv.to_dict(),
            "related_wallets": coord_result.related_wallets,
        }

    async def _get_wallet_context(self, address: str) -> tuple:
        """Returns (first_seen, trade_count, total_volume, markets_traded)."""
        with db_session() as session:
            wallet = session.get(Wallet, address)
            if wallet and wallet.total_trades > 0:
                return (wallet.first_seen_at, wallet.total_trades,
                        wallet.total_volume_usd, wallet.markets_traded)

        first_seen = await self.data_api.get_wallet_first_trade_time(address)
        trade_count = await self.data_api.get_wallet_trade_count(address)
        return (first_seen, trade_count, 0.0, 0)

    async def _get_market_avg_trade(self, condition_id: str) -> float | None:
        with db_session() as session:
            result = session.execute(
                select(func.avg(Trade.amount_usd))
                .where(Trade.condition_id == condition_id)
            ).scalar()
            return float(result) if result else None

    async def _get_volume_baseline(self, condition_id: str) -> tuple[float, float, float]:
        """Returns (recent_volume_1h, baseline_avg_7d, baseline_std_7d)."""
        now = datetime.now(timezone.utc)
        one_hour_ago = now - timedelta(hours=1)
        seven_days_ago = now - timedelta(days=7)

        with db_session() as session:
            recent = session.execute(
                select(func.sum(Trade.amount_usd))
                .where(and_(
                    Trade.condition_id == condition_id,
                    Trade.trade_timestamp >= one_hour_ago,
                ))
            ).scalar() or 0.0

            daily_volumes = session.execute(
                select(
                    func.date_trunc("day", Trade.trade_timestamp).label("day"),
                    func.sum(Trade.amount_usd).label("vol"),
                )
                .where(and_(
                    Trade.condition_id == condition_id,
                    Trade.trade_timestamp >= seven_days_ago,
                ))
                .group_by("day")
            ).all()

            if len(daily_volumes) >= 2:
                vols = [float(r[1] or 0) for r in daily_volumes]
                baseline = sum(vols) / len(vols) / 24  # hourly average
                std = (sum((v - baseline * 24) ** 2 for v in vols) / len(vols)) ** 0.5 / 24
            else:
                baseline = float(recent) if recent else 0.0
                std = baseline * 0.5 if baseline > 0 else 1.0

            return (float(recent), float(baseline), float(std))

    async def _get_recent_same_direction_trades(self, trade: dict) -> list[dict]:
        now = trade.get("trade_timestamp", datetime.now(timezone.utc))
        if isinstance(now, str):
            now = datetime.fromisoformat(now.replace("Z", "+00:00"))
        two_hours_ago = now - timedelta(hours=2)

        with db_session() as session:
            trades = session.execute(
                select(Trade)
                .where(and_(
                    Trade.condition_id == trade["condition_id"],
                    Trade.trade_timestamp >= two_hours_ago,
                    Trade.outcome == trade.get("outcome"),
                    Trade.trade_timestamp < now,
                ))
                .limit(50)
            ).scalars().all()

            return [{
                "taker_address": t.taker_address,
                "maker_address": t.maker_address,
                "outcome": t.outcome,
                "amount_usd": t.amount_usd,
                "trade_timestamp": t.trade_timestamp,
            } for t in trades]

    async def _get_related_wallets(self, address: str) -> list[dict]:
        try:
            return await self.subgraph.get_related_wallets(address)
        except Exception:
            logger.warning(f"Failed to get related wallets for {address}")
            return []

    async def _get_wallet_categories(self, address: str) -> list[str]:
        with db_session() as session:
            result = session.execute(
                select(Market.category)
                .join(Trade, Trade.condition_id == Market.condition_id)
                .where((Trade.taker_address == address) | (Trade.maker_address == address))
                .distinct()
            ).scalars().all()
            return [c for c in result if c]

    async def _get_price_context(self, trade: dict) -> dict:
        """Get price before, after, and later for odds manipulation detection."""
        return {
            "before": trade.get("market_probability"),
            "after": None,
            "later": None,
        }

    def should_alert(self, result: dict) -> bool:
        """Check dedup: no alert for same wallet+market in dedup window."""
        dedup_key = f"{result['wallet_address']}:{result['trade']['condition_id']}"
        window = timedelta(minutes=self.cfg.alerter.dedup_window_minutes)

        with db_session() as session:
            existing = session.execute(
                select(Alert)
                .where(Alert.dedup_key == dedup_key)
                .where(Alert.created_at >= datetime.now(timezone.utc) - window)
                .limit(1)
            ).scalar_one_or_none()

            return existing is None, dedup_key
