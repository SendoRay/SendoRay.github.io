import asyncio
import logging
from datetime import datetime, timezone

import httpx
from sqlalchemy import select, func

from db.models import Trade, Market
from db.session import db_session
from utils.config import get_config

logger = logging.getLogger(__name__)


class ClobCollector:
    """Collects trade data from the Polymarket Data API (trades endpoint).

    Uses the Data API rather than the CLOB REST API for trade history,
    since Data API provides historical trades with filtering by market.
    The CLOB API is used for price/order-book snapshots.
    """

    def __init__(self):
        self.cfg = get_config()
        self.data_base = self.cfg.api.data_base
        self.clob_base = self.cfg.api.clob_base
        self.client: httpx.AsyncClient | None = None
        self._last_sync: dict[str, datetime] = {}

    async def start(self):
        self.client = httpx.AsyncClient(timeout=self.cfg.collector.request_timeout_seconds)

    async def stop(self):
        if self.client:
            await self.client.aclose()

    async def fetch_recent_trades(self, condition_id: str, since: datetime | None = None) -> list[dict]:
        """Fetch trades for a specific market since the given timestamp."""
        trades = []
        offset = 0
        limit = self.cfg.collector.batch_limit
        while True:
            params = {
                "market": condition_id,
                "limit": limit,
                "offset": offset,
            }
            if since:
                params["after"] = int(since.timestamp())

            data = await self._get(f"{self.data_base}/trades", params=params)
            batch = data if isinstance(data, list) else data.get("data", data.get("trades", []))
            if not batch:
                break
            trades.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
            await asyncio.sleep(0.1)
        return trades

    async def fetch_all_recent_trades(self, since: datetime | None = None) -> list[dict]:
        """Try to fetch recent trades across all markets (no market filter)."""
        params = {"limit": self.cfg.collector.batch_limit}
        if since:
            params["after"] = int(since.timestamp())
        data = await self._get(f"{self.data_base}/trades", params=params)
        trades = data if isinstance(data, list) else data.get("data", data.get("trades", []))
        logger.info(f"Fetched {len(trades)} trades across all markets")
        return trades

    async def store_trades(self, trades: list[dict]) -> int:
        """Store trades in DB, dedup by trade_id. Returns count of new trades."""
        if not trades:
            return 0
        new_count = 0
        with db_session() as session:
            for t in trades:
                trade_id = str(t.get("id") or t.get("transactionHash") or t.get("hash") or "")
                if not trade_id:
                    continue
                existing = session.execute(
                    select(Trade.id).where(Trade.trade_id == trade_id).limit(1)
                ).scalar_one_or_none()
                if existing:
                    continue

                condition_id = t.get("market") or t.get("conditionId") or t.get("condition_id")
                if not condition_id:
                    continue

                price = _safe_float(t.get("price"))
                size = _safe_float(t.get("size") or t.get("shares") or t.get("amount"))
                amount_usd = _safe_float(t.get("usdcSize") or t.get("usdc") or t.get("amountUsd"))
                if amount_usd is None and price is not None and size is not None:
                    amount_usd = price * size

                ts = t.get("timestamp") or t.get("createdAt") or t.get("blockTimestamp")
                if isinstance(ts, (int, float)):
                    trade_ts = datetime.fromtimestamp(ts, tz=timezone.utc)
                elif isinstance(ts, str):
                    trade_ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                else:
                    trade_ts = datetime.now(timezone.utc)

                trade = Trade(
                    trade_id=trade_id,
                    condition_id=condition_id,
                    maker_address=t.get("maker") or t.get("makerAddress"),
                    taker_address=t.get("taker") or t.get("takerAddress"),
                    side=t.get("side"),
                    outcome=t.get("outcome") or _infer_outcome(t),
                    price=price,
                    size=size,
                    amount_usd=amount_usd,
                    trade_timestamp=trade_ts,
                )
                session.add(trade)
                new_count += 1

                await self._update_wallet(session, trade)

        if new_count:
            logger.info(f"Stored {new_count} new trades")
        return new_count

    async def _update_wallet(self, session, trade: Trade):
        """Update or create wallet profile for the taker."""
        from db.models import Wallet
        addr = trade.taker_address or trade.maker_address
        if not addr:
            return
        wallet = session.get(Wallet, addr)
        if wallet is None:
            session.add(Wallet(
                address=addr,
                first_seen_at=trade.trade_timestamp,
                last_seen_at=trade.trade_timestamp,
                total_trades=1,
                total_volume_usd=trade.amount_usd or 0,
                markets_traded=1,
            ))
        else:
            wallet.total_trades += 1
            wallet.total_volume_usd += trade.amount_usd or 0
            wallet.last_seen_at = trade.trade_timestamp
            if wallet.first_seen_at is None:
                wallet.first_seen_at = trade.trade_timestamp

    async def fetch_price(self, condition_id: str) -> dict | None:
        """Fetch current price/midpoint from CLOB API."""
        try:
            resp = await self.client.get(
                f"{self.clob_base}/midpoint",
                params={"token_id": condition_id},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch price for {condition_id}: {e}")
            return None

    async def run_loop(self, stop_event: asyncio.Event):
        """Main polling loop: fetch recent trades every poll interval."""
        await self.start()
        try:
            while not stop_event.is_set():
                try:
                    since = max(self._last_sync.values(), default=None)
                    if since is None:
                        from datetime import timedelta
                        since = datetime.now(timezone.utc) - timedelta(hours=1)

                    trades = await self.fetch_all_recent_trades(since=since)
                    if trades:
                        await self.store_trades(trades)
                        for t in trades:
                            cid = t.get("market") or t.get("conditionId")
                            ts = t.get("timestamp")
                            if cid and ts:
                                if isinstance(ts, (int, float)):
                                    self._last_sync[cid] = datetime.fromtimestamp(ts, tz=timezone.utc)
                                elif isinstance(ts, str):
                                    self._last_sync[cid] = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    logger.exception("CLOB collector error")

                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self.cfg.collector.poll_interval_seconds,
                )
        finally:
            await self.stop()

    async def _get(self, url: str, params: dict | None = None) -> dict | list:
        for attempt in range(self.cfg.collector.max_retries):
            try:
                resp = await self.client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.warning(f"API {url} attempt {attempt+1} failed: {e}")
                await asyncio.sleep(self.cfg.collector.retry_backoff_seconds)
        raise RuntimeError(f"API {url} failed after retries")


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _infer_outcome(trade: dict) -> str | None:
    asset = trade.get("asset") or trade.get("tokenId")
    if asset and "yes" in str(asset).lower():
        return "yes"
    if asset and "no" in str(asset).lower():
        return "no"
    return None
