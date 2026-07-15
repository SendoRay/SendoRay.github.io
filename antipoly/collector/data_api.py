import asyncio
import logging
from datetime import datetime, timezone

import httpx
from sqlalchemy import select, func

from db.models import Trade, Wallet
from db.session import db_session
from utils.config import get_config

logger = logging.getLogger(__name__)


class DataApiClient:
    """On-demand client for the Polymarket Data API.

    Used by the detector to query wallet trade history and positions
    when a suspicious trade is flagged for deeper analysis.
    """

    def __init__(self):
        self.cfg = get_config()
        self.base_url = self.cfg.api.data_base
        self.client: httpx.AsyncClient | None = None

    async def start(self):
        self.client = httpx.AsyncClient(timeout=self.cfg.collector.request_timeout_seconds)

    async def stop(self):
        if self.client:
            await self.client.aclose()

    async def get_wallet_trades(self, address: str, limit: int = 500) -> list[dict]:
        """Fetch all trades for a specific wallet from the Data API."""
        trades = []
        offset = 0
        while True:
            params = {"user": address, "limit": limit, "offset": offset}
            data = await self._get("/trades", params=params)
            batch = data if isinstance(data, list) else data.get("data", [])
            if not batch:
                break
            trades.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
            await asyncio.sleep(0.1)
        return trades

    async def get_wallet_positions(self, address: str) -> list[dict]:
        """Fetch current positions for a wallet."""
        data = await self._get("/positions", params={"user": address})
        return data if isinstance(data, list) else data.get("data", [])

    async def get_wallet_first_trade_time(self, address: str) -> datetime | None:
        """Determine when a wallet first traded on Polymarket.

        Tries the Data API first, falls back to local DB.
        """
        trades = await self.get_wallet_trades(address, limit=500)
        if trades:
            earliest = None
            for t in trades:
                ts = t.get("timestamp") or t.get("createdAt")
                dt = _parse_ts(ts)
                if dt and (earliest is None or dt < earliest):
                    earliest = dt
            if earliest:
                return earliest

        with db_session() as session:
            result = session.execute(
                select(func.min(Trade.trade_timestamp))
                .where((Trade.taker_address == address) | (Trade.maker_address == address))
            ).scalar()
            return result

    async def get_wallet_trade_count(self, address: str) -> int:
        """Count total trades for a wallet (API + DB)."""
        trades = await self.get_wallet_trades(address, limit=1)
        if trades and isinstance(trades, list) and len(trades) > 0:
            full_trades = await self.get_wallet_trades(address, limit=500)
            return len(full_trades)
        with db_session() as session:
            count = session.execute(
                select(func.count())
                .select_from(Trade)
                .where((Trade.taker_address == address) | (Trade.maker_address == address))
            ).scalar()
            return count or 0

    async def _get(self, path: str, params: dict | None = None) -> dict | list:
        if not self.client:
            await self.start()
        for attempt in range(self.cfg.collector.max_retries):
            try:
                resp = await self.client.get(f"{self.base_url}{path}", params=params)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.warning(f"Data API {path} attempt {attempt+1} failed: {e}")
                await asyncio.sleep(self.cfg.collector.retry_backoff_seconds)
        raise RuntimeError(f"Data API {path} failed after retries")


def _parse_ts(ts) -> datetime | None:
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        if isinstance(ts, str):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        pass
    return None
