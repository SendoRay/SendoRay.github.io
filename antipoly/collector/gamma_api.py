import asyncio
import logging
from datetime import datetime, timezone

import httpx
from sqlalchemy import select, update

from db.models import Market
from db.session import db_session
from utils.config import get_config

logger = logging.getLogger(__name__)


class GammaCollector:
    """Collects market metadata from the Gamma API."""

    def __init__(self):
        self.cfg = get_config()
        self.base_url = self.cfg.api.gamma_base
        self.client: httpx.AsyncClient | None = None

    async def start(self):
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.cfg.collector.request_timeout_seconds,
        )

    async def stop(self):
        if self.client:
            await self.client.aclose()

    async def fetch_active_markets(self) -> list[dict]:
        """Fetch all active markets from Gamma API."""
        markets = []
        offset = 0
        limit = 100
        while True:
            resp = await self._get("/markets", params={
                "active": "true",
                "limit": limit,
                "offset": offset,
                "order": "volume24hr",
                "ascending": "false",
            })
            batch = resp.get("data", resp) if isinstance(resp, dict) else resp
            if not batch:
                break
            markets.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
            await asyncio.sleep(0.2)
        logger.info(f"Fetched {len(markets)} active markets from Gamma API")
        return markets

    async def update_markets_db(self, markets: list[dict]):
        """Upsert market metadata into the database."""
        with db_session() as session:
            for m in markets:
                condition_id = m.get("conditionId") or m.get("condition_id")
                if not condition_id:
                    continue
                existing = session.get(Market, condition_id)
                values = {
                    "question": m.get("question", ""),
                    "slug": m.get("slug"),
                    "category": m.get("category"),
                    "yes_price": _safe_float(m.get("outcomePrices", ["0", "0"])[0]) if m.get("outcomePrices") else _safe_float(m.get("yesPrice")),
                    "no_price": _safe_float(m.get("outcomePrices", ["0", "0"])[1]) if m.get("outcomePrices") else _safe_float(m.get("noPrice")),
                    "volume": _safe_float(m.get("volume")),
                    "liquidity": _safe_float(m.get("liquidity")),
                    "start_date": m.get("startDate"),
                    "end_date": m.get("endDate"),
                    "active": m.get("active", True),
                    "updated_at": datetime.now(timezone.utc),
                }
                if existing:
                    session.execute(
                        update(Market)
                        .where(Market.condition_id == condition_id)
                        .values(**values)
                    )
                else:
                    session.add(Market(condition_id=condition_id, **values))
        logger.info(f"Upserted {len(markets)} markets into DB")

    async def run_loop(self, stop_event: asyncio.Event):
        """Periodic loop: fetch and update market metadata."""
        await self.start()
        try:
            while not stop_event.is_set():
                try:
                    markets = await self.fetch_active_markets()
                    await self.update_markets_db(markets)
                except Exception:
                    logger.exception("Gamma collector error")
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self.cfg.collector.gamma_poll_interval_seconds,
                )
        finally:
            await self.stop()

    async def _get(self, path: str, params: dict | None = None) -> dict | list:
        for attempt in range(self.cfg.collector.max_retries):
            try:
                resp = await self.client.get(path, params=params)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.warning(f"Gamma API {path} attempt {attempt+1} failed: {e}")
                await asyncio.sleep(self.cfg.collector.retry_backoff_seconds)
        raise RuntimeError(f"Gamma API {path} failed after {self.cfg.collector.max_retries} retries")


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
