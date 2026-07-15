import asyncio
import logging

import httpx

from utils.config import get_config

logger = logging.getLogger(__name__)


class SubgraphClient:
    """Queries the Polymarket subgraph (Goldsky) for on-chain fund flow data.

    Used by the detector for pattern 3 (coordinated betting) to trace
    fund transfers between wallets and identify shared funding sources.
    """

    def __init__(self):
        self.cfg = get_config()
        self.endpoint = self.cfg.api.goldsky_subgraph
        self.client: httpx.AsyncClient | None = None

    async def start(self):
        self.client = httpx.AsyncClient(timeout=self.cfg.collector.request_timeout_seconds)

    async def stop(self):
        if self.client:
            await self.client.aclose()

    async def get_fund_transfers(self, address: str, limit: int = 100) -> list[dict]:
        """Query the subgraph for transfers involving a wallet.

        Looks for USDC transfers and deposits to the Polymarket proxy contract
        that funded this wallet.
        """
        query = """
        query WalletTransfers($address: String!, $limit: Int!) {
            transfers(
                where: { or: [{ from: $address }, { to: $address }] }
                first: $limit
                orderBy: timestamp
                orderDirection: desc
            ) {
                id
                from
                to
                amount
                timestamp
                transactionHash
            }
        }
        """
        data = await self._query(query, {"address": address.lower(), "limit": limit})
        return data.get("data", {}).get("transfers", [])

    async def get_related_wallets(self, address: str, depth: int = 1) -> list[dict]:
        """Find wallets that share a funding source with the given address.

        Traces fund transfers back to common source addresses.
        Returns list of {address, shared_source, transfer_time} dicts.
        """
        transfers = await self.get_fund_transfers(address, limit=200)
        incoming = [t for t in transfers if t.get("to", "").lower() == address.lower()]

        if not incoming:
            return []

        source_addresses = {t["from"].lower() for t in incoming if t.get("from")}
        related = []

        for source in source_addresses:
            source_transfers = await self.get_fund_transfers(source, limit=200)
            outgoing = [
                t for t in source_transfers
                if t.get("from", "").lower() == source
                and t.get("to", "").lower() != address.lower()
            ]
            for t in outgoing:
                related.append({
                    "address": t.get("to"),
                    "shared_source": source,
                    "transfer_time": t.get("timestamp"),
                    "amount": t.get("amount"),
                })

        return related

    async def get_positions(self, condition_id: str, limit: int = 100) -> list[dict]:
        """Query positions for a specific market from the subgraph."""
        query = """
        query MarketPositions($conditionId: String!, $limit: Int!) {
            positions(
                where: { conditionId: $conditionId }
                first: $limit
                orderBy: size
                orderDirection: desc
            ) {
                id
                account
                size
                outcomeIndex
                conditionId
            }
        }
        """
        data = await self._query(query, {
            "conditionId": condition_id.lower(),
            "limit": limit,
        })
        return data.get("data", {}).get("positions", [])

    async def _query(self, query: str, variables: dict) -> dict:
        if not self.client:
            await self.start()
        for attempt in range(self.cfg.collector.max_retries):
            try:
                resp = await self.client.post(
                    self.endpoint,
                    json={"query": query, "variables": variables},
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.warning(f"Subgraph query attempt {attempt+1} failed: {e}")
                await asyncio.sleep(self.cfg.collector.retry_backoff_seconds)
        raise RuntimeError(f"Subgraph query failed after retries")
