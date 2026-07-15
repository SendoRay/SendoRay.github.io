"""antipoly — Polymarket Anomaly Detection System

Main entry point: runs the asyncio event loop with:
- Gamma collector (market metadata, every 5 min)
- CLOB collector (trade data, every 1 min)
- Detection pipeline (process new trades, run L1→L2)
- Telegram alerter (push alerts)
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta

from collector.gamma_api import GammaCollector
from collector.clob_api import ClobCollector
from detector.pipeline import DetectionPipeline
from alerter.telegram import TelegramAlerter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("antipoly")


async def gamma_loop(stop_event: asyncio.Event):
    """Periodic market metadata collector."""
    collector = GammaCollector()
    await collector.run_loop(stop_event)


async def trade_collection_loop(stop_event: asyncio.Event):
    """Periodic trade data collector."""
    collector = ClobCollector()
    await collector.run_loop(stop_event)


async def detection_loop(stop_event: asyncio.Event):
    """Detection pipeline: poll for new trades and evaluate them."""
    from db.session import db_session
    from db.models import Trade
    from sqlalchemy import select, func

    pipeline = DetectionPipeline()
    await pipeline.start()
    alerter = TelegramAlerter()
    await alerter.start()

    poll_interval = 30  # Check for new candidate trades every 30s
    last_check = datetime.now(timezone.utc) - timedelta(minutes=5)

    try:
        while not stop_event.is_set():
            try:
                # L1 filter: get candidate trades since last check
                candidates = await pipeline.run_l1_filter(last_check)
                last_check = datetime.now(timezone.utc)

                if candidates:
                    logger.info(f"Processing {len(candidates)} candidate trades")

                for trade in candidates:
                    result = await pipeline.run_l2_detection(trade)
                    if result is None:
                        continue

                    # Dedup check
                    should_alert, dedup_key = pipeline.should_alert(result)
                    if not should_alert:
                        logger.debug(f"Dedup skip: {dedup_key}")
                        continue

                    # Send alert
                    await alerter.send_alert(result)
                    logger.info(
                        f"Alert [{result['severity']}] wallet={result['wallet_address']} "
                        f"score={result['ml_score']:.3f}"
                    )

            except Exception:
                logger.exception("Detection loop error")

            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
    finally:
        await pipeline.stop()
        await alerter.stop()


async def main():
    logger.info("antipoly starting up...")

    stop_event = asyncio.Event()

    tasks = [
        asyncio.create_task(gamma_loop(stop_event)),
        asyncio.create_task(trade_collection_loop(stop_event)),
        asyncio.create_task(detection_loop(stop_event)),
    ]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
        stop_event.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("antipoly stopped")


if __name__ == "__main__":
    asyncio.run(main())
