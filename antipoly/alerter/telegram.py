"""Telegram Bot alerter: pushes anomaly alerts to a Telegram chat."""
import logging
from datetime import datetime, timezone

import httpx

from db.models import Alert
from db.session import db_session
from utils.config import get_config

logger = logging.getLogger(__name__)


class TelegramAlerter:
    """Sends formatted anomaly alerts via Telegram Bot API."""

    def __init__(self):
        self.cfg = get_config()
        self.bot_token = self.cfg.telegram_bot_token
        self.chat_id = self.cfg.telegram_chat_id
        self.enabled = self.cfg.alerter.telegram_enabled and bool(self.bot_token)
        self.client: httpx.AsyncClient | None = None

    async def start(self):
        if self.enabled:
            self.client = httpx.AsyncClient(timeout=30)
            logger.info("Telegram alerter started")
        else:
            logger.warning("Telegram alerter disabled (no token/chat_id)")

    async def stop(self):
        if self.client:
            await self.client.aclose()

    async def send_alert(self, result: dict) -> int | None:
        """Format and send a Telegram alert. Returns alert DB ID."""
        if not self.enabled or not self.client:
            logger.info("Alert skipped (Telegram disabled), logging to DB only")
            return self._store_alert(result)

        message = self._format_message(result)

        try:
            resp = await self.client.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
            )
            resp.raise_for_status()
            logger.info(f"Telegram alert sent for {result['wallet_address']}")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

        return self._store_alert(result, message)

    def _format_message(self, result: dict) -> str:
        trade = result["trade"]
        severity = result["severity"]
        score = result["ml_score"]
        rules = result["triggered_rules"]
        shap = result["shap_values"]

        emoji = "🔴" if severity == "high" else "🟡"
        market_q = trade.get("market_question", "Unknown market")
        wallet = result["wallet_address"]
        amount = trade.get("amount_usd", 0)
        prob = trade.get("market_probability")

        # Top SHAP contributors
        top_features = sorted(shap.items(), key=lambda x: x[1], reverse=True)[:3]
        shap_str = "\n".join(
            f"  • {name}: {val:.2f}" for name, val in top_features
        ) if shap else "  (model not loaded)"

        # Triggered rules
        active_rules = [name for name, score in rules.items() if score > 0.5]
        rules_str = ", ".join(active_rules) if active_rules else "none above threshold"

        lines = [
            f"{emoji} *Polymarket Anomaly Alert*",
            f"",
            f"*Market:* {market_q}",
            f"*Wallet:* `{wallet}`",
            f"*Amount:* ${amount:,.2f}",
            f"*Market Prob:* {prob:.1%}" if prob else "*Market Prob:* N/A",
            f"*Severity:* {severity.upper()}",
            f"*ML Score:* {score:.3f}",
            f"*Rule Trigger:* {'YES' if result.get('rule_hard_trigger') else 'No'}",
            f"",
            f"*Active Rules:* {rules_str}",
            f"",
            f"*Top SHAP Factors:*",
            shap_str,
            f"",
            f"[View Profile](https://polymarket.com/@{wallet}?tab=activity)",
        ]
        return "\n".join(lines)

    def _store_alert(self, result: dict, message: str = "") -> int | None:
        """Store alert in the database."""
        trade = result["trade"]
        dedup_key = f"{result['wallet_address']}:{trade['condition_id']}"

        with db_session() as session:
            alert = Alert(
                condition_id=trade["condition_id"],
                wallet_address=result["wallet_address"],
                severity=result["severity"],
                ml_score=result["ml_score"],
                triggered_rules=result["triggered_rules"],
                shap_values=result.get("shap_values", {}),
                trade_ids=[result["trade"]["trade_id"]] if result["trade"].get("trade_id") else None,
                total_amount_usd=trade.get("amount_usd"),
                market_question=trade.get("market_question"),
                market_probability=trade.get("market_probability"),
                message=message,
                dedup_key=dedup_key,
            )
            session.add(alert)
            session.flush()
            return alert.id
