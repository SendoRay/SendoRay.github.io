"""Streamlit dashboard: alert list, wallet profile, market overview."""
import logging

import pandas as pd
import streamlit as st
from sqlalchemy import select, func, desc, and_

from db.models import Alert, Wallet, Market, Trade
from db.session import db_session, get_engine
from utils.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="antipoly — Polymarket Anomaly Monitor",
    page_icon="🔍",
    layout="wide",
)

cfg = get_config()


@st.cache_data(ttl=30)
def get_alerts(severity: str | None = None, limit: int = 100) -> pd.DataFrame:
    with db_session() as session:
        query = select(Alert).order_by(desc(Alert.created_at)).limit(limit)
        if severity and severity != "all":
            query = query.where(Alert.severity == severity)
        alerts = session.execute(query).scalars().all()
        if not alerts:
            return pd.DataFrame()
        rows = []
        for a in alerts:
            rows.append({
                "id": a.id,
                "time": a.created_at,
                "severity": a.severity,
                "wallet": a.wallet_address,
                "market": a.market_question or a.condition_id,
                "amount_usd": a.total_amount_usd,
                "ml_score": a.ml_score,
                "probability": a.market_probability,
                "triggered_rules": a.triggered_rules,
                "shap_values": a.shap_values,
            })
        return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def get_wallet_profile(address: str) -> dict | None:
    with db_session() as session:
        wallet = session.get(Wallet, address)
        if not wallet:
            return None

        alerts = session.execute(
            select(Alert)
            .where(Alert.wallet_address == address)
            .order_by(desc(Alert.created_at))
            .limit(20)
        ).scalars().all()

        recent_trades = session.execute(
            select(Trade)
            .where((Trade.taker_address == address) | (Trade.maker_address == address))
            .order_by(desc(Trade.trade_timestamp))
            .limit(50)
        ).scalars().all()

        return {
            "wallet": wallet,
            "alerts": alerts,
            "trades": recent_trades,
        }


@st.cache_data(ttl=30)
def get_markets_overview(limit: int = 100) -> pd.DataFrame:
    with db_session() as session:
        markets = session.execute(
            select(Market)
            .where(Market.active == True)
            .order_by(desc(Market.volume))
            .limit(limit)
        ).scalars().all()

        rows = []
        for m in markets:
            alert_count = session.execute(
                select(func.count())
                .select_from(Alert)
                .where(Alert.condition_id == m.condition_id)
            ).scalar() or 0

            rows.append({
                "condition_id": m.condition_id,
                "question": m.question,
                "category": m.category or "—",
                "yes_price": m.yes_price,
                "volume": m.volume,
                "liquidity": m.liquidity,
                "alerts": alert_count,
                "updated": m.updated_at,
            })
        return pd.DataFrame(rows)


# ============ Sidebar Navigation ============
st.sidebar.title("antipoly")
st.sidebar.caption("Polymarket Anomaly Monitor")
page = st.sidebar.radio("Navigate", ["Alerts", "Wallet Profile", "Market Overview"])
st.sidebar.divider()
st.sidebar.markdown(f"**Config**")
st.sidebar.markdown(f"Poll interval: {cfg.collector.poll_interval_seconds}s")
st.sidebar.markdown(f"L1 threshold: ${cfg.detector.l1_min_amount_usd:,.0f}")
st.sidebar.markdown(f"ML high: {cfg.detector.ml_score_high}")
st.sidebar.markdown(f"ML low: {cfg.detector.ml_score_low}")

# ============ Page: Alerts ============
if page == "Alerts":
    st.title("Anomaly Alerts")

    col1, col2 = st.columns([1, 3])
    with col1:
        severity_filter = st.selectbox("Severity", ["all", "high", "low"])
    with col2:
        st.write("")

    df = get_alerts(severity=severity_filter, limit=cfg.dashboard.page_size)

    if df.empty:
        st.info("No alerts yet. The system is monitoring...")
    else:
        for _, row in df.iterrows():
            severity_color = "🔴" if row["severity"] == "high" else "🟡"
            with st.expander(
                f"{severity_color} [{row['severity'].upper()}] "
                f"{row['market'][:60]} — ${row['amount_usd']:,.2f} "
                f"(score: {row['ml_score']:.3f}) — {row['time']}"
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Wallet:** `{row['wallet']}`")
                    st.markdown(f"**Amount:** ${row['amount_usd']:,.2f}")
                    st.markdown(f"**ML Score:** {row['ml_score']:.4f}")
                    if row['probability'] is not None:
                        st.markdown(f"**Market Prob:** {row['probability']:.1%}")

                with col_b:
                    st.markdown("**Triggered Rules:**")
                    rules = row['triggered_rules'] or {}
                    for rule, score in sorted(rules.items(), key=lambda x: x[1], reverse=True):
                        bar = "█" * int(score * 10)
                        st.markdown(f"`{rule}` {bar} {score:.3f}")

                shap = row.get('shap_values', {})
                if shap:
                    st.markdown("**SHAP Feature Contributions:**")
                    top_shap = sorted(shap.items(), key=lambda x: x[1], reverse=True)[:5]
                    for name, val in top_shap:
                        st.markdown(f"- {name}: {val:.3f}")

                st.markdown(
                    f"[View on Polymarket](https://polymarket.com/@{row['wallet']}?tab=activity)"
                )

# ============ Page: Wallet Profile ============
elif page == "Wallet Profile":
    st.title("Wallet Profile")

    address = st.text_input("Enter wallet address:", placeholder="0x...")

    if address:
        profile = get_wallet_profile(address.strip())

        if not profile:
            st.warning(f"No data found for wallet `{address}`")
        else:
            w = profile["wallet"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("First Seen", w.first_seen_at.strftime("%Y-%m-%d %H:%M") if w.first_seen_at else "Unknown")
            col2.metric("Total Trades", w.total_trades)
            col3.metric("Total Volume", f"${w.total_volume_usd:,.2f}")
            col4.metric("Markets Traded", w.markets_traded)

            st.divider()

            st.subheader("Recent Alerts")
            if profile["alerts"]:
                for a in profile["alerts"]:
                    icon = "🔴" if a.severity == "high" else "🟡"
                    st.markdown(
                        f"{icon} **{a.severity.upper()}** — "
                        f"{a.market_question or a.condition_id} — "
                        f"Score: {a.ml_score:.3f} — "
                        f"${a.total_amount_usd:,.2f} — "
                        f"{a.created_at.strftime('%Y-%m-%d %H:%M')}"
                    )
            else:
                st.info("No alerts for this wallet")

            st.divider()

            st.subheader("Recent Trades")
            if profile["trades"]:
                trade_data = [{
                    "time": t.trade_timestamp,
                    "market": t.condition_id[:16] + "...",
                    "side": t.side,
                    "outcome": t.outcome,
                    "price": t.price,
                    "size": t.size,
                    "amount_usd": t.amount_usd,
                } for t in profile["trades"]]
                st.dataframe(pd.DataFrame(trade_data), use_container_width=True)
            else:
                st.info("No trades found")

# ============ Page: Market Overview ============
elif page == "Market Overview":
    st.title("Market Overview")

    df = get_markets_overview(limit=100)

    if df.empty:
        st.info("No markets loaded yet. Waiting for Gamma collector...")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Active Markets", len(df))
        col2.metric("Markets with Alerts", int((df["alerts"] > 0).sum()))
        col3.metric("Total Volume", f"${df['volume'].sum():,.0f}")

        st.divider()

        st.subheader("Monitored Markets")
        display_df = df.copy()
        display_df["yes_price"] = display_df["yes_price"].apply(
            lambda x: f"{x:.1%}" if x is not None else "—"
        )
        display_df["volume"] = display_df["volume"].apply(
            lambda x: f"${x:,.0f}" if x is not None else "—"
        )
        display_df = display_df[["question", "category", "yes_price", "volume", "alerts", "updated"]]
        display_df.columns = ["Market", "Category", "Yes Prob", "Volume", "Alerts", "Updated"]
        st.dataframe(display_df, use_container_width=True, height=600)
