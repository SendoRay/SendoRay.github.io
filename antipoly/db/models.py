from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, BigInteger, String, Text, Boolean, Float,
    DateTime, ForeignKey, JSON, Index
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Market(Base):
    __tablename__ = "markets"

    condition_id = Column(Text, primary_key=True)
    question = Column(Text, nullable=False)
    slug = Column(Text)
    category = Column(Text)
    yes_price = Column(Float)
    no_price = Column(Float)
    volume = Column(Float)
    liquidity = Column(Float)
    start_date = Column(DateTime(timezone=True))
    end_date = Column(DateTime(timezone=True))
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    trades = relationship("Trade", back_populates="market")


class Trade(Base):
    __tablename__ = "trades"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    trade_id = Column(Text, unique=True)
    condition_id = Column(Text, ForeignKey("markets.condition_id", ondelete="CASCADE"), nullable=False)
    maker_address = Column(Text)
    taker_address = Column(Text)
    side = Column(Text)
    outcome = Column(Text)
    price = Column(Float)
    size = Column(Float)
    amount_usd = Column(Float)
    trade_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    synced_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    market = relationship("Market", back_populates="trades")


class Wallet(Base):
    __tablename__ = "wallets"

    address = Column(Text, primary_key=True)
    first_seen_at = Column(DateTime(timezone=True))
    last_seen_at = Column(DateTime(timezone=True))
    total_trades = Column(Integer, default=0)
    total_volume_usd = Column(Float, default=0.0)
    markets_traded = Column(Integer, default=0)
    is_flagged = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    condition_id = Column(Text, ForeignKey("markets.condition_id", ondelete="CASCADE"))
    wallet_address = Column(Text)
    severity = Column(Text, nullable=False)
    ml_score = Column(Float)
    triggered_rules = Column(JSONB)
    shap_values = Column(JSONB)
    trade_ids = Column(ARRAY(BigInteger))
    total_amount_usd = Column(Float)
    market_question = Column(Text)
    market_probability = Column(Float)
    message = Column(Text)
    dedup_key = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(Text, nullable=False)
    trained_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    training_data_start = Column(DateTime(timezone=True))
    training_data_end = Column(DateTime(timezone=True))
    training_samples = Column(Integer)
    contamination = Column(Float)
    model_path = Column(Text)
    metrics = Column(JSONB)
    is_active = Column(Boolean, default=False)


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    condition_id = Column(Text, nullable=False)
    yes_price = Column(Float)
    no_price = Column(Float)
    volume_24h = Column(Float)
    liquidity = Column(Float)
    snapshot_time = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
