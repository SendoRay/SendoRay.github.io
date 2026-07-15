-- antipoly database schema
-- Requires TimescaleDB extension

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============ markets ============
CREATE TABLE markets (
    condition_id      TEXT PRIMARY KEY,
    question          TEXT NOT NULL,
    slug              TEXT,
    category          TEXT,
    yes_price         DOUBLE PRECISION,
    no_price          DOUBLE PRECISION,
    volume            DOUBLE PRECISION,
    liquidity         DOUBLE PRECISION,
    start_date        TIMESTAMPTZ,
    end_date          TIMESTAMPTZ,
    active            BOOLEAN DEFAULT TRUE,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_markets_active ON markets(active) WHERE active = TRUE;
CREATE INDEX idx_markets_category ON markets(category);

-- ============ trades (hypertable) ============
CREATE TABLE trades (
    id                BIGSERIAL PRIMARY KEY,
    trade_id          TEXT UNIQUE,          -- Polymarket's trade ID
    condition_id      TEXT NOT NULL REFERENCES markets(condition_id) ON DELETE CASCADE,
    maker_address     TEXT,
    taker_address     TEXT,
    side              TEXT,                 -- 'buy' or 'sell'
    outcome           TEXT,                 -- 'yes' or 'no'
    price             DOUBLE PRECISION,
    size              DOUBLE PRECISION,     -- shares
    amount_usd        DOUBLE PRECISION,     -- price * size
    trade_timestamp   TIMESTAMPTZ NOT NULL,  -- when the trade happened
    synced_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_trades_condition_ts ON trades(condition_id, trade_timestamp DESC);
CREATE INDEX idx_trades_taker_ts ON trades(taker_address, trade_timestamp DESC);
CREATE INDEX idx_trades_maker_ts ON trades(maker_address, trade_timestamp DESC);
CREATE INDEX idx_trades_ts ON trades(trade_timestamp DESC);

SELECT create_hypertable('trades', 'trade_timestamp', if_not_exists => TRUE);

-- Continuous aggregate: hourly volume per market (for pattern 2 baseline)
CREATE MATERIALIZED VIEW trades_hourly_volume
WITH (timescaledb.continuous) AS
SELECT
    condition_id,
    time_bucket('1 hour', trade_timestamp) AS bucket,
    COUNT(*) AS trade_count,
    SUM(amount_usd) AS total_volume_usd,
    COUNT(DISTINCT taker_address) AS unique_takers
FROM trades
GROUP BY condition_id, bucket;

SELECT add_continuous_aggregate_policy('trades_hourly_volume',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- ============ wallets ============
CREATE TABLE wallets (
    address           TEXT PRIMARY KEY,
    first_seen_at     TIMESTAMPTZ,
    last_seen_at      TIMESTAMPTZ,
    total_trades      INTEGER DEFAULT 0,
    total_volume_usd  DOUBLE PRECISION DEFAULT 0,
    markets_traded    INTEGER DEFAULT 0,
    is_flagged        BOOLEAN DEFAULT FALSE,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_wallets_first_seen ON wallets(first_seen_at);
CREATE INDEX idx_wallets_flagged ON wallets(is_flagged) WHERE is_flagged = TRUE;

-- ============ alerts ============
CREATE TABLE alerts (
    id                BIGSERIAL PRIMARY KEY,
    condition_id      TEXT REFERENCES markets(condition_id) ON DELETE CASCADE,
    wallet_address    TEXT,
    severity          TEXT NOT NULL,        -- 'high' or 'low'
    ml_score          DOUBLE PRECISION,
    triggered_rules   JSONB,                -- {"new_wallet": 0.35, "volume_spike": 0.0, ...}
    shap_values       JSONB,                -- {"wallet_age_score": 0.35, ...}
    trade_ids         BIGINT[],
    total_amount_usd  DOUBLE PRECISION,
    market_question   TEXT,
    market_probability DOUBLE PRECISION,
    message           TEXT,
    dedup_key         TEXT,                  -- wallet_address + condition_id for dedup
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_alerts_condition ON alerts(condition_id, created_at DESC);
CREATE INDEX idx_alerts_wallet ON alerts(wallet_address, created_at DESC);
CREATE INDEX idx_alerts_severity ON alerts(severity, created_at DESC) WHERE severity = 'high';
CREATE INDEX idx_alerts_dedup ON alerts(dedup_key, created_at DESC);

-- ============ model_versions ============
CREATE TABLE model_versions (
    id                    SERIAL PRIMARY KEY,
    version               TEXT NOT NULL,
    trained_at            TIMESTAMPTZ DEFAULT NOW(),
    training_data_start   TIMESTAMPTZ,
    training_data_end     TIMESTAMPTZ,
    training_samples      INTEGER,
    contamination         DOUBLE PRECISION,
    model_path            TEXT,
    metrics               JSONB,
    is_active             BOOLEAN DEFAULT FALSE
);

-- ============ market_snapshots (hypertable) ============
CREATE TABLE market_snapshots (
    id                BIGSERIAL PRIMARY KEY,
    condition_id      TEXT NOT NULL,
    yes_price         DOUBLE PRECISION,
    no_price          DOUBLE PRECISION,
    volume_24h        DOUBLE PRECISION,
    liquidity         DOUBLE PRECISION,
    snapshot_time     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_snapshots_condition ON market_snapshots(condition_id, snapshot_time DESC);

SELECT create_hypertable('market_snapshots', 'snapshot_time', if_not_exists => TRUE);
