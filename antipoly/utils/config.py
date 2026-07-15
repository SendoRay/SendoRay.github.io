import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIConfig:
    gamma_base: str = "https://gamma-api.polymarket.com"
    clob_base: str = "https://clob.polymarket.com"
    data_base: str = "https://data-api.polymarket.com"
    goldsky_subgraph: str = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"


@dataclass
class CollectorConfig:
    poll_interval_seconds: int = 60
    gamma_poll_interval_seconds: int = 300
    batch_limit: int = 500
    request_timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_seconds: int = 5


@dataclass
class DetectorConfig:
    l1_low_probability_threshold: float = 0.30
    l1_min_amount_usd: float = 5000.0
    ml_score_high: float = 0.80
    ml_score_low: float = 0.60
    rule_new_wallet_max_age_days: int = 7
    rule_low_probability_max: float = 0.10
    rule_min_amount_usd: float = 10000.0


@dataclass
class AlerterConfig:
    dedup_window_minutes: int = 30
    telegram_enabled: bool = True


@dataclass
class TrainerConfig:
    retrain_interval_days: int = 7
    training_window_days: int = 90
    contamination: float = 0.05


@dataclass
class DashboardConfig:
    page_size: int = 50


@dataclass
class Config:
    api: APIConfig = field(default_factory=APIConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    alerter: AlerterConfig = field(default_factory=AlerterConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    # Environment variables
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    thegraph_api_key: str = ""
    postgres_dsn: str = ""

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "Config":
        config_path = Path(config_path)
        raw: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}

        def populate(dataclass_cls, section_key):
            section = raw.get(section_key, {})
            kwargs = {}
            for f_name in dataclass_cls.__dataclass_fields__:
                if f_name in section:
                    kwargs[f_name] = section[f_name]
            return dataclass_cls(**kwargs)

        cfg = cls(
            api=populate(APIConfig, "api"),
            collector=populate(CollectorConfig, "collector"),
            detector=populate(DetectorConfig, "detector"),
            alerter=populate(AlerterConfig, "alerter"),
            trainer=populate(TrainerConfig, "trainer"),
            dashboard=populate(DashboardConfig, "dashboard"),
        )

        cfg.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        cfg.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        cfg.thegraph_api_key = os.getenv("THEGRAPH_API_KEY", "")

        pg_user = os.getenv("POSTGRES_USER", "antipoly")
        pg_pass = os.getenv("POSTGRES_PASSWORD", "change_me_in_production")
        pg_db = os.getenv("POSTGRES_DB", "antipoly")
        pg_host = os.getenv("POSTGRES_HOST", "localhost")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        cfg.postgres_dsn = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"

        return cfg


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.load()
    return _config
