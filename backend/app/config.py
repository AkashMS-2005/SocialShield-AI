"""
config.py – Application settings loaded from environment / .env file.
All values have safe defaults for local development.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "SocialShield AI"
    app_version: str = "0.1.0"
    debug: bool = True

    # Database
    database_url: str = "sqlite+aiosqlite:///./socialshield.db"

    # CORS – tighten in production
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

    # ML model directory (relative to backend root)
    model_dir: str = "app/models"

    # Risk Engine Thresholds (0 - 100 scale)
    risk_low_threshold: float = 30.0         # Below 30 is Low risk
    risk_high_threshold: float = 70.0        # 70 and above is High risk

    # Base modality weight distribution (re-normalized dynamically over available modalities)
    weight_ml: float = 0.40
    weight_nlp: float = 0.30
    weight_behavioral: float = 0.20
    weight_network: float = 0.10

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "protected_namespaces": ("settings_",),
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()

