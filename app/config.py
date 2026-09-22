"""Настройки процесса из `.env`."""

from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    API_TOKEN: str = ""

    BASE_URL: str = ""
    API_KEY: str = ""
    MODEL: str = ""

    HOST: str = "127.0.0.1"
    PORT: int = 8000

    DATA_DIR: str = "./data"
    SQLITE_PATH: str = "./data/tasks.db"
    LOG_DIR: str = "./data/logs"
    PERFORMANCE_LOG: str = "./data/logs/performance_log.csv"
    LOG_ENABLED: bool = True
    LOG_MAX_BYTES: int = 5 * 1024 * 1024
    LOG_BACKUP_COUNT: int = 5
    PERFORMANCE_LOG_ENABLED: bool = True
    METRICS_ENABLED: bool = True

    WORKERS: int = Field(default=1, validation_alias=AliasChoices("WORKERS", "WORKERS_MAX"))
    WORKER_QUEUE_SIZE: int = 4
    MAX_PAYLOAD_BYTES: int = 10 * 1024 * 1024
    TASK_TTL_SEC: int = 3600
    TASK_MAX_RESTARTS: int = 1
    TASK_TIMEOUT_SEC: float = 3600

    LLM_TIMEOUT_SEC: float = 120
    LLM_MAX_RETRIES: int = 2
    LLM_PROBE_TTL_SEC: float = 15
    MAX_TOKENS: int | None = None
    TEMPERATURE: float = 0.2

    @field_validator("MAX_TOKENS", mode="before")
    @classmethod
    def _empty_max_tokens(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("TASK_MAX_RESTARTS")
    @classmethod
    def _non_negative_restarts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("TASK_MAX_RESTARTS must be >= 0")
        return value

    @field_validator("TASK_TIMEOUT_SEC")
    @classmethod
    def _non_negative_task_timeout(cls, value: float) -> float:
        if value < 0:
            raise ValueError("TASK_TIMEOUT_SEC must be >= 0")
        return value

    @field_validator("MAX_PAYLOAD_BYTES")
    @classmethod
    def _positive_payload_limit(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("MAX_PAYLOAD_BYTES must be > 0")
        return value

    @field_validator("LOG_MAX_BYTES")
    @classmethod
    def _positive_log_max_bytes(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("LOG_MAX_BYTES must be > 0")
        return value

    @field_validator("LOG_BACKUP_COUNT")
    @classmethod
    def _positive_log_backup_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("LOG_BACKUP_COUNT must be >= 1")
        return value

    @field_validator("WORKERS")
    @classmethod
    def _positive_workers(cls, value: int) -> int:
        if value < 1:
            raise ValueError("WORKERS must be >= 1")
        return value

    @field_validator("LLM_MAX_RETRIES")
    @classmethod
    def _non_negative_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("LLM_MAX_RETRIES must be >= 0")
        return value

    @field_validator("LLM_PROBE_TTL_SEC")
    @classmethod
    def _non_negative_probe_ttl(cls, value: float) -> float:
        if value < 0:
            raise ValueError("LLM_PROBE_TTL_SEC must be >= 0")
        return value

    def llm_configured(self) -> bool:
        return bool(self.BASE_URL.strip() and self.API_KEY.strip() and self.MODEL.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
