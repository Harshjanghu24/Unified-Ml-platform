from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Unified ML Platform"
    version: str = "1.0.0"
    db_path: str = "data/platform.db"
    log_level: str = "INFO"
    testing: bool = False

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache(maxsize=1)
def get_settings():
    return Settings()
