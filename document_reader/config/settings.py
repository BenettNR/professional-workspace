from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    claude_model: str = Field("claude-sonnet-4-6", alias="CLAUDE_MODEL")

    max_file_size_mb: int = Field(50, alias="MAX_FILE_SIZE_MB")
    supported_formats: list[str] = ["pdf", "csv", "xlsx", "xls", "png", "jpg", "jpeg", "tiff"]

    pii_keep_last_digits: int = 4
    merchant_cache_size: int = 1000

    log_level: str = Field("INFO", alias="LOG_LEVEL")

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


settings = Settings()
