
from enum import Enum
from typing import Optional
from .base_provider import  ProviderConfig


class LoggingConfig(ProviderConfig):
    provider: str = "local"
    log_table: str = "logs"
    log_info_table: str = "logs_pipeline_info"
    logging_path: Optional[str] = None

    def validate(self) -> None:
        pass

    @property
    def supported_providers(self) -> list[str]:
        return ["local", "postgres", "redis"]