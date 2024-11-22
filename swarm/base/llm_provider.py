"""Base classes for language model providers."""

import logging
from abc import abstractmethod
from typing import Optional

from swarm.models.llm import GenerationConfig
from .base_provider import Provider, ProviderConfig

logger = logging.getLogger(__name__)


class LLMConfig(ProviderConfig):
    """A base LLM config class"""

    provider: Optional[str] = None
    generation_config: Optional[GenerationConfig] = None

    def validate(self) -> None:
        if not self.provider:
            raise ValueError("Provider must be set.")

        if self.provider and self.provider not in self.supported_providers:
            raise ValueError(f"Provider '{self.provider}' is not supported.")

    @property
    def supported_providers(self) -> list[str]:
        return ["litellm", "openai","langchain","crewai"]


class LLMProvider(Provider):
    """An abstract class to provide a common interface for LLMs."""

    def __init__(
        self,
        config: LLMConfig,
    ) -> None:
        if not isinstance(config, LLMConfig):
            raise ValueError(
                "LLMProvider must be initialized with a `LLMConfig`."
            )
        logger.info(f"Initializing LLM provider with config: {config}")

        super().__init__(config)
