import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

from swarm.models.document import DocumentInfo
from swarm.models.search import VectorSearchResult
from swarm.models.vector import VectorEntry
from .base_provider import Provider, ProviderConfig

logger = logging.getLogger(__name__)


class VectorDBConfig(ProviderConfig):
    provider: str

    def __post_init__(self):
        self.validate()
        # Capture additional fields
        for key, value in self.extra_fields.items():
            setattr(self, key, value)

    def validate(self) -> None:
        if self.provider not in self.supported_providers:
            raise ValueError(f"Provider '{self.provider}' is not supported.")

    @property
    def supported_providers(self) -> list[str]:
        return ["local", "pgvector"]


class VectorDBProvider(Provider, ABC):
    def __init__(self, config: VectorDBConfig):
        if not isinstance(config, VectorDBConfig):
            raise ValueError(
                "VectorDBProvider must be initialized with a `VectorDBConfig`."
            )
        logger.info(f"Initializing VectorDBProvider with config {config}.")
        super().__init__(config)

    @abstractmethod
    def initialize_collection(self, dimension: int) -> None:
        pass

   
    @abstractmethod
    def upsert(self, entry: VectorEntry, commit: bool = True) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        filters: dict[str, Union[bool, int, str]] = {},
        limit: int = 10,
        *args,
        **kwargs,
    ) -> list[VectorSearchResult]:
        pass

    @abstractmethod
    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        limit: int = 10,
        filters: Optional[dict[str, Union[bool, int, str]]] = None,
        *args,
        **kwargs,
    ) -> list[VectorSearchResult]:
        pass

    @abstractmethod
    def create_index(self, index_type, column_name, index_options):
        pass

    def upsert_entries(
        self, entries: list[VectorEntry], commit: bool = True
    ) -> None:
        for entry in entries:
            self.upsert(entry, commit=commit)

   

