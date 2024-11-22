from typing import Optional

from pydantic import BaseModel

from swarm.base import (
    AsyncPipe,
    EmbeddingProvider,
    KGProvider,
    LLMProvider,
    PromptProvider,
    VectorDBProvider,
)
from swarm.pipelines import (
    IngestionPipeline
)


class SwarmProviders(BaseModel):
    vector_db: Optional[VectorDBProvider]
    embedding: Optional[EmbeddingProvider]
    llm: Optional[LLMProvider]
    prompt: Optional[PromptProvider]
    kg: Optional[KGProvider]

    class Config:
        arbitrary_types_allowed = True


class SwarmPipes(BaseModel):
    parsing_pipe: Optional[AsyncPipe]
    embedding_pipe: Optional[AsyncPipe]
    vector_storage_pipe: Optional[AsyncPipe]
    vector_search_pipe: Optional[AsyncPipe]
    kg_pipe: Optional[AsyncPipe]
    kg_storage_pipe: Optional[AsyncPipe]
    kg_agent_search_pipe: Optional[AsyncPipe]

    class Config:
        arbitrary_types_allowed = True


class SwarmPipelines(BaseModel):
    ingestion_pipeline: IngestionPipeline
    class Config:
        arbitrary_types_allowed = True
