import uuid
from typing import Optional, Union

from pydantic import BaseModel

from swarm.logging.log_processor import AnalysisTypes, FilterCriteria


class UpdatePromptRequest(BaseModel):
    name: str
    template: Optional[str] = None
    input_types: Optional[dict[str, str]] = {}


class IngestFilesRequest(BaseModel):
    document_ids: Optional[list[uuid.UUID]] = None
    metadatas: Optional[list[dict]] = None
    versions: Optional[list[str]] = None


class UpdateFilesRequest(BaseModel):
    metadatas: Optional[list[dict]] = None
    document_ids: Optional[list[uuid.UUID]] = None


class SearchRequest(BaseModel):
    query: str
    vector_search_settings: Optional[dict] = None
    kg_search_settings: Optional[dict] = None


class RAGRequest(BaseModel):
    query: str
    vector_search_settings: Optional[dict] = None
    kg_search_settings: Optional[dict] = None
    rag_generation_config: Optional[dict] = None


class EvalRequest(BaseModel):
    query: str
    context: str
    completion: str


class DeleteRequest(BaseModel):
    keys: list[str]
    values: list[Union[bool, int, str]]


class AnalyticsRequest(BaseModel):
    filter_criteria: FilterCriteria
    analysis_types: AnalysisTypes


class UsersOverviewRequest(BaseModel):
    user_ids: Optional[list[uuid.UUID]]


class DocumentsOverviewRequest(BaseModel):
    document_ids: Optional[list[uuid.UUID]]
    user_ids: Optional[list[uuid.UUID]]


class DocumentChunksRequest(BaseModel):
    document_id: uuid.UUID


class LogsRequest(BaseModel):
    log_type_filter: Optional[str] = (None,)
    max_runs_requested: int = 100


class PrintRelationshipsRequest(BaseModel):
    limit: int = 100


class ExtractionRequest(BaseModel):
    entity_types: list[str]
    relations: list[str]
