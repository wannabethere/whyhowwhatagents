import json
import logging
import os
import time
from typing import Literal, Optional, Union,Sequence, List
from uuid import uuid4

import chromadb
from chromadb.config import Settings

from swarm.base import (
    VectorDBConfig,
    VectorDBProvider
)

from swarm.models.document import DocumentInfo, UserStats
from swarm.models.vector import VectorEntry, VectorType
from swarm.models.search import VectorSearchResult


logger = logging.getLogger(__name__)


class ChromaVectorDB(VectorDBProvider):
    def __init__(self, config: VectorDBConfig):
            if not isinstance(config, VectorDBConfig):
                raise ValueError(
                    "VectorDBProvider must be initialized with a `VectorDBConfig`."
                )
            logger.info(f"Initializing VectorDBProvider with config {config}.")
            super().__init__(config)
            print(config)
            self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                    persist_directory="db/"
                                ))

            self.collection_name = self.config.extra_fields.get(
            "vecs_collection"
            ) or os.getenv("POSTGRES_VECS_COLLECTION")
            self.initialize_collection(dimension=1)
    

    def initialize_collection(self, dimension: int) -> None:
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        return
    

    
    def upsert(self, entry: VectorEntry, commit: bool = True) -> None:
        self.upsert_entries(entries= [entry], commit=True)

    
    def search(
        self,
        query_vector: list[float],
        filters: dict[str, Union[bool, int, str]] = {},
        limit: int = 10,
        *args,
        **kwargs,
    ) -> list[VectorSearchResult]:
        pass

    
    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        limit: int = 10,
        filters: Optional[dict[str, Union[bool, int, str]]] = None,
        *args,
        **kwargs,
    ) -> list[VectorSearchResult]:
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where_document={"$contains":query_text}
        )

        

    
    def create_index(self, index_type, column_name, index_options):
        #Chromadb does not support creating of Indexes.
        pass

    def upsert_entries(
        self, entries: list[VectorEntry], commit: bool = True
    ) -> None:
       ids = []
       metadatas = []
       documents = []
       embeddings =[]
       for entry in entries:
            ids.append(entry.id)
            metadata = entry.metadata
            vector = entry.vector
            metadata['vectortype'] = vector.type
            metadata['vectorlength'] = vector.length
            if vector.type = VectorType.FIXED or vector.type = VectorType.VECTOR:
                embeddings.append(vector.data)
            else:
                documents.append(vector.data)
            metadatas.append(metadata)

       if len(documents) >0:
            self.collection.upsert(
                documents=documents
                metadatas=metadatas,
                ids=ids
            )
       if len(embeddings) > 0 :
            self.collection.upsert(
                embeddings=embeddings
                metadatas=metadatas,
                ids=ids
            )

   



    
    