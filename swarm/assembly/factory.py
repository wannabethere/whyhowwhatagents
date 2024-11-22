import logging
import os
from typing import Any, Optional

from swarm.base import (
    AsyncPipe,
    EmbeddingConfig,
    EmbeddingProvider,
    KGProvider,
    LLMConfig,
    LLMProvider,
    PromptProvider,
    VectorDBConfig,
    VectorDBProvider,
)
from swarm.logging.kv_logger import KVLoggingSingleton
from swarm.pipelines import (
    IngestionPipeline   
)


from swarm.abstractions import SwarmPipelines, SwarmPipes, SwarmProviders
from .config import SwarmConfig

logger = logging.getLogger(__name__)


class SwarmProviderFactory:
    def __init__(self, config: SwarmConfig):
        self.config = config

    def create_vector_db_provider(
        self, vector_db_config: VectorDBConfig, *args, **kwargs
    ) -> VectorDBProvider:
        vector_db_provider: Optional[VectorDBProvider] = None
        if not vector_db_provider:
            raise ValueError("Vector database provider not found")

        if not self.config.embedding.base_dimension:
            raise ValueError("Search dimension not found in embedding config")

        vector_db_provider.initialize_collection(
            self.config.embedding.base_dimension
        )
        return vector_db_provider

    def create_embedding_provider(
        self, embedding: EmbeddingConfig, *args, **kwargs
    ) -> EmbeddingProvider:
        embedding_provider: Optional[EmbeddingProvider] = None

        if embedding.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError(
                    "Must set OPENAI_API_KEY in order to initialize OpenAIEmbeddingProvider."
                )
            from swarm.providers.embeddings.openai import OpenAIEmbeddingProvider

            embedding_provider = OpenAIEmbeddingProvider(embedding)
        elif embedding.provider == "sentence-transformers":
            from swarm.providers.embeddings.sentence_transformer import (
                SentenceTransformerEmbeddingProvider
            )

            embedding_provider = SentenceTransformerEmbeddingProvider(
                embedding
            )
        elif embedding is None:
            embedding_provider = None
        else:
            raise ValueError(
                f"Embedding provider {embedding.provider} not supported"
            )

        return embedding_provider

    
    def create_llm_provider(
        self, llm_config: LLMConfig, *args, **kwargs
    ) -> LLMProvider:
        llm_provider: Optional[LLMProvider] = None
        if llm_config.provider == "langchain":
            from swarm.providers.llms.langchain.langchain import LangchainLLM
            llm_provider = LangchainLLM(llm_config)
        else:
            raise ValueError(
                f"Language model provider {llm_config.provider} not supported"
            )
        if not llm_provider:
            raise ValueError("Language model provider not found")
        return llm_provider

    def create_prompt_provider(
        self, prompt_config, *args, **kwargs
    ) -> PromptProvider:
        prompt_provider = None
        if prompt_config.provider == "local":
            from swarm.prompts.swarmpromptprovider import SwarmPromptProvider
            prompt_provider = SwarmPromptProvider()
        else:
            raise ValueError(
                f"Prompt provider {prompt_config.provider} not supported"
            )
        return prompt_provider

    def create_kg_provider(self, kg_config, *args, **kwargs):
        if kg_config.provider == "neo4j":
            from swarm.providers.kg import Neo4jKGProvider

            return Neo4jKGProvider(kg_config)
        elif kg_config.provider is None:
            return None
        else:
            raise ValueError(
                f"KG provider {kg_config.provider} not supported."
            )

    def create_providers(
        self,
        vector_db_provider_override: Optional[VectorDBProvider] = None,
        embedding_provider_override: Optional[EmbeddingProvider] = None,
        llm_provider_override: Optional[LLMProvider] = None,
        prompt_provider_override: Optional[PromptProvider] = None,
        kg_provider_override: Optional[KGProvider] = None,
        *args,
        **kwargs,
    ) -> SwarmProviders:
        prompt_provider = (
            prompt_provider_override
            or self.create_prompt_provider(self.config.prompt, *args, **kwargs)
        )
        return SwarmProviders(
            vector_db=vector_db_provider_override
            or self.create_vector_db_provider(
                self.config.vector_database, *args, **kwargs
            ),
            embedding=embedding_provider_override
            or self.create_embedding_provider(
                self.config.embedding, *args, **kwargs
            ),
            llm=llm_provider_override
            or self.create_llm_provider(
                self.config.completions, *args, **kwargs
            ),
            prompt=prompt_provider_override
            or self.create_prompt_provider(
                self.config.prompt, *args, **kwargs
            ),
            kg=kg_provider_override
            or self.create_kg_provider(self.config.kg, *args, **kwargs),
        )


class SwarmPipeFactory:
    def __init__(self, config: SwarmConfig, providers: SwarmProviders):
        self.config = config
        self.providers = providers

    def create_pipes(
        self,
        parsing_pipe_override: Optional[AsyncPipe] = None,
        embedding_pipe_override: Optional[AsyncPipe] = None,
        kg_pipe_override: Optional[AsyncPipe] = None,
        kg_storage_pipe_override: Optional[AsyncPipe] = None,
        kg_agent_pipe_override: Optional[AsyncPipe] = None,
        vector_storage_pipe_override: Optional[AsyncPipe] = None,
        *args,
        **kwargs,
    ) -> SwarmPipes:
        return SwarmPipes(
            parsing_pipe=parsing_pipe_override
            or self.create_parsing_pipe(
                self.config.ingestion.get("excluded_parsers"), *args, **kwargs
            ),
            embedding_pipe=embedding_pipe_override
            or self.create_embedding_pipe(*args, **kwargs),
            kg_pipe=kg_pipe_override or self.create_kg_pipe(*args, **kwargs),
            kg_storage_pipe=kg_storage_pipe_override
            or self.create_kg_storage_pipe(*args, **kwargs),
            kg_agent_search_pipe=kg_agent_pipe_override
            or self.create_kg_agent_pipe(*args, **kwargs),
            vector_storage_pipe=vector_storage_pipe_override
            or self.create_vector_storage_pipe(*args, **kwargs),
        )

    def create_parsing_pipe(
        self, excluded_parsers: Optional[list] = None, *args, **kwargs
    ) -> Any:
        from swarm.pipes import ParsingPipe

        return ParsingPipe(excluded_parsers=excluded_parsers or [])

    def create_embedding_pipe(self, *args, **kwargs) -> Any:
        if self.config.embedding.provider is None:
            return None

        from swarm.splitter import RecursiveCharacterTextSplitter
        from swarm.pipes import EmbeddingPipe

        text_splitter_config = self.config.embedding.extra_fields.get(
            "text_splitter"
        )
        if not text_splitter_config:
            raise ValueError(
                "Text splitter config not found in embedding config"
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_splitter_config["chunk_size"],
            chunk_overlap=text_splitter_config["chunk_overlap"],
            length_function=len,
            is_separator_regex=False,
        )
        return EmbeddingPipe(
            embedding_provider=self.providers.embedding,
            vector_db_provider=self.providers.vector_db,
            text_splitter=text_splitter,
            embedding_batch_size=self.config.embedding.batch_size,
        )

    def create_vector_storage_pipe(self, *args, **kwargs) -> Any:
        if self.config.embedding.provider is None:
            return None

        from swarm.pipes import VectorStoragePipe

        return VectorStoragePipe(vector_db_provider=self.providers.vector_db)


    def create_kg_pipe(self, *args, **kwargs) -> Any:
        if self.config.kg.provider is None:
            return None

        from swarm.splitter import RecursiveCharacterTextSplitter
        from swarm.pipes import KGExtractionPipe

        text_splitter_config = self.config.kg.extra_fields.get("text_splitter")
        if not text_splitter_config:
            raise ValueError("Text splitter config not found in kg config.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_splitter_config["chunk_size"],
            chunk_overlap=text_splitter_config["chunk_overlap"],
            length_function=len,
            is_separator_regex=False,
        )
        return KGExtractionPipe(
            kg_provider=self.providers.kg,
            llm_provider=self.providers.llm,
            prompt_provider=self.providers.prompt,
            vector_db_provider=self.providers.vector_db,
            text_splitter=text_splitter,
            kg_batch_size=self.config.kg.batch_size,
        )

    def create_kg_storage_pipe(self, *args, **kwargs) -> Any:
        if self.config.kg.provider is None:
            return None

        from swarm.pipes import KGStoragePipe

        return KGStoragePipe(
            kg_provider=self.providers.kg,
            embedding_provider=self.providers.embedding,
        )

    def create_kg_agent_pipe(self, *args, **kwargs) -> Any:
        if self.config.kg.provider is None:
            return None

        from swarm.pipes import KGAgentSearchPipe

        return KGAgentSearchPipe(
            kg_provider=self.providers.kg,
            llm_provider=self.providers.llm,
            prompt_provider=self.providers.prompt,
        )

   
   

class SwarmPipelineFactory:
    def __init__(self, config: SwarmConfig, pipes: SwarmPipes):
        self.config = config
        self.pipes = pipes

    def create_ingestion_pipeline(self, *args, **kwargs) -> IngestionPipeline:
        """factory method to create an ingestion pipeline."""
        ingestion_pipeline = IngestionPipeline()

        ingestion_pipeline.add_pipe(
            pipe=self.pipes.parsing_pipe, parsing_pipe=True
        )
        # Add embedding pipes if provider is set
        if self.config.embedding.provider is not None:
            ingestion_pipeline.add_pipe(
                self.pipes.embedding_pipe, embedding_pipe=True
            )
            ingestion_pipeline.add_pipe(
                self.pipes.vector_storage_pipe, embedding_pipe=True
            )
        # Add KG pipes if provider is set
        if self.config.kg.provider is not None:
            ingestion_pipeline.add_pipe(self.pipes.kg_pipe, kg_pipe=True)
            ingestion_pipeline.add_pipe(
                self.pipes.kg_storage_pipe, kg_pipe=True
            )

        return ingestion_pipeline

   


    def create_pipelines(
        self,
        ingestion_pipeline: Optional[IngestionPipeline] = None,
        *args,
        **kwargs,
    ) -> SwarmPipelines:
        try:
            self.configure_logging()
        except Exception as e:
            logger.warn(f"Error configuring logging: {e}")
        search_pipeline = search_pipeline or self.create_search_pipeline(
            *args, **kwargs
        )
        return SwarmPipelines(
            ingestion_pipeline=ingestion_pipeline or self.create_ingestion_pipeline(*args, **kwargs)
        )

    def configure_logging(self):
        KVLoggingSingleton.configure(self.config.logging)
