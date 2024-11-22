import os
from typing import Optional, Type

from swarm.base import (
    AsyncPipe,
    EmbeddingProvider,
    LLMProvider,
    PromptProvider,
    VectorDBProvider,
)
from swarm.pipelines import (
    EvalPipeline,
    IngestionPipeline,
)
from swarm.swarm import SwarmEngine, Swarm
from .config import SwarmConfig
from .factory import SwarmPipeFactory, SwarmPipelineFactory, SwarmProviderFactory


class SwarmBuilder:
    current_file_path = os.path.dirname(__file__)
    config_root = os.path.join(
        current_file_path, "..", "..", "examples", "configs"
    )
    CONFIG_OPTIONS = {
        "default": None,
        "local_ollama": os.path.join(config_root, "local_ollama.json"),
        "local_ollama_rerank": os.path.join(
            config_root, "local_ollama_rerank.json"
        ),
        "neo4j_kg": os.path.join(config_root, "neo4j_kg.json"),
        "local_neo4j_kg": os.path.join(config_root, "local_neo4j_kg.json"),
        "postgres_logging": os.path.join(config_root, "postgres_logging.json"),
    }

    @staticmethod
    def _get_config(config_name):
        if config_name is None:
            return SwarmConfig.from_json()
        if config_name in SwarmBuilder.CONFIG_OPTIONS:
            return SwarmConfig.from_json(SwarmBuilder.CONFIG_OPTIONS[config_name])
        raise ValueError(f"Invalid config name: {config_name}")

    def __init__(
        self,
        config: Optional[SwarmConfig] = None,
        from_config: Optional[str] = None,
    ):
        if config and from_config:
            raise ValueError("Cannot specify both config and config_name")
        self.config = config or SwarmBuilder._get_config(from_config)
        self.provider_factory_override: Optional[Type[SwarmProviderFactory]] = (
            None
        )
        self.pipe_factory_override: Optional[SwarmPipeFactory] = None
        self.pipeline_factory_override: Optional[SwarmPipelineFactory] = None
        self.vector_db_provider_override: Optional[VectorDBProvider] = None
        self.embedding_provider_override: Optional[EmbeddingProvider] = None
        self.llm_provider_override: Optional[LLMProvider] = None
        self.prompt_provider_override: Optional[PromptProvider] = None
        self.parsing_pipe_override: Optional[AsyncPipe] = None
        self.embedding_pipe_override: Optional[AsyncPipe] = None
        self.vector_storage_pipe_override: Optional[AsyncPipe] = None
        self.ingestion_pipeline: Optional[IngestionPipeline] = None
       

    
    def with_provider_factory(self, factory: Type[SwarmProviderFactory]):
        self.provider_factory_override = factory
        return self

    def with_pipe_factory(self, factory: SwarmPipeFactory):
        self.pipe_factory_override = factory
        return self

    def with_pipeline_factory(self, factory: SwarmPipelineFactory):
        self.pipeline_factory_override = factory
        return self

    def with_vector_db_provider(self, provider: VectorDBProvider):
        self.vector_db_provider_override = provider
        return self

    def with_embedding_provider(self, provider: EmbeddingProvider):
        self.embedding_provider_override = provider
        return self

   
    def with_llm_provider(self, provider: LLMProvider):
        self.llm_provider_override = provider
        return self

    def with_prompt_provider(self, provider: PromptProvider):
        self.prompt_provider_override = provider
        return self

    def with_parsing_pipe(self, pipe: AsyncPipe):
        self.parsing_pipe_override = pipe
        return self

    def with_embedding_pipe(self, pipe: AsyncPipe):
        self.embedding_pipe_override = pipe
        return self

    def with_vector_storage_pipe(self, pipe: AsyncPipe):
        self.vector_storage_pipe_override = pipe
        return self


    def with_ingestion_pipeline(self, pipeline: IngestionPipeline):
        self.ingestion_pipeline = pipeline
        return self



    def build(self, *args, **kwargs) -> Swarm:
        provider_factory = self.provider_factory_override or SwarmProviderFactory
        pipe_factory = self.pipe_factory_override or SwarmPipeFactory
        pipeline_factory = self.pipeline_factory_override or SwarmPipelineFactory

        providers = provider_factory(self.config).create_providers(
            vector_db_provider_override=self.vector_db_provider_override,
            embedding_provider_override=self.embedding_provider_override,
            llm_provider_override=self.llm_provider_override,
            prompt_provider_override=self.prompt_provider_override,
            *args,
            **kwargs,
        )

        pipes = pipe_factory(self.config, providers).create_pipes(
            parsing_pipe_override=self.parsing_pipe_override,
            embedding_pipe_override=self.embedding_pipe_override,
            vector_storage_pipe_override=self.vector_storage_pipe_override,
            *args,
            **kwargs,
        )

        pipelines = pipeline_factory(self.config, pipes).create_pipelines(
            ingestion_pipeline=self.ingestion_pipeline,
            *args,
            **kwargs,
        )

        engine =  SwarmEngine(
            self.config, providers, pipelines
        )
        
        return Swarm(engine=engine)


