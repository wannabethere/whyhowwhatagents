from typing import Optional


from swarm.base import AsyncSyncMeta,syncable
from swarm.assembly.config import SwarmConfig
from swarm.base import generate_id_from_label
from swarm.services.ingestion_service import IngestionService
from swarm.logging.kv_logger import KVLoggingSingleton
from swarm.logging.run_manager import RunManager
from swarm.abstractions import SwarmProviders, SwarmPipelines




class SwarmEngine(metaclass=AsyncSyncMeta):
    def __init__(
        self,
        config: SwarmConfig,
        providers: SwarmProviders,
        pipelines: SwarmPipelines,
        run_manager: Optional[RunManager] = None,
    ):
        logging_connection = KVLoggingSingleton()
        run_manager = run_manager or RunManager(logging_connection)

        self.config = config
        self.providers = providers
        self.pipelines = pipelines
        self.logging_connection = KVLoggingSingleton()
        self.run_manager = run_manager

        self.ingestion_service = IngestionService(
            config, providers, pipelines, run_manager, logging_connection
        )
       
    @syncable
    async def aingest_documents(self, *args, **kwargs):
        return await self.ingestion_service.ingest_documents(*args, **kwargs)

    @syncable
    async def aupdate_documents(self, *args, **kwargs):
        return await self.ingestion_service.update_documents(*args, **kwargs)

    @syncable
    async def aingest_files(self, *args, **kwargs):
        return await self.ingestion_service.ingest_files(*args, **kwargs)

    @syncable
    async def aupdate_files(self, *args, **kwargs):
        return await self.ingestion_service.update_files(*args, **kwargs)

class Swarm:
    engine: SwarmEngine
    
    def __init__(
        self,
        config: Optional[SwarmConfig] = None,
        from_config: Optional[str] = None,
        *args,
        **kwargs
    ):
        
        if (config or from_config) or (
            config is None and from_config is None
        ):
            from .assembly.builder import R2RBuilder

            # Handle the case where 'from_config' is None and 'config' is None
            if not config and not from_config:
                from_config = "default"
            builder = R2RBuilder(
                config=config,
                from_config=from_config,
            )
            built = builder.build()
            self.engine = built.engine
        else:
            raise ValueError(
                "Must provide either 'engine' and 'app', or 'config'/'from_config' to build the R2R object."
            )

    def __getattr__(self, name):
        # Check if the attribute name is 'app' and return it directly
        if name == "app":
            return self.app
        elif name == "serve":
            return self.app.serve
        # Otherwise, delegate to the engine
        return getattr(self.engine, name)
    
    def getengine(self)->SwarmEngine:
        return self.engine