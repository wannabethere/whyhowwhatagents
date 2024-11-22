from abc import ABC

from swarm.logging.kv_logger import KVLoggingSingleton
from swarm.logging.run_manager import RunManager
from swarm.abstractions import SwarmPipelines, SwarmProviders
from swarm.assembly.config import SwarmConfig


class Service(ABC):
    def __init__(
        self,
        config: SwarmConfig,
        providers: SwarmProviders,
        pipelines: SwarmPipelines,
        run_manager: RunManager,
        logging_connection: KVLoggingSingleton,
    ):
        self.config = config
        self.providers = providers
        self.pipelines = pipelines
        self.run_manager = run_manager
        self.logging_connection = logging_connection
