import uuid
from abc import abstractmethod
from typing import Any, AsyncGenerator, Optional

from swarm.base import (
    AsyncState,
    LLMProvider,
    PipeType,
    PromptProvider,
    AsyncPipe
)

from swarm.logging.kv_logger import KVLoggingSingleton,
from swarm.models.llm import GenerationConfig



class GeneratorPipe(AsyncPipe):
    class Config(AsyncPipe.PipeConfig):
        name: str
        task_prompt: str
        system_prompt: str = "default_system"

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_provider: PromptProvider,
        type: PipeType = PipeType.GENERATOR,
        config: Optional[Config] = None,
        pipe_logger: Optional[KVLoggingSingleton] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            type=type,
            config=config or self.Config(),
            pipe_logger=pipe_logger,
            *args,
            **kwargs,
        )
        self.llm_provider = llm_provider
        self.prompt_provider = prompt_provider

    @abstractmethod
    async def _run_logic(
        self,
        input: AsyncPipe.Input,
        state: AsyncState,
        run_id: uuid.UUID,
        rag_generation_config: GenerationConfig,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        pass

    @abstractmethod
    def _get_message_payload(
        self, message: str, *args: Any, **kwargs: Any
    ) -> list:
        pass
