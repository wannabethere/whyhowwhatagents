import asyncio
import copy
import logging
import uuid
from typing import Any, AsyncGenerator, Optional, Union

from swarm.splitter.text import TextSplitter
from swarm.base import (
    AsyncPipe,
    AsyncState,
    EmbeddingProvider,
    PipeType,
    generate_id_from_label
)
from swarm.models.document import  Extraction, Fragment, FragmentType

from swarm.models.exception import SwarmDocumentProcessingError
from swarm.logging.kv_logger import KVLoggingSingleton
from swarm.models.vector import Vector, VectorEntry

logger = logging.getLogger(__name__)


class EmbeddingPipe(AsyncPipe):
    """
    Embeds and stores documents using a specified embedding model and database.
    """

    class Input(AsyncPipe.Input):
        message: AsyncGenerator[
            Union[Extraction, SwarmDocumentProcessingError], None
        ]

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        text_splitter: TextSplitter,
        embedding_batch_size: int = 1,
        id_prefix: str = "demo",
        pipe_logger: Optional[KVLoggingSingleton] = None,
        type: PipeType = PipeType.INGESTOR,
        config: Optional[AsyncPipe.PipeConfig] = None,
        *args,
        **kwargs,
    ):
        """
        Initializes the embedding pipe with necessary components and configurations.
        """
        super().__init__(
            pipe_logger=pipe_logger,
            type=type,
            config=config
            or AsyncPipe.PipeConfig(name="default_embedding_pipe"),
        )
        self.embedding_provider = embedding_provider
        self.text_splitter = text_splitter
        self.embedding_batch_size = embedding_batch_size
        self.id_prefix = id_prefix
        self.pipe_run_info = None

    async def fragment(
        self, extraction: Extraction, run_id: uuid.UUID
    ) -> AsyncGenerator[Fragment, None]:
        """
        Splits text into manageable chunks for embedding.
        """
        if not isinstance(extraction, Extraction):
            raise ValueError(
                f"Expected an Extraction, but received {type(extraction)}."
            )
        if not isinstance(extraction.data, str):
            raise ValueError(
                f"Expected a string, but received {type(extraction.data)}."
            )
        text_chunks = [
            ele.page_content
            for ele in self.text_splitter.create_documents([extraction.data])
        ]
        for iteration, chunk in enumerate(text_chunks):
            fragment = Fragment(
                id=generate_id_from_label(f"{extraction.id}-{iteration}"),
                type=FragmentType.TEXT,
                data=chunk,
                metadata=copy.deepcopy(extraction.metadata),
                extraction_id=extraction.id,
                document_id=extraction.document_id,
            )
            yield fragment
            iteration += 1

    async def transform_fragments(
        self, fragments: list[Fragment], metadatas: list[dict]
    ) -> AsyncGenerator[Fragment, None]:
        """
        Transforms text chunks based on their metadata, e.g., adding prefixes.
        """
        async for fragment, metadata in zip(fragments, metadatas):
            if "chunk_prefix" in metadata:
                prefix = metadata.pop("chunk_prefix")
                fragment.data = f"{prefix}\n{fragment.data}"
            yield fragment

    async def embed(self, fragments: list[Fragment]) -> list[float]:
        return await self.embedding_provider.async_get_embeddings(
            [fragment.data for fragment in fragments],
            EmbeddingProvider.PipeStage.BASE,
        )

    async def _process_batch(
        self, fragment_batch: list[Fragment]
    ) -> list[VectorEntry]:
        """
        Embeds a batch of fragments and yields vector entries.
        """
        vectors = await self.embed(fragment_batch)
        return [
            VectorEntry(
                id=fragment.id,
                vector=Vector(data=raw_vector),
                metadata={
                    "document_id": fragment.document_id,
                    "extraction_id": fragment.extraction_id,
                    "text": fragment.data,
                    **fragment.metadata,
                },
            )
            for raw_vector, fragment in zip(vectors, fragment_batch)
        ]

    async def _process_and_enqueue_batch(
        self, fragment_batch: list[Fragment], vector_entry_queue: asyncio.Queue
    ):
        try:
            batch_result = await self._process_batch(fragment_batch)
            for vector_entry in batch_result:
                await vector_entry_queue.put(vector_entry)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            await vector_entry_queue.put(
                SwarmDocumentProcessingError(
                    error_message=str(e),
                    document_id=fragment_batch[0].document_id,
                )
            )
        finally:
            await vector_entry_queue.put(None)  # Signal completion

    async def _run_logic(
        self,
        input: Input,
        state: AsyncState,
        run_id: uuid.UUID,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[Union[SwarmDocumentProcessingError, VectorEntry], None]:
        """
        Executes the embedding pipe: chunking, transforming, embedding, and storing documents.
        """
        vector_entry_queue = asyncio.Queue()
        fragment_batch = []
        active_tasks = 0

        fragment_info = {}
        async for extraction in input.message:
            if isinstance(extraction, SwarmDocumentProcessingError):
                yield extraction
                continue

            async for fragment in self.fragment(extraction, run_id):
                if extraction.document_id in fragment_info:
                    fragment_info[extraction.document_id] += 1
                else:
                    fragment_info[extraction.document_id] = 0  # Start with 0
                fragment.metadata["chunk_order"] = fragment_info[
                    extraction.document_id
                ]

                version = fragment.metadata.get("version", "v0")

                # Ensure fragment ID is set correctly
                if not fragment.id:
                    fragment.id = generate_id_from_label(
                        f"{extraction.id}-{fragment_info[extraction.document_id]}-{version}"
                    )

                fragment_batch.append(fragment)
                if len(fragment_batch) >= self.embedding_batch_size:
                    asyncio.create_task(
                        self._process_and_enqueue_batch(
                            fragment_batch.copy(), vector_entry_queue
                        )
                    )
                    active_tasks += 1
                    fragment_batch.clear()

        logger.debug(
            f"Fragmented the input document ids into counts as shown: {fragment_info}"
        )

        if fragment_batch:
            asyncio.create_task(
                self._process_and_enqueue_batch(
                    fragment_batch.copy(), vector_entry_queue
                )
            )
            active_tasks += 1

        while active_tasks > 0:
            vector_entry = await vector_entry_queue.get()
            if vector_entry is None:  # Check for termination signal
                active_tasks -= 1
            elif isinstance(vector_entry, Exception):
                yield vector_entry  # Propagate the exception
                active_tasks -= 1
            else:
                yield vector_entry
