import asyncio

import pytest

from swarm.base import EmbeddingConfig, generate_id_from_label
from swarm.models.search import VectorSearchResult

from swarm.providers import (
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)


@pytest.fixture(scope="session", autouse=True)
def event_loop_policy():
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


@pytest.fixture(scope="function")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
    asyncio.set_event_loop(None)


@pytest.fixture(scope="session", autouse=True)
async def cleanup_tasks():
    yield
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.fixture
def openai_provider():
    config = EmbeddingConfig(
        provider="openai",
        base_model="text-embedding-3-small",
        base_dimension=1536,
    )
    return OpenAIEmbeddingProvider(config)


def test_openai_initialization(openai_provider):
    assert isinstance(openai_provider, OpenAIEmbeddingProvider)
    assert openai_provider.base_model == "text-embedding-3-small"
    assert openai_provider.base_dimension == 1536


def test_openai_invalid_provider_initialization():
    config = EmbeddingConfig(provider="invalid_provider")
    with pytest.raises(ValueError):
        OpenAIEmbeddingProvider(config)


def test_openai_get_embedding(openai_provider):
    embedding = openai_provider.get_embedding("test text")
    assert len(embedding) == 1536
    assert isinstance(embedding, list)


@pytest.mark.asyncio
async def test_openai_async_get_embedding(openai_provider):
    try:
        embedding = await openai_provider.async_get_embedding("test text")
        assert len(embedding) == 1536
        assert isinstance(embedding, list)
    except asyncio.CancelledError:
        pass  # Task cancelled as expected


def test_openai_get_embeddings(openai_provider):
    embeddings = openai_provider.get_embeddings(["text1", "text2"])
    assert len(embeddings) == 2
    assert all(len(emb) == 1536 for emb in embeddings)


@pytest.mark.asyncio
async def test_openai_async_get_embeddings(openai_provider):
    try:
        embeddings = await openai_provider.async_get_embeddings(
            ["text1", "text2"]
        )
        assert len(embeddings) == 2
        assert all(len(emb) == 1536 for emb in embeddings)
    except asyncio.CancelledError:
        pass  # Task cancelled as expected


def test_openai_tokenize_string(openai_provider):
    tokens = openai_provider.tokenize_string(
        "test text", "text-embedding-3-small"
    )
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)


@pytest.fixture
def sentence_transformer_provider():
    config = EmbeddingConfig(
        provider="sentence-transformers",
        base_model="mixedbread-ai/mxbai-embed-large-v1",
        base_dimension=512,
        rerank_model="jinaai/jina-reranker-v1-turbo-en",
        rerank_dimension=384,
    )
    return SentenceTransformerEmbeddingProvider(config)


def test_sentence_transformer_initialization(sentence_transformer_provider):
    assert isinstance(
        sentence_transformer_provider, SentenceTransformerEmbeddingProvider
    )
    assert sentence_transformer_provider.do_search
    # assert sentence_transformer_provider.do_rerank


def test_sentence_transformer_invalid_provider_initialization():
    config = EmbeddingConfig(provider="invalid_provider")
    with pytest.raises(ValueError):
        SentenceTransformerEmbeddingProvider(config)


def test_sentence_transformer_get_embedding(sentence_transformer_provider):
    embedding = sentence_transformer_provider.get_embedding("test text")
    assert len(embedding) == 512
    assert isinstance(embedding, list)


def test_sentence_transformer_get_embeddings(sentence_transformer_provider):
    embeddings = sentence_transformer_provider.get_embeddings(
        ["text1", "text2"]
    )
    assert len(embeddings) == 2
    assert all(len(emb) == 512 for emb in embeddings)


def test_sentence_transformer_rerank(sentence_transformer_provider):
    results = [
        VectorSearchResult(
            id=generate_id_from_label("x"),
            score=0.9,
            metadata={"text": "doc1"},
        ),
        VectorSearchResult(
            id=generate_id_from_label("y"),
            score=0.8,
            metadata={"text": "doc2"},
        ),
    ]
    reranked_results = sentence_transformer_provider.rerank("query", results)
    assert len(reranked_results) == 2
    assert reranked_results[0].metadata["text"] == "doc1"
    assert reranked_results[1].metadata["text"] == "doc2"


def test_sentence_transformer_tokenize_string(sentence_transformer_provider):
    with pytest.raises(ValueError):
        sentence_transformer_provider.tokenize_string("test text")
