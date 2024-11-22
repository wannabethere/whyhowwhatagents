import random

import pytest
from dotenv import load_dotenv

from swarm.base import (
    
    VectorDBConfig,
    VectorDBProvider,
    generate_id_from_label,
)

from swarm.models.vector import Vector, VectorEntry
from swarm.providers.vector_dbs.chromadb.chromadb_vector import ChromaVectorDB
#from r2r.providers.vector_dbs import PGVectorDB

load_dotenv()


# Sample vector entries
def generate_random_vector_entry(id: str, dimension: int) -> VectorEntry:
    vector = [random.random() for _ in range(dimension)]
    metadata = {"key": f"value_{id}"}
    return VectorEntry(
        id=generate_id_from_label(id), vector=Vector(vector), metadata=metadata
    )


dimension = 3
num_entries = 100
sample_entries = [
    generate_random_vector_entry(f"id_{i}", dimension)
    for i in range(num_entries)
]


# Fixture for PGVectorDB
@pytest.fixture
def chroma_vector_db():
    random_collection_name = (
        f"test_collection_{random.randint(0, 1_000_000_000)}"
    )
    config = VectorDBConfig.create(
        provider="pgvector", vecs_collection=random_collection_name
    )
    db = ChromaVectorDB(config)
    db.initialize_collection(dimension=1)
    yield db
    # Teardown
    db.vx.delete_collection(
        db.config.extra_fields.get("vecs_collection", None)
    )


@pytest.mark.parametrize("db_fixture", ["chroma_vector_db"])
def test_get_metadatas(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    for entry in sample_entries:
        db.upsert(entry)

    unique_metadatas = db.get_metadatas(metadata_fields=["key"])
    unique_values = set([ele["key"] for ele in unique_metadatas])
    assert len(unique_values) == num_entries
    assert all(f"value_id_{i}" in unique_values for i in range(num_entries))


@pytest.mark.parametrize("db_fixture", ["chroma_vector_db"])
def test_db_initialization(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    assert isinstance(db, VectorDBProvider)


@pytest.mark.parametrize("db_fixture", ["chroma_vector_db"])
def test_db_copy_and_search(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    db.upsert(sample_entries[0])
    results = db.search(query_vector=sample_entries[0].vector.data)
    assert len(results) == 1
    assert results[0].id == sample_entries[0].id
    assert results[0].score == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("db_fixture", ["chroma_vector_db"])
def test_db_upsert_and_search(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    db.upsert(sample_entries[0])
    results = db.search(query_vector=sample_entries[0].vector.data)
    assert len(results) == 1
    assert results[0].id == sample_entries[0].id
    assert results[0].score == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("db_fixture", ["chroma_vector_db"])
def test_imperfect_match(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    db.upsert(sample_entries[0])
    query_vector = [val + 0.1 for val in sample_entries[0].vector.data]
    results = db.search(query_vector=query_vector)
    assert len(results) == 1
    assert results[0].id == sample_entries[0].id
    assert results[0].score < 1.0


@pytest.mark.parametrize("db_fixture", ["pg_vector_db"])
def test_bulk_insert_and_search(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    for entry in sample_entries:
        db.upsert(entry)

    query_vector = sample_entries[0].vector.data
    results = db.search(query_vector=query_vector, limit=5)
    assert len(results) == 5
    assert results[0].id == sample_entries[0].id
    assert results[0].score == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("db_fixture", ["pg_vector_db"])
def test_search_with_filters(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    for entry in sample_entries:
        db.upsert(entry)

    filtered_id = sample_entries[0].metadata["key"]
    query_vector = sample_entries[0].vector.data
    results = db.search(
        query_vector=query_vector, filters={"key": filtered_id}
    )
    assert len(results) == 1
    assert results[0].id == sample_entries[0].id
    assert results[0].metadata["key"] == filtered_id


@pytest.mark.parametrize("db_fixture", ["pg_vector_db"])
def test_delete_by_metadata(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    for entry in sample_entries:
        db.upsert(entry)

    key_to_delete = sample_entries[0].metadata["key"]
    db.delete_by_metadata(
        metadata_fields=["key"], metadata_values=[key_to_delete]
    )

    results = db.search(query_vector=sample_entries[0].vector.data)
    assert all(result.metadata["key"] != key_to_delete for result in results)


@pytest.mark.parametrize("db_fixture", ["pg_vector_db"])
def test_upsert(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    db.upsert(sample_entries[0])
    modified_entry = VectorEntry(
        id=sample_entries[0].id,
        vector=Vector([0.5, 0.5, 0.5]),
        metadata={"key": "new_value"},
    )
    db.upsert(modified_entry)

    results = db.search(query_vector=[0.5, 0.5, 0.5])
    assert len(results) == 1
    assert results[0].id == sample_entries[0].id
    assert results[0].metadata["key"] == "new_value"
