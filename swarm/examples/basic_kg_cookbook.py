from swarm.swarm import R2RClient

if __name__ == "__main__":
    client = R2RClient(base_url="http://localhost:8000")

    with open("john.txt", "w") as f:
        f.write("John is a person that works at Google.")
    with open("paul.txt", "w") as f:
        f.write("Paul is a person that works at Microsoft that knows John.")

    client.ingest_files(
        ["john.txt", "paul.txt"],
        metadatas=[
            {
                "title": "KG Document 1",
                "user_id": "063edaf8-3e63-4cb9-a4d6-a855f36376c3",
            },
            {
                "title": "KG Document 2",
                "user_id": "063edaf8-3e63-4cb9-a4d6-a855f36376c3",
            },
        ],
    )

    # Get the KG provider
    # neo4j_kg = app.providers.kg

    # # The expected entities
    # entity_names = ["John", "Paul", "Google", "Microsoft"]

    # print("\nEntities:")
    # for entity in entity_names:
    #     print(
    #         f"Locating {entity}:\n", neo4j_kg.get(properties={"name": entity})
    #     )

    # relationships = neo4j_kg.get_triplets(entity_names=entity_names)

    # print("\nRelationships:")
    # for triplet in relationships:
    #     source, relation, target = triplet
    #     print(f"{source} -[{relation.label}]-> {target} ")

    # # Search the vector database
    # search_results = app.search(query="Who is john")
    # print("\nSearch Results:\n", search_results)

    # # Semantic search over the knowledge graph
    # from r2r.base import VectorStoreQuery

    # node_result = neo4j_kg.vector_query(
    #     VectorStoreQuery(
    #         query_embedding=app.providers.embedding.get_embedding("A person"),
    #     )
    # )
    # print("\nNode Result:", node_result)

    # # Structured query
    # structured_query = """
    # MATCH (p1:person)-[:KNOWS]->(p2:person)
    # RETURN p1.name AS Person1, p2.name AS Person2
    # ORDER BY p1.name
    # LIMIT 10;
    # """
    # print("Executing query:\n", structured_query)
    # structured_result = neo4j_kg.structured_query(structured_query)
    # print("Structured Results:\n", structured_result)
