import os
import logging
from swarm.base import LLMConfig, KGConfig, KGProvider
from swarm.providers.llms.langchain.langchain import LangchainLLM

logging.basicConfig(level=logging.INFO)

# Define a mock version of KGProvider to avoid the abstract class error
class MockKGProvider(KGProvider):
    def __init__(self, config, llm_agent, neo4j_provider):
        super().__init__(config=config, llm_agent=llm_agent, neo4j_provider=neo4j_provider)

    @property
    def client(self):
        return None

    def delete(self, subj: str, rel: str, obj: str) -> None:
        logging.info(f"Mock delete: {subj}, {rel}, {obj}")

    def get(self, subj: str):
        logging.info(f"Mock get for subject: {subj}")
        return [["John", "works at", "ACME Corp"]]

    def get_rel_map(self, subjs=None, depth=2, limit=30):
        logging.info(f"Mock get_rel_map for subjects: {subjs}")
        return {"John": [["John", "works at", "ACME Corp"]]}

    def get_schema(self, refresh=False):
        logging.info("Mock get schema.")
        return "Mock schema"

    def structured_query(self, query, param_map=None):
        logging.info(f"Mock structured query: {query} with params {param_map}")
        return [{"output": "Mock result"}]

    def upsert_nodes(self, nodes):
        logging.info(f"Mock upsert nodes: {nodes}")

    def upsert_relations(self, relations):
        logging.info(f"Mock upsert relations: {relations}")

    def vector_query(self, query, **kwargs):
        logging.info(f"Mock vector query: {query}")
        return [], []

# Ensure necessary environment variables are set
os.environ["OPENAI_API_KEY"] = "openai-api-key"
os.environ["NEO4J_USER"] = "neo4j_username"
os.environ["NEO4J_PASSWORD"] = "neo4j_password"
os.environ["NEO4J_URL"] = "bolt://localhost:7687"
os.environ["NEO4J_DATABASE"] = "neo4j"

# Initialize LLM Agent (LangchainLLM)
llm_config = LLMConfig(provider="openai")
llm_agent = LangchainLLM(llm_config)

# Initialize KG Config
kg_config = KGConfig(provider="neo4j")

# Initialize Mock KG Provider
kg_provider = MockKGProvider(config=kg_config, llm_agent=llm_agent, neo4j_provider=None)

def test_extraction(input_text: str):
    """Extract and store relationships from a given input text."""
    print(f"\nTesting with input text: {input_text}\n")
    kg_provider.extract_and_store_relations(input_text)

# Define a list of sample texts to test different scenarios
sample_texts = [
    "John works at ACME Corp and manages a team that includes Sarah and Jake.",
    "Alice is a software engineer at OpenAI, collaborating with Bob on AI projects.",
    "The president announced a new initiative for education reform.",
    "Maria and Carlos are siblings, and they both live in New York.",
    "Tesla launched a new electric car model with autonomous driving capabilities.",
]

# Run tests on all sample texts
for text in sample_texts:
    test_extraction(text)