"""Base classes for knowledge graph providers."""
import ast
import json
import os
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import jinja2
from .prompt_provider import PromptProvider
from .base_utils import EntityType, Relation
from swarm.models.llama_index_abstractions import EntityNode, LabelledNode
from swarm.models.llama_index_abstractions import Relation as LlamaRelation
from swarm.models.llama_index_abstractions import VectorStoreQuery
from swarm.models.llm import GenerationConfig
from .base_provider import ProviderConfig

logger = logging.getLogger(__name__)


class KGConfig(ProviderConfig):
    """Configuration for Knowledge Graph with prompts and rules."""
    provider: Optional[str] = None
    kg_extraction_prompt: str = "kg_extraction_prompt.jinja2"
    template_dir: str = "swarm/prompts"  # Specify the directory where prompts are stored

    def validate(self) -> None:
        supported_providers = ["neo4j", None]
        if self.provider not in supported_providers:
            raise ValueError(f"Provider '{self.provider}' is not supported. Supported providers: {supported_providers}")

    @property
    def jinja_env(self) -> jinja2.Environment:
        """Lazy-loaded Jinja2 environment for loading templates."""
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir)
        )

    def render_prompt(self, template_name: str, variables: dict) -> str:
        """Renders a prompt template with the provided variables."""
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(variables)
        except jinja2.TemplateNotFound:
            raise FileNotFoundError(f"Template '{template_name}' not found in '{self.template_dir}' directory.")

class KGProvider(ABC):
    def __init__(self, config: KGConfig, llm_agent: Any) -> None:
        self.config = config
        self.llm_agent = llm_agent
        logger.info("KGProvider initialized with config: %s", config)

    def _chunk_document(self, text: str, chunk_size: int = 500) -> List[str]:
        """Splits the input text into chunks of a specified size."""
        words = text.split()
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        # Log each chunk to ensure they are created as expected
        for idx, chunk in enumerate(chunks):
            logger.info(f"Chunk {idx + 1}/{len(chunks)}: {chunk[:200]}...")  # Log the first 200 characters of each chunk
        return chunks

    def extract_and_store_relations(self, text: str) -> dict:
        chunks = self._chunk_document(text)
        all_entities, all_rules, all_relationships = [], [], []

        for idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {idx + 1}/{len(chunks)}")
            extractions = self._extract_with_unified_prompt(chunk)

            # Log the extracted data structure for each chunk
            logger.info(f"Extracted data for chunk {idx + 1}: {extractions}")

            if isinstance(extractions, dict):
                all_entities.extend(extractions.get("entities", []))
                all_rules.extend(extractions.get("rules", []))
                all_relationships.extend(extractions.get("relationships", []))
            else:
                logger.warning(f"No valid extractions for chunk {idx + 1}")

        # Return all data in one dictionary
        full_extraction = {
            "entities": all_entities,
            "rules": all_rules,
            "relationships": all_relationships,
        }

        # Export relationships to CSV
        self._export_to_csv(all_relationships)
        return full_extraction

    def _extract_with_unified_prompt(self, chunk: str) -> List[dict]:
        """Extracts entities, rules, and relationships using a unified prompt with the LLM."""
        if not chunk.strip():
            logger.warning("Skipping empty chunk.")
            return []

        # Log the chunk content before rendering
        logger.info(f"Rendering prompt for chunk: {chunk[:200]}...")

        # Render the prompt with variables
        prompt_content = self.config.render_prompt(self.config.kg_extraction_prompt, {"input_text": chunk})
        logger.info(f"Rendered prompt content: {prompt_content}")  # Log the full prompt content

        # Pass rendered prompt content to the LLM agent
        response = self.llm_agent.extract(prompt_content)

        # Parse the response to extract entities, rules, and relationships
        return self._parse_llm_response(response)

    def _parse_llm_response(self, response: str) -> dict:
        """Parses LLM response to extract entities, rules, and relationships."""
        if not response:
            logger.error("Received empty response from LLM; cannot parse.")
            return {"entities": [], "rules": [], "relationships": []}

        try:
            # Adjusted regex to ensure we capture the entire relationships section
            entities_match = re.search(r"- \*\*Entities\*\*:\n\s*([\s\S]*?)(?=\n- \*\*Rules\*\*|$)", response)
            rules_match = re.search(r"- \*\*Rules\*\*:\n\s*([\s\S]*?)(?=\n- \*\*Relationships\*\*|$)", response)
            relationships_match = re.search(r"- \*\*Relationships\*\*:\s*\n?([\s\S]*?)(?=\n\s*-|\Z)", response)

            def parse_items(section_text):
                """Parses a section containing list items in JSON format."""
                items = []
                # Split each item by lines and remove unnecessary bullet points or whitespace
                for line in section_text.strip().splitlines():
                    # Remove leading bullet points and whitespace, then parse as JSON
                    line = line.strip().lstrip('- ')
                    if line:
                        try:
                            # Try parsing each line as a valid JSON object
                            items.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON parsing error on line '{line}': {e}")
                return items

            # Extract entities, rules, and relationships based on the regex matches
            entities = parse_items(entities_match.group(1)) if entities_match else []
            rules = parse_items(rules_match.group(1)) if rules_match else []
            relationships = parse_items(relationships_match.group(1)) if relationships_match else []

            # Ensure the relationships are correctly parsed and returned
            parsed_response = {
                "entities": entities,
                "rules": rules,
                "relationships": relationships
            }

            logger.info("Parsed LLM response: %s", parsed_response)
            return parsed_response

        except Exception as e:
            logger.error(f"Failed to parse LLM response due to error: {e}")
            return {"entities": [], "rules": [], "relationships": []}

    def _export_to_csv(self, relationships: List[dict]) -> None:
        """Exports the relationships information to a CSV file."""
        import csv
        with open("extracted_relations.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Subject", "Predicate", "Object", "Properties"])

            for relation in relationships:
                writer.writerow([
                    relation.get("subject"),
                    relation.get("predicate"),
                    relation.get("object"),
                    json.dumps(relation.get("properties", {}))
                ])
        logger.info("Extraction results exported to 'extracted_relations.csv'")