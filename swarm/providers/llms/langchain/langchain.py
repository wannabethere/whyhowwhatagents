import logging
import os
import jinja2
import openai
from openai import OpenAI, APIError
from typing import Any, Generator, Union, Optional

from swarm.base.llm_provider import (
    LLMConfig,
    LLMProvider
)
from swarm.models.llm import GenerationConfig

logger = logging.getLogger(__name__)

class LangchainLLM(LLMProvider):
    """Class for extracting entities, rules, and relationships using OpenAI API and Jinja2 templates."""

    def __init__(self, config: LLMConfig, template_dir: Optional[str] = None) -> None:
        super().__init__(config)
        self.api_key = "Dummy key"
        template_dir = template_dir or "prompts"
        self.jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
        logger.info("LangchainLLM initialized with template directory at '%s'.", template_dir)

    def extract(self, prompt_content: str) -> str:
        """
        Sends the rendered prompt to OpenAI's API.
        Args:
            prompt_content (str): The fully rendered prompt content.
        """
        logger.debug("Extracting with prompt: %s", prompt_content)
        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an assistant for extracting relationships from text."},
                    {"role": "user", "content": prompt_content}
                ],
                max_tokens=4096,
                temperature=0.5
            )

            print(response)

            # Access the response content directly from the structured response
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                output = response.choices[0].message.content.strip()
            else:
                logger.error("Response has no content in 'choices'.")
                raise ValueError("Empty response content from OpenAI.")

            logger.debug("LLM Response: %s", output)
            return output
        except APIError as e:
            logger.error("OpenAI API error: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error in LLM extract method: %s", e)
            raise