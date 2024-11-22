import json
import logging
import os
import os.path
from typing import Any, Optional

from swarm.base import  PromptProvider
from swarm.models.prompt import Prompt
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

THIS_DIR = os.path.abspath(os.path.dirname(__file__))




class SwarmPromptProvider(PromptProvider):
    def __init__(self, file_path: Optional[str] = None):
        self.prompts: dict[str, Prompt] = {}
        #self.load_prompt(prompt="file://"+file_path)
        self._load_prompts_from_jsonl(file_path=file_path)
        #TODO: ADD JSONL Support
    

    def load_prompt(self, prompt: str) -> str:
        """
        prompt is either in the format 'builtin://' or 'file://' or a regular string
        builtins are loaded as a file from this directory
        files are loaded from the file system normally
        regular strings are returned as is (as literal strings)
        """
        if prompt.startswith("builtin://"):
            path = os.path.join(THIS_DIR, prompt[len("builtin://"):])
        elif prompt.startswith("file://"):
            path = prompt[len("file://"):]
        else:
            return prompt
        
        return open(path).read(), path

    def load_and_render_prompt(self, prompt: str, context: dict = {}) -> str:
        """
        prompt is in the format 'builtin://' or 'file://' or a regular string
        see load_prompt() for details

        context is a dictionary of variables to be passed to the jinja2 template
        """
        prompt_as_str, key = self.load_prompt(prompt)

        env = Environment(
            loader=FileSystemLoader(THIS_DIR),
        )
        template = env.from_string(prompt_as_str)
        self.prompts[key] = template.render(**context)


    def _load_prompts_from_jsonl(self, file_path: Optional[str] = None):
        if not file_path:
            file_path = os.path.join(
                os.path.dirname(__file__), "defaults.jsonl"
            )
        try:
            with open(file_path, "r") as file:
                for line in file:
                    if line.strip():
                        data = json.loads(line)
                        self.add_prompt(
                            data["name"],
                            data["template"],
                            data.get("input_types", {}),
                        )
        except json.JSONDecodeError as e:
            error_msg = f"Error loading prompts from JSONL file: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def add_prompt(
        self, name: str, template: str, input_types: dict[str, str]
    ) -> None:
        if name in self.prompts:
            raise ValueError(f"Prompt '{name}' already exists.")
        self.prompts[name] = Prompt(
            name=name, template=template, input_types=input_types
        )

    def get_prompt(
        self, prompt_name: str, inputs: Optional[dict[str, Any]] = None
    ) -> str:
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found.")
        prompt = self.prompts[prompt_name]
        if inputs is None:
            return prompt.template
        return prompt.format_prompt(inputs)
    
    def get_lang_chain_prompt_template(self, prompt_name: str, inputs: Optional[dict[str, Any]] = None):
        """
        ChatMessagePromptTemplate.from_template(
    role="Jedi", template=prompt
)
        """
        pass

    def update_prompt(
        self,
        name: str,
        template: Optional[str] = None,
        input_types: Optional[dict[str, str]] = None,
    ) -> None:
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found.")
        if template:
            self.prompts[name].template = template
        if input_types:
            self.prompts[name].input_types = input_types

    def get_all_prompts(self) -> dict[str, Prompt]:
        return self.prompts