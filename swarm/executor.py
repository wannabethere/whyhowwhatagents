import asyncio
import ast
import json
import os
import uuid
from typing import Optional, Union
from fastapi import UploadFile
from swarm.assembly.config import SwarmConfig
from swarm.swarm import Swarm, SwarmEngine
from swarm.base import generate_id_from_label
    
        
class SwarmExecutor:
    """A demo class for the R2R library."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_name: Optional[str] = "default",
    ):
        if config_path and config_name:
            raise Exception("Cannot specify both config_path and config_name")

        
        
        self.config = (
                SwarmConfig.from_json(config_path)
                if config_path
                else SwarmConfig.from_json(
                    SwarmConfig.CONFIG_OPTIONS[config_name or "default"]
                )
            )
        swarm = Swarm(config=self.config)
        self.engine:SwarmEngine = swarm.getengine()
        


    def _parse_metadata_string(metadata_string: str) -> list[dict]:
        """
        Convert a string representation of metadata into a list of dictionaries.

        The input string can be in one of two formats:
        1. JSON array of objects: '[{"key": "value"}, {"key2": "value2"}]'
        2. Python-like list of dictionaries: "[{'key': 'value'}, {'key2': 'value2'}]"

        Args:
        metadata_string (str): The string representation of metadata.

        Returns:
        list[dict]: A list of dictionaries representing the metadata.

        Raises:
        ValueError: If the string cannot be parsed into a list of dictionaries.
        """
        if not metadata_string:
            return []

        try:
            # First, try to parse as JSON
            return json.loads(metadata_string)
        except json.JSONDecodeError as e:
            try:
                # If JSON parsing fails, try to evaluate as a Python literal
                result = ast.literal_eval(metadata_string)
                if not isinstance(result, list) or not all(
                    isinstance(item, dict) for item in result
                ):
                    raise ValueError(
                        "The string does not represent a list of dictionaries"
                    ) from e
                return result
            except (ValueError, SyntaxError) as exc:
                raise ValueError(
                    "Unable to parse the metadata string. "
                    "Please ensure it's a valid JSON array or Python list of dictionaries."
                ) from exc
   
    def ingest_files(
        self,
        file_paths: list[str],
        metadatas: Optional[list[dict]] = None,
        document_ids: Optional[list[Union[uuid.UUID, str]]] = None,
        versions: Optional[list[str]] = None,
    ):
        if isinstance(file_paths, str):
            file_paths = list(file_paths.split(","))
        if isinstance(metadatas, str):
            metadatas = self._parse_metadata_string(metadatas)
        if isinstance(document_ids, str):
            document_ids = list(document_ids.split(","))
        if isinstance(versions, str):
            versions = list(versions.split(","))

        all_file_paths = []
        for path in file_paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    all_file_paths.extend(
                        os.path.join(root, file) for file in files
                    )
            else:
                all_file_paths.append(path)

        if not document_ids:
            document_ids = [
                generate_id_from_label(os.path.basename(file_path))
                for file_path in all_file_paths
            ]

        files = [
            UploadFile(
                filename=os.path.basename(file_path),
                file=open(file_path, "rb"),
            )
            for file_path in all_file_paths
        ]

        for file in files:
            file.file.seek(0, 2)
            file.size = file.file.tell()
            file.file.seek(0)

        result = self.engine.aingest_files(
                    files=files,
                    document_ids=document_ids,
                    metadatas=metadatas,
                    versions=versions,
                )
        return result