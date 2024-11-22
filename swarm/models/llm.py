"""Abstractions for the LLM model."""

from typing import TYPE_CHECKING, ClassVar, Optional
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .search import AggregateSearchResult


class GenerationConfig(BaseModel):
    _defaults: ClassVar[dict] = {
        "model": "gpt-4o",
        "temperature": 0.1,
        "top_p": 1.0,
        "top_k": 100,
        "max_tokens_to_sample": 1024,
        "stream": False,
        "functions": None,
        "skip_special_tokens": False,
        "stop_token": None,
        "num_beams": 1,
        "do_sample": True,
        "generate_with_chat": False,
        "add_generation_kwargs": None,
        "api_base": None,
    }

    model: str = Field(
        default_factory=lambda: GenerationConfig._defaults["model"]
    )
    temperature: float = Field(
        default_factory=lambda: GenerationConfig._defaults["temperature"]
    )
    top_p: float = Field(
        default_factory=lambda: GenerationConfig._defaults["top_p"]
    )
    top_k: int = Field(
        default_factory=lambda: GenerationConfig._defaults["top_k"]
    )
    max_tokens_to_sample: int = Field(
        default_factory=lambda: GenerationConfig._defaults[
            "max_tokens_to_sample"
        ]
    )
    stream: bool = Field(
        default_factory=lambda: GenerationConfig._defaults["stream"]
    )
    functions: Optional[list[dict]] = Field(
        default_factory=lambda: GenerationConfig._defaults["functions"]
    )
    skip_special_tokens: bool = Field(
        default_factory=lambda: GenerationConfig._defaults[
            "skip_special_tokens"
        ]
    )
    stop_token: Optional[str] = Field(
        default_factory=lambda: GenerationConfig._defaults["stop_token"]
    )
    num_beams: int = Field(
        default_factory=lambda: GenerationConfig._defaults["num_beams"]
    )
    do_sample: bool = Field(
        default_factory=lambda: GenerationConfig._defaults["do_sample"]
    )
    generate_with_chat: bool = Field(
        default_factory=lambda: GenerationConfig._defaults[
            "generate_with_chat"
        ]
    )
    add_generation_kwargs: Optional[dict] = Field(
        default_factory=lambda: GenerationConfig._defaults[
            "add_generation_kwargs"
        ]
    )
    api_base: Optional[str] = Field(
        default_factory=lambda: GenerationConfig._defaults["api_base"]
    )

    @classmethod
    def set_default(cls, **kwargs):
        for key, value in kwargs.items():
            if key in cls._defaults:
                cls._defaults[key] = value
            else:
                raise AttributeError(
                    f"No default attribute '{key}' in GenerationConfig"
                )

    def __init__(self, **data):
        model = data.pop("model", None)
        if model is not None:
            super().__init__(model=model, **data)
        else:
            super().__init__(**data)
