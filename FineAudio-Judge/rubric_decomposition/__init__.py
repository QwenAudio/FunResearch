"""Structured rubric decomposition components."""

from .models import AudioSceneGraph, RubricQuestion
from .tuples import build_tuples

__all__ = ["AudioSceneGraph", "RubricQuestion", "build_tuples", "decompose_prompt"]


def __getattr__(name: str):
    if name == "decompose_prompt":
        from .decompose import decompose_prompt
        return decompose_prompt
    raise AttributeError(name)
