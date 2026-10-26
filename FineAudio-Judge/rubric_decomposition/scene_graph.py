"""Event-centered graph normalization and deterministic tuple compilation."""

from .tuples import build_tuples
from .validation import canonicalize_scene_graph

__all__ = ["build_tuples", "canonicalize_scene_graph"]
