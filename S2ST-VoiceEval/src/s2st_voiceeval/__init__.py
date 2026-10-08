"""Voice-preservation evaluation with source-language references."""

__version__ = "0.1.0"
__all__ = ["VoiceEvaluator", "__version__"]


def __getattr__(name):
    # Keep CLI help and metric utilities usable without loading the model stack.
    if name == "VoiceEvaluator":
        from .evaluator import VoiceEvaluator
        return VoiceEvaluator
    raise AttributeError(name)
