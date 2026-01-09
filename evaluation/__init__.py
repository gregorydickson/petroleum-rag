"""Evaluation metrics and benchmark utilities."""

from evaluation.evaluator import Evaluator
from evaluation.metrics import MetricsCalculator
from evaluation.vertex_evaluator import VertexEvaluator


def get_evaluator(provider: str | None = None):
    """Factory function to get the appropriate evaluator based on provider.

    Args:
        provider: LLM provider ('anthropic' or 'vertex').
                 If None, uses settings.eval_llm_provider.

    Returns:
        Evaluator or VertexEvaluator instance

    Raises:
        ValueError: If provider is not supported
    """
    from config import settings

    provider = provider or settings.eval_llm_provider

    if provider == "anthropic":
        return Evaluator()
    elif provider == "vertex":
        return VertexEvaluator()
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: 'anthropic', 'vertex'"
        )


__all__ = [
    "Evaluator",
    "VertexEvaluator",
    "MetricsCalculator",
    "get_evaluator",
]
