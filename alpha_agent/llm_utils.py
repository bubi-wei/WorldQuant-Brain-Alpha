"""LLM capability helpers used across agents."""

from __future__ import annotations


def supports_json_response_format(model: str | None) -> bool:
    """Return whether the model is expected to support response_format=json_object.

    We intentionally disable this for DeepSeek models (especially deepseek-reasoner),
    which reject the response_format parameter in LiteLLM.
    """
    model_name = (model or "").strip().lower()
    if not model_name:
        return True
    if "deepseek" in model_name:
        return False
    return True
