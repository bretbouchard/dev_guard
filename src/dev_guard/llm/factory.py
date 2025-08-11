"""Central factory for building LLM interfaces used by agents.

Returns a SmartLLM (Ollama-only loop/error-mitigating wrapper) for now.
This allows future expansion to cross-provider smart fallback without touching agents.
"""
from __future__ import annotations

from typing import Any

from .smart import SmartLLM


def get_llm_interface(llm_config: Any) -> SmartLLM:
    """Return the LLM interface to be used by agents.

    Currently returns SmartLLM configured from the provided llm_config.
    """
    return SmartLLM(llm_config)

