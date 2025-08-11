"""Smart LLM wrapper that detects loops/errors and mitigates them.

This wrapper keeps the current provider (Ollama by default) and retries with
light prompt shaping to reduce looping. It does NOT switch providers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Sequence

from .provider import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMRole,
    LLMProviderError,
    LLMProviderTimeoutError,
    LLMProviderNotAvailableError,
)

logger = logging.getLogger(__name__)


@dataclass
class SmartLLMConfig:
    """Minimal configuration needed by SmartLLM.

    Accepts a subset of the project's LLMConfig to avoid tight coupling.
    """

    provider: str = "ollama"
    model: str = "gpt-oss:20b"
    base_url: str | None = None
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: float = 60.0
    max_retries: int = 2


class SmartLLM:
    """A smarter LLM interface with loop/error mitigation.

    - Detects repeated/echoed responses across attempts
    - Adjusts system guidance between attempts to break loops
    - Retries on transient provider errors/timeouts (within primary provider)
    - Does not switch providers; meant for local-first operation
    """

    def __init__(self, config: Any):
        # Accept project Config.LLMConfig or a plain dict
        if hasattr(config, "provider"):
            cfg = SmartLLMConfig(
                provider=str(getattr(config, "provider", "ollama")),
                model=str(getattr(config, "model", "gpt-oss:20b")),
                base_url=getattr(config, "base_url", None),
                temperature=float(getattr(config, "temperature", 0.1)),
                max_tokens=int(getattr(config, "max_tokens", 4096)),
                timeout=float(getattr(config, "timeout", 60.0)),
                max_retries=int(getattr(config, "max_retries", 2)),
            )
        elif isinstance(config, dict):
            cfg = SmartLLMConfig(
                provider=str(config.get("provider", "ollama")),
                model=str(config.get("model", "gpt-oss:20b")),
                base_url=config.get("base_url"),
                temperature=float(config.get("temperature", 0.1)),
                max_tokens=int(config.get("max_tokens", 4096)),
                timeout=float(config.get("timeout", 60.0)),
                max_retries=int(config.get("max_retries", 2)),
            )
        else:
            raise TypeError("Unsupported config type for SmartLLM")

        self.config = cfg
        self._provider: LLMProvider = self._build_primary_provider(cfg)
        self._last_responses: list[str] = []

    def _build_primary_provider(self, cfg: SmartLLMConfig) -> LLMProvider:
        # Currently supports only Ollama as primary (per user's preference)
        from .ollama import OllamaClient  # local import to avoid hard dependency

        return OllamaClient(
            {
                "model": cfg.model,
                "base_url": cfg.base_url or "http://localhost:11434",
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
                "timeout": cfg.timeout,
                "max_retries": max(1, cfg.max_retries),
            }
        )

    @staticmethod
    def _to_llm_messages(messages: Sequence[dict[str, Any] | LLMMessage]) -> list[LLMMessage]:
        out: list[LLMMessage] = []
        for m in messages:
            if isinstance(m, LLMMessage):
                out.append(m)
            else:
                role = LLMRole(m.get("role", "user"))
                out.append(LLMMessage(role=role, content=str(m.get("content", ""))))
        return out

    def _is_loop(self, new_text: str) -> bool:
        # Simple heuristics for loop detection
        if not new_text:
            return False
        texts = [t for t in self._last_responses if t]
        if not texts:
            return False
        # 1) Exact repeat of previous response
        if new_text.strip() == texts[-1].strip():
            return True
        # 2) Very short repeated phrase (common loop symptom)
        if len(new_text.strip()) < 40 and new_text.strip() in texts[-1]:
            return True
        # 3) Echoing user prompt directly (if we can infer it)
        # Not robust without full convo history; keep it conservative
        return False

    def _loop_breaker_system(self) -> LLMMessage:
        guidance = (
            "You appear to be repeating yourself. Stop looping. Provide a concise, "
            "direct answer and do not restate the previous text."
        )
        return LLMMessage(role=LLMRole.SYSTEM, content=guidance)

    async def generate(
        self,
        messages: Sequence[dict[str, Any] | LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion with loop/error mitigation.

        Returns the final provider LLMResponse.
        """
        base_msgs = self._to_llm_messages(messages)
        attempts = max(1, self.config.max_retries)
        last_error: Exception | None = None
        last_response: LLMResponse | None = None

        for attempt in range(attempts):
            try:
                # On second/third attempt, inject a loop-breaking system message
                msgs = list(base_msgs)
                if attempt > 0:
                    msgs.insert(0, self._loop_breaker_system())

                resp = await self._provider.chat_completion_with_retry(
                    messages=msgs,
                    model=self.config.model,
                    temperature=temperature if temperature is not None else self.config.temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
                    **kwargs,
                )

                # Track and detect loop
                content = (resp.content or "").strip()
                is_loop = self._is_loop(content)
                self._last_responses.append(content)
                # keep last two only
                self._last_responses = self._last_responses[-2:]

                if is_loop and attempt < attempts - 1:
                    logger.warning("SmartLLM detected potential loop; retrying with guidance")
                    last_response = resp
                    continue

                return resp

            except (LLMProviderTimeoutError, LLMProviderNotAvailableError, LLMProviderError) as e:
                last_error = e
                logger.warning(f"SmartLLM provider error on attempt {attempt+1}: {e}")
                last_response = None
                # Continue to next attempt (still on primary provider)
                continue
            except Exception as e:  # safety net
                last_error = e
                logger.exception("Unexpected SmartLLM error")
                last_response = None
                continue

        # If we get here, all attempts either looped or failed
        if last_response is not None:
            return last_response
        if last_error:
            raise last_error
        # Fallback to a neutral empty response to avoid crashes
        return LLMResponse(
            content="",
            model=self.config.model,
            provider="smart-llm",
            usage=None,
            finish_reason="error",
            response_time=None,
            metadata={"note": "empty due to loop/error"},
            timestamp=datetime.now(UTC),
        )

