"""Base LLM provider interface and response models."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LLMRole(str, Enum):
    """Message roles for LLM conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class LLMMessage:
    """A single message in an LLM conversation."""
    role: LLMRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


@dataclass
class LLMUsage:
    """Token usage information from LLM response."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    provider: str
    usage: Optional[LLMUsage] = None
    finish_reason: Optional[str] = None
    response_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get("model", "")
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 4096)
        self.timeout = config.get("timeout", 30.0)
        self.max_retries = config.get("max_retries", 3)
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a chat completion."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
        pass
    
    async def chat_completion_with_retry(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a chat completion with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await self.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed, "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(
                        f"All {self.max_retries} attempts failed: {e}"
                    )
        
        if last_exception:
            raise last_exception
        
        raise Exception("Unknown error during chat completion")


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMProviderNotAvailableError(LLMProviderError):
    """Exception raised when LLM provider is not available."""
    pass


class LLMProviderTimeoutError(LLMProviderError):
    """Exception raised when LLM provider times out."""
    pass


class LLMProviderRateLimitError(LLMProviderError):
    """Exception raised when LLM provider rate limit is exceeded."""
    pass
