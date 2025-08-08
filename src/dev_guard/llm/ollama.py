"""Ollama local LLM client for DevGuard."""

import asyncio
import time
from typing import Dict, List, Optional, Any
import aiohttp
import logging

from .provider import (
    LLMProvider, LLMMessage, LLMResponse, LLMUsage,
    LLMProviderError, LLMProviderNotAvailableError,
    LLMProviderTimeoutError
)

logger = logging.getLogger(__name__)


class OllamaClient(LLMProvider):
    """Ollama local LLM client."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        # Default to a common free model if none specified
        if not self.model:
            self.model = "llama2"
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "ollama"
    
    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.warning(f"Ollama availability check failed: {e}")
            return False
    
    def _messages_to_ollama_format(
        self, messages: List[LLMMessage]
    ) -> List[Dict[str, Any]]:
        """Convert LLMMessage objects to Ollama API format."""
        return [
            {
                "role": msg.role.value,
                "content": msg.content
            }
            for msg in messages
        ]
    
    async def chat_completion(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a chat completion using Ollama."""
        start_time = time.time()
        
        # Use provided model or fall back to default
        model_to_use = model or self.model
        temperature_to_use = (
            temperature if temperature is not None else self.temperature
        )
        
        # Prepare request payload for Ollama chat API
        payload = {
            "model": model_to_use,
            "messages": self._messages_to_ollama_format(messages),
            "stream": False,
            "options": {
                "temperature": temperature_to_use,
            }
        }
        
        # Add max_tokens if specified (Ollama uses num_predict)
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        # Add any additional options
        if kwargs:
            payload["options"].update(kwargs)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMProviderError(
                            f"Ollama API error {response.status}: {error_text}"
                        )
                    
                    response_data = await response.json()
                    
                    # Extract response content
                    message = response_data.get("message", {})
                    content = message.get("content", "")
                    
                    # Calculate response time
                    response_time = time.time() - start_time
                    
                    # Ollama doesn't provide detailed usage info,
                    # so we create basic usage data
                    usage = None
                    if "eval_count" in response_data or "prompt_eval_count" in response_data:
                        usage = LLMUsage(
                            prompt_tokens=response_data.get("prompt_eval_count", 0),
                            completion_tokens=response_data.get("eval_count", 0),
                            total_tokens=(
                                response_data.get("prompt_eval_count", 0) +
                                response_data.get("eval_count", 0)
                            )
                        )
                    
                    return LLMResponse(
                        content=content,
                        model=model_to_use,
                        provider=self.get_provider_name(),
                        usage=usage,
                        finish_reason=response_data.get("done_reason"),
                        response_time=response_time,
                        metadata={
                            "raw_response": response_data,
                            "request_payload": payload
                        }
                    )
                    
        except asyncio.TimeoutError:
            raise LLMProviderTimeoutError(
                f"Ollama request timed out after {self.timeout}s"
            )
        except aiohttp.ClientError as e:
            raise LLMProviderNotAvailableError(
                f"Ollama connection error: {e}"
            )
        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise
            raise LLMProviderError(f"Unexpected Ollama error: {e}")
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("models", [])
                    else:
                        self.logger.error(
                            f"Failed to get models: HTTP {response.status}"
                        )
                        return []
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama library."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name},
                    timeout=aiohttp.ClientTimeout(total=300.0)  # 5 min timeout
                ) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {e}")
            return False
