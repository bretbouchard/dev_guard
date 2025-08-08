"""OpenRouter API client for DevGuard."""

import asyncio
import time
from typing import Dict, List, Optional, Any
import aiohttp
import logging

from .provider import (
    LLMProvider, LLMMessage, LLMResponse, LLMUsage,
    LLMProviderError, LLMProviderNotAvailableError,
    LLMProviderTimeoutError, LLMProviderRateLimitError
)

logger = logging.getLogger(__name__)

# List of preferred free models (OpenRouter and z.ai)
PREFERRED_FREE_MODELS = [
    # OpenRouter free models
    "openai/gpt-3.5-turbo",
    "openai/gpt-3.5-turbo-16k",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    # z.ai free models
    "zhipuai/glm-4",
    "zhipuai/glm-4-air",
    "zhipuai/glm-3-turbo",
    "zhipuai/glm-3-32b",
    "zhipuai/glm-3-6b",
    "zhipuai/glm-4-flash",
    "zhipuai/glm-4-0520",
    "zhipuai/glm-4-airx",
]


class OpenRouterClient(LLMProvider):
    """OpenRouter API client with fallback support."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get(
            "base_url", "https://openrouter.ai/api/v1"
        )
        self.fallback_model = config.get(
            "fallback_model", PREFERRED_FREE_MODELS[0]
        )
        self.site_url = config.get("site_url")
        self.site_name = config.get("site_name", "DevGuard")
        
        if not self.api_key:
            raise LLMProviderError("OpenRouter API key is required")
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "openrouter"
    
    async def is_available(self) -> bool:
        """Check if OpenRouter is available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.warning(f"OpenRouter availability check failed: {e}")
            return False
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
            
        return headers
    
    def _messages_to_openai_format(
        self, messages: List[LLMMessage]
    ) -> List[Dict[str, Any]]:
        """Convert LLMMessage objects to OpenAI API format."""
        return [
            {
                "role": msg.role.value,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {}),
                **({"function_call": msg.function_call} 
                   if msg.function_call else {})
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
        """Generate a chat completion using OpenRouter."""
        start_time = time.time()
        
        # Use provided model or fall back to default
        model_to_use = model or self.model or PREFERRED_FREE_MODELS[0]
        temperature_to_use = (
            temperature if temperature is not None else self.temperature
        )
        max_tokens_to_use = max_tokens or self.max_tokens
        
        # Prepare request payload
        payload = {
            "model": model_to_use,
            "messages": self._messages_to_openai_format(messages),
            "temperature": temperature_to_use,
            "max_tokens": max_tokens_to_use,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response_data = await response.json()
                    
                    if response.status == 429:
                        raise LLMProviderRateLimitError(
                            f"Rate limit exceeded: "
                            f"{response_data.get('error', {}).get('message', 'Unknown error')}"
                        )
                    elif response.status >= 400:
                        error_msg = response_data.get(
                            'error', {}
                        ).get(
                            'message', f'HTTP {response.status}'
                        )
                        
                        # Try fallback model if available
                        if (self.fallback_model and 
                            model_to_use != self.fallback_model):
                            self.logger.warning(
                                f"Model {model_to_use} failed, "
                                f"trying fallback {self.fallback_model}"
                            )
                            return await self.chat_completion(
                                messages=messages,
                                model=self.fallback_model,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                **kwargs
                            )
                        
                        raise LLMProviderError(
                            f"OpenRouter API error: {error_msg}"
                        )
                    
                    # Parse successful response
                    choice = response_data["choices"][0]
                    content = choice["message"]["content"]
                    
                    # Extract usage information if available
                    usage = None
                    if "usage" in response_data:
                        usage_data = response_data["usage"]
                        usage = LLMUsage(
                            prompt_tokens=usage_data.get("prompt_tokens", 0),
                            completion_tokens=usage_data.get(
                                "completion_tokens", 0
                            ),
                            total_tokens=usage_data.get("total_tokens", 0)
                        )
                    
                    response_time = time.time() - start_time
                    
                    return LLMResponse(
                        content=content,
                        model=model_to_use,
                        provider=self.get_provider_name(),
                        usage=usage,
                        finish_reason=choice.get("finish_reason"),
                        response_time=response_time,
                        metadata={
                            "raw_response": response_data,
                            "request_payload": payload
                        }
                    )
                    
        except asyncio.TimeoutError:
            raise LLMProviderTimeoutError(
                f"OpenRouter request timed out after {self.timeout}s"
            )
        except aiohttp.ClientError as e:
            raise LLMProviderNotAvailableError(
                f"OpenRouter connection error: {e}"
            )
        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise
            raise LLMProviderError(f"Unexpected OpenRouter error: {e}")
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data", [])
                    else:
                        self.logger.error(
                            f"Failed to get models: HTTP {response.status}"
                        )
                        return []
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return []
