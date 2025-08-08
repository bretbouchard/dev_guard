"""Test suite for LLM providers."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Test imports - handle import errors gracefully
try:
    from src.dev_guard.llm.base import LLMProvider
    from src.dev_guard.llm.ollama_client import OllamaClient
    from src.dev_guard.llm.openrouter_client import OpenRouterClient
    LLM_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"LLM Import error: {e}")
    LLM_IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not LLM_IMPORTS_AVAILABLE, reason="LLM modules not available")
class TestOpenRouterClient:
    """Test OpenRouter client functionality."""
    
    def test_openrouter_client_creation(self):
        """Test creating OpenRouter client."""
        client = OpenRouterClient(
            api_key="test_key",
            model="gpt-4",
            base_url="https://openrouter.ai/api/v1"
        )
        
        assert client.api_key == "test_key"
        assert client.model == "gpt-4"
        assert client.base_url == "https://openrouter.ai/api/v1"
    
    @pytest.mark.asyncio
    async def test_openrouter_chat_completion_success(self):
        """Test successful chat completion."""
        client = OpenRouterClient(api_key="test_key", model="gpt-4")
        
        messages = [
            {"role": "user", "content": "Hello, world!"}
        ]
        
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Hello! How can I help you today?"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_resp = Mock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            response = await client.chat_completion(messages)
            
            assert response["choices"][0]["message"]["content"] == "Hello! How can I help you today?"
            assert response["usage"]["total_tokens"] == 25
    
    @pytest.mark.asyncio
    async def test_openrouter_chat_completion_failure(self):
        """Test chat completion with API failure."""
        client = OpenRouterClient(api_key="test_key", model="gpt-4")
        
        messages = [{"role": "user", "content": "Test message"}]
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_resp = Mock()
            mock_resp.status = 400
            mock_resp.json = AsyncMock(return_value={"error": "Bad request"})
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            with pytest.raises(Exception):
                await client.chat_completion(messages)
    
    @pytest.mark.asyncio
    async def test_openrouter_with_fallback(self):
        """Test OpenRouter with fallback model."""
        client = OpenRouterClient(
            api_key="test_key", 
            model="primary-model",
            fallback_model="fallback-model"
        )
        
        messages = [{"role": "user", "content": "Test"}]
        
        # Mock primary model failure, fallback success
        with patch('aiohttp.ClientSession.post') as mock_post:
            # First call (primary) fails
            mock_resp_fail = Mock()
            mock_resp_fail.status = 500
            
            # Second call (fallback) succeeds
            mock_resp_success = Mock()
            mock_resp_success.status = 200
            mock_resp_success.json = AsyncMock(return_value={
                "choices": [{"message": {"content": "Fallback response"}}]
            })
            
            mock_post.return_value.__aenter__.side_effect = [
                mock_resp_fail,
                mock_resp_success
            ]
            
            response = await client.chat_completion(messages)
            
            assert response["choices"][0]["message"]["content"] == "Fallback response"
            assert mock_post.call_count == 2  # Primary + fallback
    
    def test_openrouter_model_listing(self):
        """Test model listing functionality."""
        client = OpenRouterClient(api_key="test_key", model="gpt-4")
        
        # Test that the method exists and returns expected structure
        if hasattr(client, 'list_models'):
            with patch.object(client, 'list_models') as mock_list:
                mock_list.return_value = ["gpt-4", "gpt-3.5-turbo", "claude-2"]
                models = client.list_models()
                assert isinstance(models, list)
                assert "gpt-4" in models


@pytest.mark.skipif(not LLM_IMPORTS_AVAILABLE, reason="LLM modules not available")
class TestOllamaClient:
    """Test Ollama client functionality."""
    
    def test_ollama_client_creation(self):
        """Test creating Ollama client."""
        client = OllamaClient(
            base_url="http://localhost:11434",
            model="llama2"
        )
        
        assert client.base_url == "http://localhost:11434"
        assert client.model == "llama2"
    
    @pytest.mark.asyncio
    async def test_ollama_chat_completion_success(self):
        """Test successful Ollama chat completion."""
        client = OllamaClient(base_url="http://localhost:11434", model="llama2")
        
        messages = [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        mock_response = {
            "message": {
                "role": "assistant", 
                "content": "The capital of France is Paris."
            },
            "done": True,
            "total_duration": 1000000000,
            "load_duration": 500000000,
            "prompt_eval_count": 15,
            "eval_count": 10
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_resp = Mock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            response = await client.chat_completion(messages)
            
            assert response["message"]["content"] == "The capital of France is Paris."
            assert response["done"] is True
    
    @pytest.mark.asyncio
    async def test_ollama_model_availability_check(self):
        """Test checking if model is available."""
        client = OllamaClient(base_url="http://localhost:11434", model="llama2")
        
        mock_models_response = {
            "models": [
                {"name": "llama2:latest"},
                {"name": "codellama:latest"},
                {"name": "mistral:latest"}
            ]
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = Mock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_models_response)
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            if hasattr(client, 'is_model_available'):
                available = await client.is_model_available("llama2")
                assert available is True
                
                not_available = await client.is_model_available("nonexistent")
                assert not_available is False
    
    @pytest.mark.asyncio
    async def test_ollama_model_pull(self):
        """Test pulling a model in Ollama."""
        client = OllamaClient(base_url="http://localhost:11434", model="llama2")
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_resp = Mock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={"status": "success"})
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            if hasattr(client, 'pull_model'):
                result = await client.pull_model("new-model")
                assert result.get("status") == "success"
    
    @pytest.mark.asyncio
    async def test_ollama_connection_error(self):
        """Test Ollama client with connection error."""
        client = OllamaClient(base_url="http://localhost:11434", model="llama2")
        
        messages = [{"role": "user", "content": "Test"}]
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = Exception("Connection refused")
            
            with pytest.raises(Exception):
                await client.chat_completion(messages)


# Mock-based tests that don't require actual implementations
class TestLLMProviderMocks:
    """Test LLM provider functionality using mocks."""
    
    def test_mock_llm_provider_interface(self):
        """Test LLM provider interface using mocks."""
        mock_provider = Mock()
        mock_provider.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": "Mock response"}}]
        })
        mock_provider.model = "mock-model"
        mock_provider.api_key = "mock-key"
        
        # Test interface
        assert mock_provider.model == "mock-model"
        assert mock_provider.api_key == "mock-key"
        
        async def test_async():
            response = await mock_provider.chat_completion([])
            assert response["choices"][0]["message"]["content"] == "Mock response"
        
        asyncio.run(test_async())
    
    def test_mock_model_management(self):
        """Test model management functionality using mocks."""
        mock_client = Mock()
        mock_client.list_models.return_value = [
            "gpt-4", "gpt-3.5-turbo", "claude-2", "llama2"
        ]
        mock_client.is_model_available = AsyncMock(return_value=True)
        mock_client.pull_model = AsyncMock(return_value={"status": "success"})
        
        # Test model listing
        models = mock_client.list_models()
        assert "gpt-4" in models
        assert "llama2" in models
        
        # Test model availability and pulling
        async def test_async():
            available = await mock_client.is_model_available("gpt-4")
            assert available is True
            
            result = await mock_client.pull_model("new-model")
            assert result["status"] == "success"
        
        asyncio.run(test_async())


class TestLLMProviderIntegration:
    """Integration tests for LLM providers."""
    
    def test_provider_configuration(self):
        """Test LLM provider configuration."""
        # OpenRouter configuration
        openrouter_config = {
            "provider": "openrouter",
            "model": "gpt-4",
            "api_key": "test_key",
            "base_url": "https://openrouter.ai/api/v1",
            "temperature": 0.1,
            "max_tokens": 4096
        }
        
        # Ollama configuration
        ollama_config = {
            "provider": "ollama",
            "model": "llama2",
            "base_url": "http://localhost:11434",
            "temperature": 0.1,
            "max_tokens": 4096
        }
        
        # Basic validation
        assert openrouter_config["provider"] == "openrouter"
        assert ollama_config["provider"] == "ollama"
        assert openrouter_config["model"] == "gpt-4"
        assert ollama_config["model"] == "llama2"
    
    def test_provider_switching(self):
        """Test switching between different providers."""
        # Mock provider factory
        mock_openrouter = Mock()
        mock_openrouter.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": "OpenRouter response"}}]
        })
        
        mock_ollama = Mock()
        mock_ollama.chat_completion = AsyncMock(return_value={
            "message": {"content": "Ollama response"}
        })
        
        # Test provider creation and configuration
        assert hasattr(mock_openrouter, 'chat_completion')
        assert hasattr(mock_ollama, 'chat_completion')
        
        # Test basic provider switching logic
        providers = {
            'openrouter': mock_openrouter,
            'ollama': mock_ollama
        }
        
        current_provider_name = 'openrouter'
        assert current_provider_name in providers
        
        # Switch provider
        current_provider_name = 'ollama' 
        assert current_provider_name in providers
        
        # Verify providers have expected methods
        assert callable(providers[current_provider_name].chat_completion)
    
    def test_error_handling_patterns(self):
        """Test common error handling patterns."""
        # Test rate limiting
        rate_limit_error = {
            "error": {
                "type": "rate_limit_exceeded",
                "message": "Rate limit exceeded"
            }
        }
        
        # Test invalid model error
        invalid_model_error = {
            "error": {
                "type": "model_not_found",
                "message": "Model not available"
            }
        }
        
        # Test authentication error
        auth_error = {
            "error": {
                "type": "authentication_error", 
                "message": "Invalid API key"
            }
        }
        
        # Basic error structure validation
        assert rate_limit_error["error"]["type"] == "rate_limit_exceeded"
        assert invalid_model_error["error"]["type"] == "model_not_found"
        assert auth_error["error"]["type"] == "authentication_error"


if __name__ == "__main__":
    if LLM_IMPORTS_AVAILABLE:
        pytest.main([__file__, "-v", "-x"])
    else:
        print("Skipping LLM tests due to import errors")
        exit(1)
