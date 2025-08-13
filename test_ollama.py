#!/usr/bin/env python3
"""Simple test script to verify Ollama connection."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dev_guard.llm.ollama import OllamaClient
from dev_guard.llm.provider import LLMMessage, LLMRole

async def test_ollama():
    """Test Ollama connection and simple chat."""
    print("Testing Ollama connection...")
    
    client = OllamaClient({
        "base_url": "http://localhost:11434",
        "model": "gpt-oss:20b",
        "temperature": 0.1,
        "timeout": 120.0
    })
    
    # Check availability
    print("Checking if Ollama is available...")
    if not await client.is_available():
        print("❌ Ollama is not available")
        return False
    
    print("✅ Ollama is available")
    
    # Test simple chat
    print("Testing simple chat completion...")
    messages = [
        LLMMessage(role=LLMRole.USER, content="Hello! Please respond with just 'Hello, DevGuard is working!' and nothing else.")
    ]
    
    try:
        response = await client.chat_completion(messages)
        print(f"✅ Chat completion successful!")
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        return True
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ollama())
    sys.exit(0 if success else 1)