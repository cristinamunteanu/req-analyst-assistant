"""
LLM Provider configuration and utilities for the model-based extractor.

Supports multiple LLM providers including OpenAI, Anthropic, and local models.
"""

import os
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self._client = None
        
    def _get_client(self):
        """Lazy load OpenAI client"""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI library not installed. Run: pip install openai")
        return self._client
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API"""
        client = self._get_client()
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a requirements extraction assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get('temperature', 0.1),
                max_tokens=kwargs.get('max_tokens', 2000)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model
        self._client = None
        
    def _get_client(self):
        """Lazy load Anthropic client"""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic library not installed. Run: pip install anthropic")
        return self._client
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API"""
        client = self._get_client()
        
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.1),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

class OllamaProvider(LLMProvider):
    """Local Ollama LLM provider"""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._client = None
        
    def _get_client(self):
        """Lazy load Ollama client"""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise ImportError("Ollama library not installed. Run: pip install ollama")
        return self._client
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama API"""
        client = self._get_client()
        
        try:
            response = client.generate(
                model=self.model,
                prompt=prompt,
                stream=False
            )
            return response['response']
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

class LLMConfig:
    """Configuration manager for LLM providers"""
    
    SUPPORTED_PROVIDERS = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider, 
        'ollama': OllamaProvider
    }
    
    @classmethod
    def create_provider(cls, provider_name: str, **kwargs) -> LLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            provider_name: Name of the provider ('openai', 'anthropic', 'ollama')
            **kwargs: Provider-specific configuration
            
        Returns:
            LLM provider instance
        """
        if provider_name not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider_name}. Supported: {list(cls.SUPPORTED_PROVIDERS.keys())}")
        
        provider_class = cls.SUPPORTED_PROVIDERS[provider_name]
        return provider_class(**kwargs)
    
    @classmethod
    def auto_detect_provider(cls) -> Optional[LLMProvider]:
        """
        Auto-detect available LLM provider based on environment.
        
        Returns:
            LLM provider instance or None if none available
        """
        # Check for API keys in environment
        if os.getenv('OPENAI_API_KEY'):
            logger.info("OpenAI API key found, using OpenAI provider")
            return cls.create_provider('openai')
        
        if os.getenv('ANTHROPIC_API_KEY'):
            logger.info("Anthropic API key found, using Anthropic provider")
            return cls.create_provider('anthropic')
        
        # Check if Ollama is running locally
        try:
            import requests
            response = requests.get('http://localhost:11434/api/version', timeout=2)
            if response.status_code == 200:
                logger.info("Ollama detected, using Ollama provider")
                return cls.create_provider('ollama')
        except:
            pass
        
        logger.warning("No LLM provider detected. Extraction will use fallback methods.")
        return None

def get_default_provider() -> Optional[LLMProvider]:
    """Get the default LLM provider based on configuration"""
    return LLMConfig.auto_detect_provider()