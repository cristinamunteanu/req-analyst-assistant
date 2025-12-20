import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from analysis.llm_providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    LLMConfig,
    get_default_provider
)


class TestLLMProvider:
    """Test the abstract LLMProvider base class"""
    
    def test_abstract_base_class(self):
        """Test that LLMProvider cannot be instantiated directly"""
        with pytest.raises(TypeError):
            LLMProvider()


class TestOpenAIProvider:
    """Test OpenAI provider functionality"""
    
    def test_init_with_api_key(self):
        """Test initialization with provided API key"""
        provider = OpenAIProvider(api_key="test-key", model="gpt-4")
        assert provider.api_key == "test-key"
        assert provider.model == "gpt-4"
        assert provider._client is None
    
    def test_init_with_env_var(self):
        """Test initialization using environment variable"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}, clear=True):
            provider = OpenAIProvider()
            assert provider.api_key == "env-key"
            assert provider.model == "gpt-3.5-turbo"  # default model
    
    def test_init_no_api_key(self):
        """Test initialization with no API key"""
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider()
            assert provider.api_key is None
    
    @patch('openai.OpenAI')
    def test_get_client_success(self, mock_openai_class):
        """Test successful client creation"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(api_key="test-key")
        client = provider._get_client()
        
        assert client == mock_client
        assert provider._client == mock_client
        mock_openai_class.assert_called_once_with(api_key="test-key")
    
    @patch('builtins.__import__')
    def test_get_client_import_error(self, mock_import):
        """Test client creation with missing openai library"""
        mock_import.side_effect = ImportError("No module named 'openai'")
        
        provider = OpenAIProvider(api_key="test-key")
        
        with pytest.raises(ImportError, match="OpenAI library not installed"):
            provider._get_client()
    
    @patch('openai.OpenAI')
    def test_generate_success(self, mock_openai_class):
        """Test successful text generation"""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(api_key="test-key")
        result = provider.generate("test prompt")
        
        assert result == "Generated response"
        mock_client.chat.completions.create.assert_called_once()
        
        # Check call arguments
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == "gpt-3.5-turbo"
        assert call_args[1]['temperature'] == 0.1
        assert call_args[1]['max_tokens'] == 2000
        assert len(call_args[1]['messages']) == 2
    
    @patch('openai.OpenAI')
    def test_generate_with_kwargs(self, mock_openai_class):
        """Test generation with custom parameters"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(api_key="test-key")
        result = provider.generate("test prompt", temperature=0.5, max_tokens=1000)
        
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['temperature'] == 0.5
        assert call_args[1]['max_tokens'] == 1000
    
    @patch('openai.OpenAI')
    def test_generate_api_error(self, mock_openai_class):
        """Test handling of API errors"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(api_key="test-key")
        
        with pytest.raises(Exception, match="API Error"):
            provider.generate("test prompt")


class TestAnthropicProvider:
    """Test Anthropic provider functionality"""
    
    def test_init_with_api_key(self):
        """Test initialization with provided API key"""
        provider = AnthropicProvider(api_key="test-key", model="claude-3-opus-20240229")
        assert provider.api_key == "test-key"
        assert provider.model == "claude-3-opus-20240229"
        assert provider._client is None
    
    def test_init_with_env_var(self):
        """Test initialization using environment variable"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'env-key'}, clear=True):
            provider = AnthropicProvider()
            assert provider.api_key == "env-key"
            assert provider.model == "claude-3-haiku-20240307"  # default model
    
    @patch('anthropic.Anthropic')
    def test_get_client_success(self, mock_anthropic_class):
        """Test successful client creation"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        provider = AnthropicProvider(api_key="test-key")
        client = provider._get_client()
        
        assert client == mock_client
        assert provider._client == mock_client
        mock_anthropic_class.assert_called_once_with(api_key="test-key")
    
    @patch('builtins.__import__')
    def test_get_client_import_error(self, mock_import):
        """Test client creation with missing anthropic library"""
        mock_import.side_effect = ImportError("No module named 'anthropic'")
        
        provider = AnthropicProvider(api_key="test-key")
        
        with pytest.raises(ImportError, match="Anthropic library not installed"):
            provider._get_client()
    
    @patch('anthropic.Anthropic')
    def test_generate_success(self, mock_anthropic_class):
        """Test successful text generation"""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Generated response"
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        provider = AnthropicProvider(api_key="test-key")
        result = provider.generate("test prompt")
        
        assert result == "Generated response"
        mock_client.messages.create.assert_called_once()
        
        # Check call arguments
        call_args = mock_client.messages.create.call_args
        assert call_args[1]['model'] == "claude-3-haiku-20240307"
        assert call_args[1]['temperature'] == 0.1
        assert call_args[1]['max_tokens'] == 2000
        assert len(call_args[1]['messages']) == 1


class TestOllamaProvider:
    """Test Ollama provider functionality"""
    
    def test_init_default(self):
        """Test initialization with default values"""
        provider = OllamaProvider()
        assert provider.model == "llama2"
        assert provider.base_url == "http://localhost:11434"
        assert provider._client is None
    
    def test_init_custom(self):
        """Test initialization with custom values"""
        provider = OllamaProvider(model="codellama", base_url="http://custom:8080")
        assert provider.model == "codellama"
        assert provider.base_url == "http://custom:8080"
    
    @patch('ollama.Client')
    def test_get_client_success(self, mock_ollama_class):
        """Test successful client creation"""
        mock_client = Mock()
        mock_ollama_class.return_value = mock_client
        
        provider = OllamaProvider()
        client = provider._get_client()
        
        assert client == mock_client
        assert provider._client == mock_client
        mock_ollama_class.assert_called_once_with(host="http://localhost:11434")
    
    @patch('builtins.__import__')
    def test_get_client_import_error(self, mock_import):
        """Test client creation with missing ollama library"""
        mock_import.side_effect = ImportError("No module named 'ollama'")
        
        provider = OllamaProvider()
        
        with pytest.raises(ImportError, match="Ollama library not installed"):
            provider._get_client()
    
    @patch('ollama.Client')
    def test_generate_success(self, mock_ollama_class):
        """Test successful text generation"""
        mock_client = Mock()
        mock_client.generate.return_value = {'response': 'Generated response'}
        mock_ollama_class.return_value = mock_client
        
        provider = OllamaProvider()
        result = provider.generate("test prompt")
        
        assert result == "Generated response"
        mock_client.generate.assert_called_once_with(
            model="llama2",
            prompt="test prompt",
            stream=False
        )
    
    @patch('ollama.Client')
    def test_generate_api_error(self, mock_ollama_class):
        """Test handling of API errors"""
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("Ollama Error")
        mock_ollama_class.return_value = mock_client
        
        provider = OllamaProvider()
        
        with pytest.raises(Exception, match="Ollama Error"):
            provider.generate("test prompt")


class TestLLMConfig:
    """Test LLMConfig functionality"""
    
    def test_supported_providers(self):
        """Test supported providers list"""
        assert 'openai' in LLMConfig.SUPPORTED_PROVIDERS
        assert 'anthropic' in LLMConfig.SUPPORTED_PROVIDERS
        assert 'ollama' in LLMConfig.SUPPORTED_PROVIDERS
        assert LLMConfig.SUPPORTED_PROVIDERS['openai'] == OpenAIProvider
        assert LLMConfig.SUPPORTED_PROVIDERS['anthropic'] == AnthropicProvider
        assert LLMConfig.SUPPORTED_PROVIDERS['ollama'] == OllamaProvider
    
    def test_create_provider_openai(self):
        """Test creating OpenAI provider"""
        provider = LLMConfig.create_provider('openai', api_key='test-key')
        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == 'test-key'
    
    def test_create_provider_anthropic(self):
        """Test creating Anthropic provider"""
        provider = LLMConfig.create_provider('anthropic', api_key='test-key')
        assert isinstance(provider, AnthropicProvider)
        assert provider.api_key == 'test-key'
    
    def test_create_provider_ollama(self):
        """Test creating Ollama provider"""
        provider = LLMConfig.create_provider('ollama', model='codellama')
        assert isinstance(provider, OllamaProvider)
        assert provider.model == 'codellama'
    
    def test_create_provider_unsupported(self):
        """Test creating unsupported provider"""
        with pytest.raises(ValueError, match="Unsupported provider: invalid"):
            LLMConfig.create_provider('invalid')
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai-key'}, clear=True)
    def test_auto_detect_openai(self):
        """Test auto-detection with OpenAI key"""
        with patch('analysis.llm_providers.logger') as mock_logger:
            provider = LLMConfig.auto_detect_provider()
            
            assert isinstance(provider, OpenAIProvider)
            assert provider.api_key == 'test-openai-key'
            mock_logger.info.assert_called_with("OpenAI API key found, using OpenAI provider")
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-anthropic-key'}, clear=True)
    def test_auto_detect_anthropic(self):
        """Test auto-detection with Anthropic key (no OpenAI key)"""
        with patch('analysis.llm_providers.logger') as mock_logger:
            provider = LLMConfig.auto_detect_provider()
            
            assert isinstance(provider, AnthropicProvider)
            assert provider.api_key == 'test-anthropic-key'
            mock_logger.info.assert_called_with("Anthropic API key found, using Anthropic provider")
    
    @patch('requests.get')
    def test_auto_detect_ollama(self, mock_requests_get):
        """Test auto-detection with Ollama running"""
        with patch.dict(os.environ, {}, clear=True):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_requests_get.return_value = mock_response
            
            with patch('analysis.llm_providers.logger') as mock_logger:
                provider = LLMConfig.auto_detect_provider()
                
                assert isinstance(provider, OllamaProvider)
                mock_requests_get.assert_called_with('http://localhost:11434/api/version', timeout=2)
                mock_logger.info.assert_called_with("Ollama detected, using Ollama provider")
    
    @patch('requests.get')
    def test_auto_detect_ollama_connection_error(self, mock_requests_get):
        """Test auto-detection when Ollama is not accessible"""
        with patch.dict(os.environ, {}, clear=True):
            mock_requests_get.side_effect = Exception("Connection failed")
            
            with patch('analysis.llm_providers.logger') as mock_logger:
                provider = LLMConfig.auto_detect_provider()
                
                assert provider is None
                mock_logger.warning.assert_called_with("No LLM provider detected. Extraction will use fallback methods.")
    
    @patch('requests.get')
    def test_auto_detect_ollama_wrong_status(self, mock_requests_get):
        """Test auto-detection when Ollama returns wrong status"""
        with patch.dict(os.environ, {}, clear=True):
            mock_response = Mock()
            mock_response.status_code = 404
            mock_requests_get.return_value = mock_response
            
            with patch('analysis.llm_providers.logger') as mock_logger:
                provider = LLMConfig.auto_detect_provider()
                
                assert provider is None
                mock_logger.warning.assert_called_with("No LLM provider detected. Extraction will use fallback methods.")
    
    @patch('requests.get')
    def test_auto_detect_no_providers(self, mock_requests_get):
        """Test auto-detection when no providers are available"""
        with patch.dict(os.environ, {}, clear=True):
            mock_requests_get.side_effect = Exception("Connection failed")
            
            with patch('analysis.llm_providers.logger') as mock_logger:
                provider = LLMConfig.auto_detect_provider()
                
                assert provider is None
                mock_logger.warning.assert_called_with("No LLM provider detected. Extraction will use fallback methods.")


class TestGetDefaultProvider:
    """Test the get_default_provider function"""
    
    @patch('analysis.llm_providers.LLMConfig.auto_detect_provider')
    def test_get_default_provider(self, mock_auto_detect):
        """Test get_default_provider delegates correctly"""
        mock_provider = Mock()
        mock_auto_detect.return_value = mock_provider
        
        result = get_default_provider()
        
        assert result == mock_provider
        mock_auto_detect.assert_called_once()
    
    @patch('analysis.llm_providers.LLMConfig.auto_detect_provider')
    def test_get_default_provider_none(self, mock_auto_detect):
        """Test get_default_provider when no provider available"""
        mock_auto_detect.return_value = None
        
        result = get_default_provider()
        
        assert result is None
        mock_auto_detect.assert_called_once()


class TestIntegration:
    """Integration tests for LLM providers"""
    
    @patch('openai.OpenAI')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=True)
    def test_end_to_end_openai_workflow(self, mock_openai_class):
        """Test complete OpenAI workflow from detection to generation"""
        # Setup mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Extracted requirements"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Auto-detect provider
        provider = LLMConfig.auto_detect_provider()
        assert isinstance(provider, OpenAIProvider)
        
        # Generate text
        result = provider.generate("Extract requirements from this text")
        assert result == "Extracted requirements"
    
    @patch('anthropic.Anthropic')
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}, clear=True)
    def test_end_to_end_anthropic_workflow(self, mock_anthropic_class):
        """Test complete Anthropic workflow from detection to generation"""
        # Setup mock
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Extracted requirements"
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        # Auto-detect provider
        provider = LLMConfig.auto_detect_provider()
        assert isinstance(provider, AnthropicProvider)
        
        # Generate text
        result = provider.generate("Extract requirements from this text")
        assert result == "Extracted requirements"