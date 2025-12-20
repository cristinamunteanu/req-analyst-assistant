import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from analysis.qa import (
    make_llm,
    format_docs,
    QAChain,
    make_qa,
    SYSTEM,
    TEMPLATE
)


class TestConstants:
    """Test module constants"""
    
    def test_system_prompt(self):
        """Test SYSTEM constant is properly defined"""
        assert isinstance(SYSTEM, str)
        assert "careful assistant" in SYSTEM.lower()
        assert "cite sources" in SYSTEM.lower()
        assert "file path" in SYSTEM.lower()
    
    def test_template(self):
        """Test TEMPLATE constant is properly structured"""
        assert isinstance(TEMPLATE, str)
        assert "{system}" in TEMPLATE
        assert "{question}" in TEMPLATE
        assert "{context}" in TEMPLATE
        assert "Question:" in TEMPLATE
        assert "Context:" in TEMPLATE


class TestMakeLLM:
    """Test the make_llm factory function"""
    
    @patch.dict(os.environ, {'LLM_PROVIDER': 'openai', 'OPENAI_MODEL': 'gpt-4'})
    @patch('langchain_openai.ChatOpenAI')
    def test_make_llm_openai(self, mock_chat_openai):
        """Test OpenAI LLM creation"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        result = make_llm()
        
        assert result == mock_llm
        mock_chat_openai.assert_called_once_with(model="gpt-4", temperature=0)
    
    @patch.dict(os.environ, {'LLM_PROVIDER': 'openai'}, clear=True)
    @patch('langchain_openai.ChatOpenAI')
    def test_make_llm_openai_default_model(self, mock_chat_openai):
        """Test OpenAI LLM creation with default model"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        result = make_llm()
        
        assert result == mock_llm
        mock_chat_openai.assert_called_once_with(model="gpt-4o-mini", temperature=0)
    
    @patch.dict(os.environ, {'LLM_PROVIDER': 'huggingface', 'HF_CHAT_MODEL': 'custom/model'})
    @patch('langchain_huggingface.HuggingFaceHub')
    def test_make_llm_huggingface(self, mock_hf_hub):
        """Test HuggingFace LLM creation"""
        mock_llm = Mock()
        mock_hf_hub.return_value = mock_llm
        
        result = make_llm()
        
        assert result == mock_llm
        mock_hf_hub.assert_called_once_with(
            repo_id="custom/model",
            model_kwargs={"temperature": 0}
        )
    
    @patch.dict(os.environ, {'LLM_PROVIDER': 'huggingface'}, clear=True)
    @patch('langchain_huggingface.HuggingFaceHub')
    def test_make_llm_huggingface_default_model(self, mock_hf_hub):
        """Test HuggingFace LLM creation with default model"""
        mock_llm = Mock()
        mock_hf_hub.return_value = mock_llm
        
        result = make_llm()
        
        assert result == mock_llm
        mock_hf_hub.assert_called_once_with(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            model_kwargs={"temperature": 0}
        )
    
    @patch.dict(os.environ, {'LLM_PROVIDER': 'anthropic', 'ANTHROPIC_MODEL': 'claude-3-opus'})
    @patch('langchain_anthropic.Anthropic')
    def test_make_llm_anthropic(self, mock_anthropic):
        """Test Anthropic LLM creation"""
        mock_llm = Mock()
        mock_anthropic.return_value = mock_llm
        
        result = make_llm()
        
        assert result == mock_llm
        mock_anthropic.assert_called_once_with(model="claude-3-opus", temperature=0)
    
    @patch.dict(os.environ, {'LLM_PROVIDER': 'anthropic'}, clear=True)
    @patch('langchain_anthropic.Anthropic')
    def test_make_llm_anthropic_default_model(self, mock_anthropic):
        """Test Anthropic LLM creation with default model"""
        mock_llm = Mock()
        mock_anthropic.return_value = mock_llm
        
        result = make_llm()
        
        assert result == mock_llm
        mock_anthropic.assert_called_once_with(model="claude-3-haiku-20240307", temperature=0)
    
    @patch.dict(os.environ, {'LLM_PROVIDER': 'ollama', 'OLLAMA_MODEL': 'llama2'})
    @patch('langchain_ollama.Ollama')
    def test_make_llm_ollama(self, mock_ollama):
        """Test Ollama LLM creation"""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        result = make_llm()
        
        assert result == mock_llm
        mock_ollama.assert_called_once_with(model="llama2", temperature=0)
    
    @patch.dict(os.environ, {'LLM_PROVIDER': 'ollama'}, clear=True)
    @patch('langchain_ollama.Ollama')
    def test_make_llm_ollama_default_model(self, mock_ollama):
        """Test Ollama LLM creation with default model"""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        result = make_llm()
        
        assert result == mock_llm
        mock_ollama.assert_called_once_with(model="llama3", temperature=0)
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('langchain_openai.ChatOpenAI')
    def test_make_llm_default_provider(self, mock_chat_openai):
        """Test default provider (OpenAI) when no LLM_PROVIDER set"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        result = make_llm()
        
        assert result == mock_llm
        mock_chat_openai.assert_called_once_with(model="gpt-4o-mini", temperature=0)
    
    @patch.dict(os.environ, {'LLM_PROVIDER': 'unknown'})
    def test_make_llm_unknown_provider(self):
        """Test error handling for unknown provider"""
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER: unknown"):
            make_llm()
    
    @patch.dict(os.environ, {'LLM_PROVIDER': 'openai'})
    @patch('langchain_openai.ChatOpenAI')
    def test_make_llm_import_error(self, mock_chat_openai):
        """Test handling of import errors"""
        mock_chat_openai.side_effect = ImportError("No module named 'langchain_openai'")
        
        with pytest.raises(ImportError):
            make_llm()
    
    @patch.dict(os.environ, {'LLM_PROVIDER': 'openai'})
    @patch('langchain_openai.ChatOpenAI')
    def test_make_llm_initialization_error(self, mock_chat_openai):
        """Test handling of initialization errors"""
        mock_chat_openai.side_effect = Exception("API key not found")
        
        with pytest.raises(Exception, match="API key not found"):
            make_llm()


class TestFormatDocs:
    """Test the format_docs function"""
    
    def test_format_docs_single_document(self):
        """Test formatting single document"""
        doc = Mock()
        doc.page_content = "This is the content of document one."
        
        result = format_docs([doc])
        
        assert result == "This is the content of document one."
    
    def test_format_docs_multiple_documents(self):
        """Test formatting multiple documents"""
        doc1 = Mock()
        doc1.page_content = "Content of document one."
        
        doc2 = Mock()
        doc2.page_content = "Content of document two."
        
        doc3 = Mock()
        doc3.page_content = "Content of document three."
        
        result = format_docs([doc1, doc2, doc3])
        
        expected = "Content of document one.\n\nContent of document two.\n\nContent of document three."
        assert result == expected
    
    def test_format_docs_empty_list(self):
        """Test formatting empty document list"""
        result = format_docs([])
        assert result == ""
    
    def test_format_docs_empty_content(self):
        """Test formatting documents with empty content"""
        doc1 = Mock()
        doc1.page_content = ""
        
        doc2 = Mock()
        doc2.page_content = "Real content"
        
        doc3 = Mock()
        doc3.page_content = ""
        
        result = format_docs([doc1, doc2, doc3])
        
        expected = "\n\nReal content\n\n"
        assert result == expected
    
    def test_format_docs_whitespace_handling(self):
        """Test formatting documents with various whitespace"""
        doc1 = Mock()
        doc1.page_content = "  Content with spaces  "
        
        doc2 = Mock()
        doc2.page_content = "\nContent with newlines\n"
        
        result = format_docs([doc1, doc2])
        
        expected = "  Content with spaces  \n\n\nContent with newlines\n"
        assert result == expected


class TestQAChain:
    """Test the QAChain class"""
    
    def create_mock_retriever(self):
        """Create a mock retriever that supports LCEL operations"""
        mock_retriever = Mock()
        # Mock the pipe operation to return a chainable object
        mock_pipeline = Mock()
        mock_retriever.__or__ = Mock(return_value=mock_pipeline)
        return mock_retriever
    
    def create_mock_runnable_chain(self):
        """Create a mock chain that supports LCEL operations"""
        mock_chain = Mock()
        mock_chain.__or__ = Mock(return_value=mock_chain)
        return mock_chain
    
    @patch('langchain_core.runnables.RunnablePassthrough')
    @patch('langchain_core.output_parsers.StrOutputParser')
    def test_qa_chain_initialization(self, mock_str_parser, mock_passthrough):
        """Test QAChain initialization"""
        mock_retriever = self.create_mock_retriever()
        mock_llm = Mock()
        mock_prompt = Mock()
        
        # Mock the LCEL components
        mock_passthrough.return_value = Mock()
        mock_str_parser.return_value = Mock()
        
        qa_chain = QAChain(mock_retriever, mock_llm, mock_prompt)
        
        assert qa_chain.retriever == mock_retriever
        assert qa_chain.llm == mock_llm
        assert qa_chain.prompt == mock_prompt
        assert qa_chain.rag_chain is not None
    
    @patch('langchain_core.runnables.RunnablePassthrough')
    @patch('langchain_core.output_parsers.StrOutputParser')
    def test_qa_chain_call_with_query(self, mock_str_parser, mock_passthrough):
        """Test QAChain call with 'query' key"""
        mock_retriever = self.create_mock_retriever()
        mock_llm = Mock()
        mock_prompt = Mock()
        
        # Setup retriever mock
        mock_doc = Mock()
        mock_doc.page_content = "Retrieved document content"
        mock_retriever.invoke.return_value = [mock_doc]
        
        # Mock LCEL components
        mock_passthrough.return_value = Mock()
        mock_str_parser.return_value = Mock()
        
        # Setup the rag_chain mock
        qa_chain = QAChain(mock_retriever, mock_llm, mock_prompt)
        qa_chain.rag_chain = Mock()
        qa_chain.rag_chain.invoke.return_value = "Generated answer"
        
        inputs = {"query": "What is authentication?"}
        result = qa_chain(inputs)
        
        assert result["result"] == "Generated answer"
        assert result["source_documents"] == [mock_doc]
        qa_chain.rag_chain.invoke.assert_called_once_with("What is authentication?")
        mock_retriever.invoke.assert_called_once_with("What is authentication?")
    
    @patch('langchain_core.runnables.RunnablePassthrough')
    @patch('langchain_core.output_parsers.StrOutputParser')
    def test_qa_chain_call_with_question(self, mock_str_parser, mock_passthrough):
        """Test QAChain call with 'question' key"""
        mock_retriever = self.create_mock_retriever()
        mock_llm = Mock()
        mock_prompt = Mock()
        
        mock_doc = Mock()
        mock_doc.page_content = "Document content"
        mock_retriever.invoke.return_value = [mock_doc]
        
        # Mock LCEL components
        mock_passthrough.return_value = Mock()
        mock_str_parser.return_value = Mock()
        
        qa_chain = QAChain(mock_retriever, mock_llm, mock_prompt)
        qa_chain.rag_chain = Mock()
        qa_chain.rag_chain.invoke.return_value = "Answer"
        
        inputs = {"question": "How does it work?"}
        result = qa_chain(inputs)
        
        assert result["result"] == "Answer"
        assert result["source_documents"] == [mock_doc]
        qa_chain.rag_chain.invoke.assert_called_once_with("How does it work?")
    
    @patch('langchain_core.runnables.RunnablePassthrough')
    @patch('langchain_core.output_parsers.StrOutputParser')
    def test_qa_chain_call_no_question(self, mock_str_parser, mock_passthrough):
        """Test QAChain call without question"""
        mock_retriever = self.create_mock_retriever()
        mock_llm = Mock()
        mock_prompt = Mock()
        
        # Mock LCEL components
        mock_passthrough.return_value = Mock()
        mock_str_parser.return_value = Mock()
        
        qa_chain = QAChain(mock_retriever, mock_llm, mock_prompt)
        
        inputs = {"other_key": "value"}
        
        with pytest.raises(ValueError, match="No question provided in inputs"):
            qa_chain(inputs)
    
    @patch('langchain_core.runnables.RunnablePassthrough')
    @patch('langchain_core.output_parsers.StrOutputParser')
    def test_qa_chain_call_empty_inputs(self, mock_str_parser, mock_passthrough):
        """Test QAChain call with empty inputs"""
        mock_retriever = self.create_mock_retriever()
        mock_llm = Mock()
        mock_prompt = Mock()
        
        # Mock LCEL components
        mock_passthrough.return_value = Mock()
        mock_str_parser.return_value = Mock()
        
        qa_chain = QAChain(mock_retriever, mock_llm, mock_prompt)
        
        with pytest.raises(ValueError, match="No question provided in inputs"):
            qa_chain({})


class TestMakeQA:
    """Test the make_qa factory function"""
    
    @patch('analysis.qa.make_llm')
    @patch('langchain.prompts.PromptTemplate')
    @patch('analysis.qa.QAChain')
    def test_make_qa_success(self, mock_qa_chain_class, mock_prompt_template_class, mock_make_llm):
        """Test successful QA chain creation"""
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_prompt = Mock()
        mock_qa_chain = Mock()
        
        mock_make_llm.return_value = mock_llm
        mock_prompt_template_class.return_value = mock_prompt
        mock_qa_chain_class.return_value = mock_qa_chain
        
        result = make_qa(mock_retriever)
        
        assert result == mock_qa_chain
        
        # Verify prompt template creation
        mock_prompt_template_class.assert_called_once_with(
            input_variables=["question", "context"],
            template=TEMPLATE,
            partial_variables={"system": SYSTEM}
        )
        
        # Verify LLM creation
        mock_make_llm.assert_called_once()
        
        # Verify QAChain creation
        mock_qa_chain_class.assert_called_once_with(mock_retriever, mock_llm, mock_prompt)
    
    @patch('analysis.qa.make_llm')
    def test_make_qa_llm_error(self, mock_make_llm):
        """Test make_qa with LLM creation error"""
        mock_retriever = Mock()
        mock_make_llm.side_effect = Exception("LLM initialization failed")
        
        with pytest.raises(Exception, match="LLM initialization failed"):
            make_qa(mock_retriever)
    
    @patch('analysis.qa.make_llm')
    @patch('langchain.prompts.PromptTemplate')
    def test_make_qa_prompt_error(self, mock_prompt_template_class, mock_make_llm):
        """Test make_qa with prompt template error"""
        mock_retriever = Mock()
        mock_llm = Mock()
        
        mock_make_llm.return_value = mock_llm
        mock_prompt_template_class.side_effect = Exception("Prompt template error")
        
        with pytest.raises(Exception, match="Prompt template error"):
            make_qa(mock_retriever)
    
    @patch('analysis.qa.make_llm')
    @patch('langchain.prompts.PromptTemplate')
    @patch('analysis.qa.QAChain')
    def test_make_qa_chain_error(self, mock_qa_chain_class, mock_prompt_template_class, mock_make_llm):
        """Test make_qa with QAChain creation error"""
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_prompt = Mock()
        
        mock_make_llm.return_value = mock_llm
        mock_prompt_template_class.return_value = mock_prompt
        mock_qa_chain_class.side_effect = Exception("QAChain creation failed")
        
        with pytest.raises(Exception, match="QAChain creation failed"):
            make_qa(mock_retriever)


class TestIntegration:
    """Integration tests for the qa module"""
    
    @patch.dict(os.environ, {'LLM_PROVIDER': 'openai'})
    @patch('langchain_openai.ChatOpenAI')
    def test_end_to_end_workflow(self, mock_chat_openai):
        """Test complete end-to-end QA workflow"""
        # Setup mocks
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        mock_retriever = Mock()
        mock_doc1 = Mock()
        mock_doc1.page_content = "Authentication is the process of verifying user identity."
        mock_doc2 = Mock()
        mock_doc2.page_content = "Multi-factor authentication provides additional security."
        mock_retriever.invoke.return_value = [mock_doc1, mock_doc2]
        
        # Create QA chain
        qa_chain = make_qa(mock_retriever)
        
        # Mock the rag_chain to return a realistic answer
        qa_chain.rag_chain = Mock()
        qa_chain.rag_chain.invoke.return_value = "Authentication verifies user identity using multiple factors for security."
        
        # Test the workflow
        inputs = {"query": "What is authentication?"}
        result = qa_chain(inputs)
        
        # Verify results
        assert "result" in result
        assert "source_documents" in result
        assert result["result"] == "Authentication verifies user identity using multiple factors for security."
        assert len(result["source_documents"]) == 2
        assert result["source_documents"][0] == mock_doc1
        assert result["source_documents"][1] == mock_doc2
        
        # Verify the chain was called with the question
        qa_chain.rag_chain.invoke.assert_called_once_with("What is authentication?")
        mock_retriever.invoke.assert_called_once_with("What is authentication?")


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_format_docs_with_none_content(self):
        """Test format_docs when document has None page_content"""
        doc = Mock()
        doc.page_content = None
        
        # This should raise an TypeError when trying to join None
        with pytest.raises(TypeError):
            format_docs([doc])
    
    def test_format_docs_with_non_string_content(self):
        """Test format_docs when document has non-string page_content"""
        doc = Mock()
        doc.page_content = 123
        
        # join() should raise TypeError for non-string content
        with pytest.raises(TypeError, match="sequence item 0: expected str instance, int found"):
            format_docs([doc])
    
    @patch.dict(os.environ, {'LLM_PROVIDER': ''})
    def test_make_llm_empty_provider(self):
        """Test make_llm with empty provider string"""
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER:"):
            make_llm()
    
    def test_make_qa_with_none_retriever(self):
        """Test make_qa with None retriever raises TypeError"""
        with patch('analysis.qa.make_llm') as mock_make_llm:
            mock_llm = Mock()
            mock_make_llm.return_value = mock_llm
            
            with pytest.raises(TypeError, match="unsupported operand type"):
                make_qa(None)


# Performance and stress tests
class TestPerformance:
    """Test performance-related scenarios"""
    
    def test_format_docs_large_number(self):
        """Test formatting large number of documents"""
        docs = []
        for i in range(1000):
            doc = Mock()
            doc.page_content = f"Document {i} content"
            docs.append(doc)
        
        result = format_docs(docs)
        
        # Should handle large number of docs without issues
        assert result.count("\n\n") == 999  # 999 separators for 1000 docs
        assert "Document 0 content" in result
        assert "Document 999 content" in result
    
    def test_format_docs_very_long_content(self):
        """Test formatting documents with very long content"""
        doc = Mock()
        doc.page_content = "A" * 100000  # Very long content
        
        result = format_docs([doc])
        
        assert len(result) == 100000
        assert result == "A" * 100000


# Additional functional tests
class TestFunctionalBehavior:
    """Test functional behavior without complex mocking"""
    
    def test_make_llm_function_exists(self):
        """Test that make_llm function exists and is callable"""
        assert callable(make_llm)
    
    def test_format_docs_function_behavior(self):
        """Test format_docs with realistic document objects"""
        # Create realistic mock documents
        doc1 = Mock()
        doc1.page_content = "First paragraph of text."
        
        doc2 = Mock()  
        doc2.page_content = "Second paragraph of text."
        
        result = format_docs([doc1, doc2])
        expected = "First paragraph of text.\n\nSecond paragraph of text."
        
        assert result == expected
        assert "\n\n" in result
        assert result.count("\n\n") == 1
    
    def test_constants_are_strings(self):
        """Test that all constants are properly defined strings"""
        assert isinstance(SYSTEM, str)
        assert isinstance(TEMPLATE, str)
        assert len(SYSTEM) > 0
        assert len(TEMPLATE) > 0
        
        # Test template has required placeholders
        required_placeholders = ["{system}", "{question}", "{context}"]
        for placeholder in required_placeholders:
            assert placeholder in TEMPLATE, f"Missing placeholder: {placeholder}"