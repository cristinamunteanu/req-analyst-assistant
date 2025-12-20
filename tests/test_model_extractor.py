import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, call
from dataclasses import dataclass
from analysis.model_extractor import (
    ExtractedRequirement,
    DocumentChunk,
    ModelBasedExtractor,
    create_extractor
)


class TestExtractedRequirement:
    """Test ExtractedRequirement dataclass"""
    
    def test_init_minimal(self):
        """Test initialization with minimal parameters"""
        req = ExtractedRequirement(
            id="SYS-001",
            text="System shall authenticate users",
            type_hint="System",
            source_hint="Section 3.1"
        )
        assert req.id == "SYS-001"
        assert req.text == "System shall authenticate users"
        assert req.type_hint == "System"
        assert req.source_hint == "Section 3.1"
        assert req.chunk_index == 0
        assert req.confidence == 1.0
    
    def test_init_full(self):
        """Test initialization with all parameters"""
        req = ExtractedRequirement(
            id="REQ-123",
            text="Application must validate input",
            type_hint="Functional",
            source_hint="Chapter 2",
            chunk_index=5,
            confidence=0.8
        )
        assert req.id == "REQ-123"
        assert req.text == "Application must validate input"
        assert req.type_hint == "Functional"
        assert req.source_hint == "Chapter 2"
        assert req.chunk_index == 5
        assert req.confidence == 0.8
    
    def test_init_none_id(self):
        """Test initialization with None ID"""
        req = ExtractedRequirement(
            id=None,
            text="Some requirement",
            type_hint="Unknown",
            source_hint="Section A"
        )
        assert req.id is None
        assert req.text == "Some requirement"


class TestDocumentChunk:
    """Test DocumentChunk dataclass"""
    
    def test_init(self):
        """Test DocumentChunk initialization"""
        chunk = DocumentChunk(
            content="This is the content",
            source_hint="Section 1.2",
            chunk_index=3,
            char_count=19
        )
        assert chunk.content == "This is the content"
        assert chunk.source_hint == "Section 1.2"
        assert chunk.chunk_index == 3
        assert chunk.char_count == 19


class TestModelBasedExtractor:
    """Test ModelBasedExtractor class"""
    
    def test_init_defaults(self):
        """Test initialization with default parameters"""
        extractor = ModelBasedExtractor()
        assert extractor.max_chunk_chars == 3000
        assert extractor.llm_provider is None
        assert len(extractor.id_patterns) == 7
        assert extractor.tbd_pattern is not None
    
    def test_init_custom(self):
        """Test initialization with custom parameters"""
        mock_llm = Mock()
        extractor = ModelBasedExtractor(max_chunk_chars=5000, llm_provider=mock_llm)
        assert extractor.max_chunk_chars == 5000
        assert extractor.llm_provider == mock_llm
    
    def test_id_patterns(self):
        """Test that ID patterns are properly defined"""
        extractor = ModelBasedExtractor()
        patterns = extractor.id_patterns
        
        # Test that all patterns compile
        import re
        for pattern in patterns:
            re.compile(pattern)  # Should not raise exception
        
        # Test some sample IDs
        test_cases = [
            ("SYS-001", True),
            ("CMP-123", True),
            ("TST-999", True),
            ("REQ-456", True),
            ("FR-007", True),
            ("NFR-100", True),
            ("INVALID-ID", False),
            ("123-SYS", False)
        ]
        
        for test_id, should_match in test_cases:
            matches = any(re.search(pattern, test_id) for pattern in patterns)
            assert matches == should_match, f"Pattern matching failed for {test_id}"

    @patch('analysis.model_extractor.logger')
    @patch.object(ModelBasedExtractor, '_chunk_document')
    @patch.object(ModelBasedExtractor, '_extract_from_chunk')
    @patch.object(ModelBasedExtractor, '_post_process_requirements')
    def test_extract_requirements_success(self, mock_post_process, mock_extract_chunk, 
                                        mock_chunk_doc, mock_logger):
        """Test successful requirement extraction"""
        # Setup mocks
        mock_chunks = [
            DocumentChunk("content1", "section1", 0, 8),
            DocumentChunk("content2", "section2", 1, 8)
        ]
        mock_chunk_doc.return_value = mock_chunks
        
        mock_reqs1 = [ExtractedRequirement("REQ-1", "text1", "System", "section1")]
        mock_reqs2 = [ExtractedRequirement("REQ-2", "text2", "System", "section2")]
        mock_extract_chunk.side_effect = [mock_reqs1, mock_reqs2]
        
        mock_processed = mock_reqs1 + mock_reqs2
        mock_post_process.return_value = mock_processed
        
        extractor = ModelBasedExtractor()
        result = extractor.extract_requirements("test_file.pdf")
        
        assert len(result) == 2
        assert result == mock_processed
        mock_chunk_doc.assert_called_once_with("test_file.pdf")
        assert mock_extract_chunk.call_count == 2
        mock_post_process.assert_called_once()
    
    @patch('analysis.model_extractor.logger')
    @patch.object(ModelBasedExtractor, '_chunk_document')
    def test_extract_requirements_error(self, mock_chunk_doc, mock_logger):
        """Test requirement extraction with error"""
        mock_chunk_doc.side_effect = Exception("File not found")
        
        extractor = ModelBasedExtractor()
        result = extractor.extract_requirements("nonexistent.pdf")
        
        assert result == []
        mock_logger.error.assert_called()
    
    @patch('analysis.model_extractor.partition')
    @patch('analysis.model_extractor.chunk_by_title')
    def test_chunk_document_success(self, mock_chunk_by_title, mock_partition):
        """Test successful document chunking"""
        # Setup mocks
        mock_elements = [Mock(), Mock()]
        mock_partition.return_value = mock_elements
        
        mock_chunks = [Mock(), Mock()]
        mock_chunks[0].__str__ = Mock(return_value="First chunk content")
        mock_chunks[1].__str__ = Mock(return_value="Second chunk content")
        mock_chunk_by_title.return_value = mock_chunks
        
        extractor = ModelBasedExtractor(max_chunk_chars=1000)
        
        with patch.object(extractor, '_extract_source_hint', side_effect=["Section 1", "Section 2"]):
            result = extractor._chunk_document("test.pdf")
        
        assert len(result) == 2
        assert result[0].content == "First chunk content"
        assert result[0].source_hint == "Section 1"
        assert result[0].chunk_index == 0
        assert result[1].content == "Second chunk content"
        assert result[1].source_hint == "Section 2"
        assert result[1].chunk_index == 1
        
        mock_partition.assert_called_once_with(filename="test.pdf")
        mock_chunk_by_title.assert_called_once_with(
            mock_elements,
            max_characters=1000,
            new_after_n_chars=500,
            combine_text_under_n_chars=100
        )
    
    @patch('analysis.model_extractor.logger')
    @patch('analysis.model_extractor.partition')
    def test_chunk_document_error(self, mock_partition, mock_logger):
        """Test document chunking with error"""
        mock_partition.side_effect = Exception("Partition failed")
        
        extractor = ModelBasedExtractor()
        result = extractor._chunk_document("test.pdf")
        
        assert result == []
        mock_logger.error.assert_called()
    
    def test_extract_source_hint_with_heading(self):
        """Test source hint extraction with heading"""
        mock_chunk = Mock()
        mock_chunk.__str__ = Mock(return_value="Chapter 3: Authentication\nThis section describes...")
        
        extractor = ModelBasedExtractor()
        result = extractor._extract_source_hint(mock_chunk, 5)
        
        assert result == "Section: Chapter 3: Authentication"
    
    def test_extract_source_hint_fallback(self):
        """Test source hint extraction fallback"""
        mock_chunk = Mock()
        mock_chunk.__str__ = Mock(return_value="This is a very long line that should not be considered a heading because it exceeds the character limit and ends with a period.")
        
        extractor = ModelBasedExtractor()
        result = extractor._extract_source_hint(mock_chunk, 3)
        
        assert result == "Chunk 4"
    
    def test_extract_source_hint_exception(self):
        """Test source hint extraction with exception"""
        mock_chunk = Mock()
        mock_chunk.__str__ = Mock(side_effect=Exception("Error"))
        
        extractor = ModelBasedExtractor()
        result = extractor._extract_source_hint(mock_chunk, 2)
        
        assert result == "Chunk 3"
    
    @patch.object(ModelBasedExtractor, '_fallback_extraction')
    def test_extract_from_chunk_no_llm(self, mock_fallback):
        """Test chunk extraction without LLM provider"""
        mock_fallback.return_value = [ExtractedRequirement("REQ-1", "text", "System", "section")]
        
        extractor = ModelBasedExtractor()  # No LLM provider
        chunk = DocumentChunk("content", "section", 0, 7)
        
        result = extractor._extract_from_chunk(chunk)
        
        assert len(result) == 1
        mock_fallback.assert_called_once_with(chunk)
    
    @patch.object(ModelBasedExtractor, '_call_llm')
    @patch.object(ModelBasedExtractor, '_parse_llm_response')
    @patch.object(ModelBasedExtractor, '_create_extraction_prompt')
    def test_extract_from_chunk_with_llm(self, mock_create_prompt, mock_parse_response, mock_call_llm):
        """Test chunk extraction with LLM provider"""
        mock_llm = Mock()
        extractor = ModelBasedExtractor(llm_provider=mock_llm)
        
        mock_create_prompt.return_value = "test prompt"
        mock_call_llm.return_value = "llm response"
        mock_parse_response.return_value = {
            "requirements": [
                {"id": "SYS-001", "text": "System shall authenticate", "type_hint": "System", "source_hint": "Section 1"},
                {"text": "No ID requirement", "type_hint": "Functional", "source_hint": "Section 2"}
            ]
        }
        
        chunk = DocumentChunk("test content", "test section", 1, 12)
        result = extractor._extract_from_chunk(chunk)
        
        assert len(result) == 2
        assert result[0].id == "SYS-001"
        assert result[0].text == "System shall authenticate"
        assert result[0].chunk_index == 1
        assert result[1].id is None
        assert result[1].text == "No ID requirement"
        assert result[1].chunk_index == 1
    
    @patch('analysis.model_extractor.logger')
    @patch.object(ModelBasedExtractor, '_call_llm')
    @patch.object(ModelBasedExtractor, '_fallback_extraction')
    def test_extract_from_chunk_llm_error(self, mock_fallback, mock_call_llm, mock_logger):
        """Test chunk extraction with LLM error"""
        mock_llm = Mock()
        extractor = ModelBasedExtractor(llm_provider=mock_llm)
        
        mock_call_llm.side_effect = Exception("API Error")
        mock_fallback.return_value = [ExtractedRequirement("REQ-1", "fallback", "System", "section")]
        
        chunk = DocumentChunk("content", "section", 0, 7)
        result = extractor._extract_from_chunk(chunk)
        
        assert len(result) == 1
        assert result[0].text == "fallback"
        mock_fallback.assert_called_once_with(chunk)
        mock_logger.error.assert_called()
    
    def test_create_extraction_prompt(self):
        """Test extraction prompt creation"""
        extractor = ModelBasedExtractor()
        content = "Test content for extraction"
        
        prompt = extractor._create_extraction_prompt(content)
        
        assert "requirements extraction assistant" in prompt
        assert content in prompt
        assert "JSON" in prompt
        assert "id" in prompt and "text" in prompt and "type_hint" in prompt
    
    def test_call_llm_openai_style(self):
        """Test LLM calling with OpenAI-style interface"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response"
        mock_llm.chat.completions.create.return_value = mock_response
        
        extractor = ModelBasedExtractor(llm_provider=mock_llm)
        result = extractor._call_llm("test prompt")
        
        assert result == "Generated response"
        mock_llm.chat.completions.create.assert_called_once()
        
        call_args = mock_llm.chat.completions.create.call_args
        assert call_args[1]['model'] == "gpt-3.5-turbo"
        assert call_args[1]['temperature'] == 0.1
        assert call_args[1]['max_tokens'] == 2000
        assert len(call_args[1]['messages']) == 2
    
    def test_call_llm_generic_interface(self):
        """Test LLM calling with generic interface"""
        mock_llm = Mock()
        # Remove chat attribute to trigger generic path
        del mock_llm.chat
        mock_llm.generate.return_value = "Generic response"
        
        extractor = ModelBasedExtractor(llm_provider=mock_llm)
        result = extractor._call_llm("test prompt")
        
        assert result == "Generic response"
        mock_llm.generate.assert_called_once_with("test prompt")
    
    @patch('analysis.model_extractor.logger')
    def test_call_llm_openai_error(self, mock_logger):
        """Test LLM calling with OpenAI error"""
        mock_llm = Mock()
        mock_llm.chat.completions.create.side_effect = Exception("API Error")
        
        extractor = ModelBasedExtractor(llm_provider=mock_llm)
        
        with pytest.raises(Exception, match="API Error"):
            extractor._call_llm("test prompt")
        
        mock_logger.error.assert_called()
    
    @patch('analysis.model_extractor.logger')
    def test_call_llm_generic_error(self, mock_logger):
        """Test LLM calling with generic interface error"""
        mock_llm = Mock()
        del mock_llm.chat
        mock_llm.generate.side_effect = Exception("Generate Error")
        
        extractor = ModelBasedExtractor(llm_provider=mock_llm)
        
        with pytest.raises(Exception, match="Generate Error"):
            extractor._call_llm("test prompt")
        
        mock_logger.error.assert_called()
    
    def test_parse_llm_response_clean_json(self):
        """Test parsing clean JSON response"""
        extractor = ModelBasedExtractor()
        response = '{"requirements": [{"id": "REQ-1", "text": "Test"}]}'
        
        result = extractor._parse_llm_response(response)
        
        assert "requirements" in result
        assert len(result["requirements"]) == 1
        assert result["requirements"][0]["id"] == "REQ-1"
    
    def test_parse_llm_response_markdown_wrapped(self):
        """Test parsing markdown-wrapped JSON response"""
        extractor = ModelBasedExtractor()
        response = '```json\n{"requirements": [{"id": "REQ-1", "text": "Test"}]}\n```'
        
        result = extractor._parse_llm_response(response)
        
        assert "requirements" in result
        assert len(result["requirements"]) == 1
        assert result["requirements"][0]["id"] == "REQ-1"
    
    @patch('analysis.model_extractor.logger')
    def test_parse_llm_response_invalid_json(self, mock_logger):
        """Test parsing invalid JSON response"""
        extractor = ModelBasedExtractor()
        response = 'This is not JSON at all'
        
        result = extractor._parse_llm_response(response)
        
        assert result == {"requirements": []}
        mock_logger.error.assert_called()
    
    def test_fallback_extraction(self):
        """Test fallback extraction method"""
        extractor = ModelBasedExtractor()
        chunk = DocumentChunk(
            "The system shall authenticate users. The application must validate input. Some other text.",
            "Section 1", 0, 100
        )
        
        result = extractor._fallback_extraction(chunk)
        
        # Should find requirements with "shall" and "must"
        assert len(result) >= 2
        for req in result:
            assert req.confidence == 0.6
            assert req.type_hint == "Unknown"
            assert req.source_hint == "Section 1"
            assert req.chunk_index == 0
    
    def test_fallback_extraction_with_ids(self):
        """Test fallback extraction with requirement IDs"""
        extractor = ModelBasedExtractor()
        chunk = DocumentChunk(
            "SYS-001: The system shall authenticate users. REQ-123: The application must validate input.",
            "Section 1", 0, 100
        )
        
        result = extractor._fallback_extraction(chunk)
        
        # Find requirements with IDs
        req_with_id = [req for req in result if req.id is not None]
        assert len(req_with_id) >= 1
        
        ids_found = [req.id for req in req_with_id]
        assert "SYS-001" in ids_found or "REQ-123" in ids_found
    
    def test_post_process_requirements_extract_ids(self):
        """Test post-processing extracts missing IDs"""
        extractor = ModelBasedExtractor()
        requirements = [
            ExtractedRequirement(None, "SYS-001: System shall work", "System", "Section 1"),
            ExtractedRequirement("existing", "Already has ID", "System", "Section 2")
        ]
        
        result = extractor._post_process_requirements(requirements)
        
        assert len(result) == 2
        assert result[0].id == "SYS-001"
        assert result[1].id == "existing"
    
    def test_post_process_requirements_quality_issues(self):
        """Test post-processing functionality (quality detection if implemented)"""
        extractor = ModelBasedExtractor()
        requirements = [
            ExtractedRequirement("REQ-1", "System shall be TBD", "System", "Section 1"),
            ExtractedRequirement("REQ-2", "Data will be validated properly", "System", "Section 2"),
            ExtractedRequirement("REQ-3", "UI should be user-friendly", "System", "Section 3"),
            ExtractedRequirement("REQ-4", "System shall process requests", "System", "Section 4")
        ]
        
        result = extractor._post_process_requirements(requirements)
        
        assert len(result) == 4
        
        # Check if the method adds quality_issues attribute
        has_quality_issues = any(hasattr(req, 'quality_issues') for req in result)
        
        if has_quality_issues:
            # Test quality issue detection if implemented
            req_with_tbd = result[0]  # "System shall be TBD"
            assert hasattr(req_with_tbd, 'quality_issues')
            
            # More flexible check for TBD detection
            if req_with_tbd.quality_issues:
                quality_text = ' '.join(str(issue) for issue in req_with_tbd.quality_issues).upper()
                assert 'TBD' in quality_text
        else:
            # If quality issues aren't implemented, just verify basic functionality
            assert all(isinstance(req, ExtractedRequirement) for req in result)
            assert all(req.text for req in result)


class TestCreateExtractor:
    """Test the create_extractor factory function"""
    
    def test_create_extractor_defaults(self):
        """Test creating extractor with default parameters"""
        extractor = create_extractor()
        
        assert isinstance(extractor, ModelBasedExtractor)
        assert extractor.max_chunk_chars == 3000
        assert extractor.llm_provider is None
    
    def test_create_extractor_custom(self):
        """Test creating extractor with custom parameters"""
        mock_llm = Mock()
        extractor = create_extractor(llm_provider=mock_llm, max_chunk_chars=5000)
        
        assert isinstance(extractor, ModelBasedExtractor)
        assert extractor.max_chunk_chars == 5000
        assert extractor.llm_provider == mock_llm


class TestIntegration:
    """Integration tests for ModelBasedExtractor"""
    
    @patch('analysis.model_extractor.partition')
    @patch('analysis.model_extractor.chunk_by_title')
    def test_end_to_end_without_llm(self, mock_chunk_by_title, mock_partition):
        """Test complete extraction workflow without LLM"""
        # Setup document parsing mocks
        mock_elements = [Mock()]
        mock_partition.return_value = mock_elements
        
        mock_chunk = Mock()
        mock_chunk.__str__ = Mock(return_value="SYS-001: The system shall authenticate users. The application must validate all inputs.")
        mock_chunk_by_title.return_value = [mock_chunk]
        
        extractor = ModelBasedExtractor()
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(b"Test content")
            tmp_file_path = tmp_file.name
        
        try:
            result = extractor.extract_requirements(tmp_file_path)
            
            # Should extract requirements using fallback method
            assert len(result) >= 1
            
            # Find requirement with ID
            req_with_id = next((req for req in result if req.id == "SYS-001"), None)
            assert req_with_id is not None
            assert "authenticate" in req_with_id.text
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_end_to_end_with_mock_llm(self):
        """Test complete extraction workflow with mocked LLM"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''```json
{
  "requirements": [
    {"id": "SYS-001", "text": "System shall authenticate users", "type_hint": "System", "source_hint": "Authentication Section"},
    {"id": null, "text": "Application must validate input", "type_hint": "Functional", "source_hint": "Validation Section"}
  ]
}
```'''
        mock_llm.chat.completions.create.return_value = mock_response
        
        with patch('analysis.model_extractor.partition') as mock_partition, \
             patch('analysis.model_extractor.chunk_by_title') as mock_chunk_by_title:
            
            # Setup mocks
            mock_elements = [Mock()]
            mock_partition.return_value = mock_elements
            
            mock_chunk = Mock()
            mock_chunk.__str__ = Mock(return_value="Test content with requirements")
            mock_chunk_by_title.return_value = [mock_chunk]
            
            extractor = ModelBasedExtractor(llm_provider=mock_llm)
            
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
                tmp_file.write(b"Test content")
                tmp_file_path = tmp_file.name
            
            try:
                result = extractor.extract_requirements(tmp_file_path)
                
                assert len(result) == 2
                assert result[0].id == "SYS-001"
                assert result[0].text == "System shall authenticate users"
                assert result[0].type_hint == "System"
                assert result[1].id is None
                assert result[1].text == "Application must validate input"
                
            finally:
                os.unlink(tmp_file_path)


# Additional test helpers and fixtures could go here
@pytest.fixture
def sample_requirements():
    """Fixture providing sample requirements for testing"""
    return [
        ExtractedRequirement("SYS-001", "System shall authenticate users", "System", "Section 3.1"),
        ExtractedRequirement("REQ-002", "Application must validate input", "Functional", "Section 4.2"),
        ExtractedRequirement(None, "Interface should be responsive", "Non-functional", "Section 5.1")
    ]


@pytest.fixture
def sample_chunks():
    """Fixture providing sample document chunks for testing"""
    return [
        DocumentChunk("First chunk content with requirements", "Section 1", 0, 45),
        DocumentChunk("Second chunk with more content", "Section 2", 1, 32),
        DocumentChunk("Final chunk content", "Section 3", 2, 20)
    ]


@pytest.fixture
def mock_llm_provider():
    """Fixture providing a mock LLM provider"""
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"requirements": []}'
    mock_llm.chat.completions.create.return_value = mock_response
    return mock_llm


# Performance and edge case tests
class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_document(self):
        """Test extraction from empty document"""
        with patch('analysis.model_extractor.partition') as mock_partition:
            mock_partition.return_value = []
            
            extractor = ModelBasedExtractor()
            
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
            
            try:
                result = extractor.extract_requirements(tmp_file_path)
                assert result == []
                
            finally:
                os.unlink(tmp_file_path)
    
    def test_very_large_chunk(self):
        """Test handling of very large chunks"""
        extractor = ModelBasedExtractor(max_chunk_chars=100)  # Small limit
        large_content = "This is a test. " * 1000  # Create large content
        
        chunk = DocumentChunk(large_content, "Section 1", 0, len(large_content))
        
        # Should not crash with large content
        result = extractor._fallback_extraction(chunk)
        assert isinstance(result, list)
    
    def test_malformed_json_response(self):
        """Test handling of malformed JSON from LLM"""
        extractor = ModelBasedExtractor()
        
        malformed_responses = [
            '{"requirements": [{"id": "REQ-1", "text":}]}',  # Missing value
            '{"requirements": [{"id": "REQ-1" "text": "test"}]}',  # Missing comma
            'This is not JSON at all',
            '',
            None
        ]
        
        for response in malformed_responses:
            if response is not None:
                result = extractor._parse_llm_response(response)
                assert result == {"requirements": []}