import pytest
import os
import hashlib
import pickle
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, mock_open, MagicMock
from analysis.normalize_requirements import (
    _hash_doc,
    split_into_requirements,
    parse_normalized_requirement_response,
    normalize_requirements,
    NORMALIZE_PROMPT
)


class TestHashDoc:
    """Test the _hash_doc function"""
    
    def test_hash_doc_valid_inputs(self):
        """Test hash generation with valid inputs"""
        text = "This is a test requirement"
        model = "sentence-transformers/all-MiniLM-L6-v2"
        
        result = _hash_doc(text, model)
        
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex string length
        assert all(c in '0123456789abcdef' for c in result)
    
    def test_hash_doc_consistent(self):
        """Test that hash is consistent for same inputs"""
        text = "Test requirement text"
        model = "test-model"
        
        hash1 = _hash_doc(text, model)
        hash2 = _hash_doc(text, model)
        
        assert hash1 == hash2
    
    def test_hash_doc_different_text(self):
        """Test that different texts produce different hashes"""
        model = "test-model"
        text1 = "First requirement"
        text2 = "Second requirement"
        
        hash1 = _hash_doc(text1, model)
        hash2 = _hash_doc(text2, model)
        
        assert hash1 != hash2
    
    def test_hash_doc_different_model(self):
        """Test that different models produce different hashes"""
        text = "Same requirement text"
        model1 = "model-v1"
        model2 = "model-v2"
        
        hash1 = _hash_doc(text, model1)
        hash2 = _hash_doc(text, model2)
        
        assert hash1 != hash2
    
    def test_hash_doc_empty_text_error(self):
        """Test error handling for empty text"""
        with pytest.raises(ValueError, match="'text' must be a non-empty string"):
            _hash_doc("", "model")
        
        with pytest.raises(ValueError, match="'text' must be a non-empty string"):
            _hash_doc("   ", "model")
    
    def test_hash_doc_empty_model_error(self):
        """Test error handling for empty model"""
        with pytest.raises(ValueError, match="'embed_model' must be a non-empty string"):
            _hash_doc("text", "")
        
        with pytest.raises(ValueError, match="'embed_model' must be a non-empty string"):
            _hash_doc("text", "   ")
    
    def test_hash_doc_non_string_inputs_error(self):
        """Test error handling for non-string inputs"""
        with pytest.raises(ValueError, match="Both 'text' and 'embed_model' must be strings"):
            _hash_doc(123, "model")
        
        with pytest.raises(ValueError, match="Both 'text' and 'embed_model' must be strings"):
            _hash_doc("text", 456)
        
        with pytest.raises(ValueError, match="Both 'text' and 'embed_model' must be strings"):
            _hash_doc(None, "model")
    
    def test_hash_doc_unicode_handling(self):
        """Test hash generation with Unicode text"""
        text = "Test with émojis 🚀 and accénts"
        model = "test-model"
        
        result = _hash_doc(text, model)
        
        assert isinstance(result, str)
        assert len(result) == 64
        
        # Should be consistent
        assert result == _hash_doc(text, model)


class TestSplitIntoRequirements:
    """Test the split_into_requirements function"""
    
    def test_split_numbered_list(self):
        """Test splitting numbered list"""
        text = """Introduction text here.
        
1. The system shall authenticate users within 5 seconds.
2. The application must encrypt all sensitive data using AES-256.
3. The interface should be responsive on all devices.

Conclusion text."""
        
        result = split_into_requirements(text)
        
        assert len(result) == 4  # Including intro text
        assert "Introduction text here." in result[0]
        assert "authenticate users" in result[1]
        assert "encrypt all sensitive data" in result[2]
        assert "responsive on all devices" in result[3]
    
    def test_split_bullet_points_dash(self):
        """Test splitting dash bullet points"""
        text = """- The system shall validate all input data before processing.
- User sessions must timeout after 30 minutes of inactivity.
- Error messages should be displayed in user-friendly language."""
        
        result = split_into_requirements(text)
        
        assert len(result) == 3
        assert "validate all input data" in result[0]
        assert "timeout after 30 minutes" in result[1]
        assert "user-friendly language" in result[2]
    
    def test_split_bullet_points_asterisk(self):
        """Test splitting asterisk bullet points"""
        text = """* Authentication module shall support OAuth 2.0 protocol.
* System logs must be retained for minimum 90 days.
* Database backups should be performed daily."""
        
        result = split_into_requirements(text)
        
        assert len(result) == 3
        assert "OAuth 2.0 protocol" in result[0]
        assert "retained for minimum 90 days" in result[1]
        assert "performed daily" in result[2]
    
    def test_split_mixed_formats(self):
        """Test splitting mixed numbering and bullet formats"""
        text = """1. Primary requirement with detailed description.
- Secondary bullet point requirement.
* Another asterisk requirement.
2. Second numbered requirement."""
        
        result = split_into_requirements(text)
        
        assert len(result) == 4
        assert "Primary requirement" in result[0]
        assert "Secondary bullet" in result[1] 
        assert "asterisk requirement" in result[2]
        assert "Second numbered" in result[3]
    
    def test_split_empty_string(self):
        """Test splitting empty string"""
        result = split_into_requirements("")
        assert result == []
    
    def test_split_no_matches(self):
        """Test splitting text with no numbered/bullet items"""
        text = "This is just plain text with no numbered or bulleted items."
        
        result = split_into_requirements(text)
        
        # Should return the original text if it's long enough
        assert len(result) == 1
        assert "plain text" in result[0]
    
    def test_split_short_parts_filtered(self):
        """Test that short parts are filtered out"""
        text = """1. Valid long requirement text that exceeds the minimum length.
2. Short.
3. Another valid requirement with sufficient content length."""
        
        result = split_into_requirements(text)
        
        # Should exclude the short "Short." requirement
        assert len(result) == 2
        assert "Valid long requirement" in result[0]
        assert "Another valid requirement" in result[1]
    
    def test_split_whitespace_handling(self):
        """Test proper whitespace handling"""
        text = """   1.   Requirement with extra spaces   .
        
        2.    Another requirement with    weird spacing   .   """
        
        result = split_into_requirements(text)
        
        assert len(result) == 2
        # Results should be stripped
        assert result[0] == "Requirement with extra spaces   ."
        assert result[1] == "Another requirement with    weird spacing   ."


class TestParseNormalizedRequirementResponse:
    """Test the parse_normalized_requirement_response function"""
    
    def test_parse_clean_json(self):
        """Test parsing clean JSON response"""
        response = '{"normalized": "User Authentication", "categories": ["Security", "Functional"]}'
        
        result = parse_normalized_requirement_response(response)
        
        assert result["normalized"] == "User Authentication"
        assert result["categories"] == ["Security", "Functional"]
    
    def test_parse_json_with_whitespace(self):
        """Test parsing JSON with extra whitespace"""
        response = '  {"normalized": "Data Validation", "categories": ["Functional"]}  '
        
        result = parse_normalized_requirement_response(response)
        
        assert result["normalized"] == "Data Validation"
        assert result["categories"] == ["Functional"]
    
    def test_parse_markdown_wrapped_json(self):
        """Test parsing JSON wrapped in markdown code blocks"""
        response = '''```json
{"normalized": "Session Timeout", "categories": ["Security", "Performance"]}
```'''
        
        result = parse_normalized_requirement_response(response)
        
        assert result["normalized"] == "Session Timeout"
        assert result["categories"] == ["Security", "Performance"]
    
    def test_parse_markdown_with_extra_content(self):
        """Test parsing markdown with extra text around JSON"""
        response = '''Here's the analysis:
```json
{"normalized": "Error Handling", "categories": ["Usability", "Reliability"]}
```
Additional notes here.'''
        
        result = parse_normalized_requirement_response(response)
        
        assert result["normalized"] == "Error Handling"
        assert result["categories"] == ["Usability", "Reliability"]
    
    def test_parse_legacy_content_format(self):
        """Test parsing legacy content='```json ... ```' format"""
        response = "content='```json {\"normalized\": \"Data Backup\", \"categories\": [\"Compliance\"]} ```'"
        
        result = parse_normalized_requirement_response(response)
        
        assert result["normalized"] == "Data Backup"
        assert result["categories"] == ["Compliance"]
    
    def test_parse_legacy_multiline_format(self):
        """Test parsing legacy format with multiline JSON"""
        response = '''content='```json
{
  "normalized": "Performance Monitoring", 
  "categories": ["Performance", "Reliability"]
}
```' '''
        
        result = parse_normalized_requirement_response(response)
        
        assert result["normalized"] == "Performance Monitoring"
        assert result["categories"] == ["Performance", "Reliability"]
    
    def test_parse_invalid_json(self):
        """Test handling of invalid JSON"""
        response = '{"normalized": "Invalid JSON", "categories":}'
        
        result = parse_normalized_requirement_response(response)
        
        assert result["normalized"] == response
        assert result["categories"] == ["Other"]
    
    def test_parse_non_json_text(self):
        """Test handling of plain text (not JSON)"""
        response = "This is just plain text, not JSON at all."
        
        result = parse_normalized_requirement_response(response)
        
        assert result["normalized"] == response
        assert result["categories"] == ["Other"]
    
    def test_parse_empty_string(self):
        """Test handling of empty string"""
        result = parse_normalized_requirement_response("")
        
        assert result["normalized"] == ""
        assert result["categories"] == ["Other"]
    
    def test_parse_malformed_legacy_format(self):
        """Test handling of malformed legacy format"""
        response = "content='```json {\"normalized\": \"Test\"} ' invalid"
        
        result = parse_normalized_requirement_response(response)
        
        # Should fallback to treating the whole string as normalized text
        assert result["normalized"] == response
        assert result["categories"] == ["Other"]
    
    def test_parse_json_missing_keys(self):
        """Test handling of JSON missing required keys"""
        response = '{"other_key": "value"}'
        
        result = parse_normalized_requirement_response(response)
        
        assert result.get("normalized", "") == ""
        assert result.get("categories", []) == []


class TestNormalizeRequirements:
    """Test the normalize_requirements function"""
    
    def setUp_temp_dir(self):
        """Helper to create temporary directory for testing"""
        return tempfile.mkdtemp()
    
    def tearDown_temp_dir(self, temp_dir):
        """Helper to cleanup temporary directory"""
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @patch('analysis.normalize_requirements.make_llm')
    def test_normalize_requirements_success(self, mock_make_llm):
        """Test successful requirement normalization"""
        # Setup mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '{"normalized": "User Login", "categories": ["Functional", "Security"]}'
        mock_llm.invoke.return_value = mock_response
        mock_make_llm.return_value = mock_llm
        
        temp_dir = self.setUp_temp_dir()
        try:
            docs = [
                {"text": "SYS-001: The system shall authenticate users using OAuth 2.0", "source": "requirements.pdf"},
                {"text": "REQ-002: Data must be encrypted at rest", "path": "security.doc"}
            ]
            
            result = normalize_requirements(docs, cache_dir=temp_dir)
            
            assert len(result) == 2
            
            # Check first result
            assert result[0]["source"] == "requirements.pdf"
            assert result[0]["text"] == "SYS-001: The system shall authenticate users using OAuth 2.0"
            assert result[0]["normalized"] == "User Login"
            assert result[0]["categories"] == ["Functional", "Security"]
            
            # Check second result
            assert result[1]["source"] == "security.doc"
            assert result[1]["text"] == "REQ-002: Data must be encrypted at rest"
            assert result[1]["normalized"] == "User Login"  # Same mock response
            
            # Verify LLM was called
            assert mock_llm.invoke.call_count == 2
            
        finally:
            self.tearDown_temp_dir(temp_dir)
    
    @patch('analysis.normalize_requirements.make_llm')
    def test_normalize_requirements_with_cache(self, mock_make_llm):
        """Test requirement normalization with caching"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '{"normalized": "Data Security", "categories": ["Security"]}'
        mock_llm.invoke.return_value = mock_response
        mock_make_llm.return_value = mock_llm
        
        temp_dir = self.setUp_temp_dir()
        try:
            docs = [{"text": "Test requirement", "source": "test.pdf"}]
            
            # First call - should hit LLM
            result1 = normalize_requirements(docs, cache_dir=temp_dir)
            
            # Reset mock to verify cache usage
            mock_llm.reset_mock()
            
            # Second call - should use cache
            result2 = normalize_requirements(docs, cache_dir=temp_dir)
            
            # Results should be identical
            assert result1 == result2
            
            # LLM should not be called second time
            mock_llm.invoke.assert_not_called()
            
        finally:
            self.tearDown_temp_dir(temp_dir)
    
    @patch('analysis.normalize_requirements.make_llm')
    def test_normalize_requirements_string_response(self, mock_make_llm):
        """Test handling of string response from LLM"""
        mock_llm = Mock()
        mock_llm.invoke.return_value = '{"normalized": "String Response", "categories": ["Other"]}'
        mock_make_llm.return_value = mock_llm
        
        temp_dir = self.setUp_temp_dir()
        try:
            docs = [{"text": "Test requirement", "source": "test.pdf"}]
            
            result = normalize_requirements(docs, cache_dir=temp_dir)
            
            assert len(result) == 1
            assert result[0]["normalized"] == "String Response"
            assert result[0]["categories"] == ["Other"]
            
        finally:
            self.tearDown_temp_dir(temp_dir)
    
    @patch('analysis.normalize_requirements.make_llm')
    def test_normalize_requirements_llm_error(self, mock_make_llm):
        """Test handling of LLM errors"""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM API Error")
        mock_make_llm.return_value = mock_llm
        
        temp_dir = self.setUp_temp_dir()
        try:
            docs = [{"text": "Test requirement", "source": "test.pdf"}]
            
            result = normalize_requirements(docs, cache_dir=temp_dir)
            
            assert len(result) == 1
            assert result[0]["id"].startswith("error_")
            assert "ERROR: LLM API Error" in result[0]["normalized"]
            assert result[0]["categories"] == []
            
        finally:
            self.tearDown_temp_dir(temp_dir)
    
    @patch('analysis.normalize_requirements.make_llm')
    def test_normalize_requirements_missing_keys(self, mock_make_llm):
        """Test handling of docs with missing keys"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '{"normalized": "Test", "categories": ["Other"]}'
        mock_llm.invoke.return_value = mock_response
        mock_make_llm.return_value = mock_llm
        
        temp_dir = self.setUp_temp_dir()
        try:
            docs = [
                {"text": "Has text but no source"},
                {"source": "Has source but no text"},
                {}  # Empty dict
            ]
            
            result = normalize_requirements(docs, cache_dir=temp_dir)
            
            assert len(result) == 3
            
            # First doc: has text, no source
            assert result[0]["text"] == "Has text but no source"
            assert result[0]["source"] == "unknown"
            
            # Second doc: has source, no text
            assert result[1]["text"] == ""
            assert result[1]["source"] == "Has source but no text"
            
            # Third doc: empty
            assert result[2]["text"] == ""
            assert result[2]["source"] == "unknown"
            
        finally:
            self.tearDown_temp_dir(temp_dir)
    
    @patch('analysis.normalize_requirements.make_llm')
    @patch('builtins.open', side_effect=OSError("Permission denied"))
    def test_normalize_requirements_cache_error(self, mock_open, mock_make_llm):
        """Test handling of cache file errors"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '{"normalized": "Test", "categories": ["Other"]}'
        mock_llm.invoke.return_value = mock_response
        mock_make_llm.return_value = mock_llm
        
        docs = [{"text": "Test requirement", "source": "test.pdf"}]
        
        result = normalize_requirements(docs, cache_dir="/invalid/cache/dir")
        
        # Should still work despite cache errors
        assert len(result) == 1
        assert result[0]["normalized"] == "Test"
    
    def test_normalize_requirements_empty_docs(self):
        """Test handling of empty documents list"""
        temp_dir = self.setUp_temp_dir()
        try:
            result = normalize_requirements([], cache_dir=temp_dir)
            assert result == []
        finally:
            self.tearDown_temp_dir(temp_dir)
    
    @patch('analysis.normalize_requirements.make_llm')
    def test_normalize_requirements_custom_embed_model(self, mock_make_llm):
        """Test using custom embedding model for cache key"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '{"normalized": "Test", "categories": ["Other"]}'
        mock_llm.invoke.return_value = mock_response
        mock_make_llm.return_value = mock_llm
        
        temp_dir = self.setUp_temp_dir()
        try:
            docs = [{"text": "Same text", "source": "test.pdf"}]
            
            # Normalize with different embedding models
            result1 = normalize_requirements(docs, embed_model="model-v1", cache_dir=temp_dir)
            result2 = normalize_requirements(docs, embed_model="model-v2", cache_dir=temp_dir)
            
            # Should have different cache keys (different IDs)
            assert result1[0]["id"] != result2[0]["id"]
            
            # But both should call LLM (no cache hit)
            assert mock_llm.invoke.call_count == 2
            
        finally:
            self.tearDown_temp_dir(temp_dir)


class TestNormalizePrompt:
    """Test the NORMALIZE_PROMPT template"""
    
    def test_prompt_template_structure(self):
        """Test that the prompt template is properly structured"""
        assert isinstance(NORMALIZE_PROMPT, str)
        assert "{chunk}" in NORMALIZE_PROMPT
        assert "JSON" in NORMALIZE_PROMPT
        assert "normalized" in NORMALIZE_PROMPT
        assert "categories" in NORMALIZE_PROMPT
    
    def test_prompt_template_categories(self):
        """Test that all expected categories are mentioned in prompt"""
        expected_categories = [
            "Functional", "Performance", "Security", "Usability", 
            "Reliability", "Compliance", "Integration", "Other"
        ]
        
        for category in expected_categories:
            assert category in NORMALIZE_PROMPT
    
    def test_prompt_template_formatting(self):
        """Test that the prompt template can be formatted properly"""
        from langchain_core.prompts import PromptTemplate
        
        prompt = PromptTemplate(input_variables=["chunk"], template=NORMALIZE_PROMPT)
        test_chunk = "The system shall authenticate users"
        
        formatted = prompt.format(chunk=test_chunk)
        
        assert test_chunk in formatted
        assert "JSON" in formatted


class TestIntegration:
    """Integration tests for the normalize_requirements module"""
    
    @patch('analysis.normalize_requirements.make_llm')
    def test_end_to_end_workflow(self, mock_make_llm):
        """Test complete end-to-end normalization workflow"""
        # Setup mock LLM with realistic responses
        mock_llm = Mock()
        responses = [
            '{"normalized": "User Authentication", "categories": ["Security", "Functional"]}',
            '```json\n{"normalized": "Data Encryption", "categories": ["Security", "Compliance"]}\n```',
            'content=\'```json {"normalized": "Error Handling", "categories": ["Usability", "Reliability"]} ```\''
        ]
        mock_llm.invoke.side_effect = responses
        mock_make_llm.return_value = mock_llm
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Input documents with various formats
            docs = [
                {
                    "text": "SYS-001: The system shall authenticate users using multi-factor authentication within 30 seconds.",
                    "source": "security_requirements.pdf"
                },
                {
                    "text": "REQ-123: All sensitive data must be encrypted at rest using AES-256 encryption standard.",
                    "path": "data_protection.docx"
                },
                {
                    "text": "UI-005: Error messages should be displayed in user-friendly language without technical jargon."
                }
            ]
            
            result = normalize_requirements(docs, cache_dir=temp_dir)
            
            # Verify results
            assert len(result) == 3
            
            # First requirement
            assert result[0]["normalized"] == "User Authentication"
            assert result[0]["categories"] == ["Security", "Functional"]
            assert result[0]["source"] == "security_requirements.pdf"
            
            # Second requirement
            assert result[1]["normalized"] == "Data Encryption"
            assert result[1]["categories"] == ["Security", "Compliance"]
            assert result[1]["source"] == "data_protection.docx"
            
            # Third requirement (missing source)
            assert result[2]["normalized"] == "Error Handling"
            assert result[2]["categories"] == ["Usability", "Reliability"]
            assert result[2]["source"] == "unknown"
            
            # Verify cache files were created
            cache_files = os.listdir(temp_dir)
            assert len(cache_files) == 3
            assert all(file.endswith('.pkl') for file in cache_files)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_split_and_normalize_workflow(self):
        """Test workflow combining splitting and normalization"""
        # Test realistic requirements document
        document_text = """
        System Requirements Document
        
        1. The system shall authenticate users using OAuth 2.0 protocol within 5 seconds of login attempt.
        2. All user data must be encrypted using AES-256 encryption both in transit and at rest.
        3. The user interface should be responsive and accessible on mobile devices with screen readers.
        4. System must maintain 99.9% uptime excluding scheduled maintenance windows.
        5. Error messages shall be logged with timestamps and displayed to users in plain language.
        """
        
        # Split into requirements
        requirements_list = split_into_requirements(document_text)
        
        # Should extract 5 requirements plus intro text
        assert len(requirements_list) >= 5
        
        # Convert to document format for normalization
        docs = [
            {"text": req, "source": "system_requirements.pdf"} 
            for req in requirements_list 
            if len(req.strip()) > 20  # Filter very short parts
        ]
        
        # Test that the workflow can handle the split requirements
        assert len(docs) >= 5
        assert all("text" in doc and "source" in doc for doc in docs)


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_hash_function_edge_cases(self):
        """Test hash function with various edge cases"""
        # Very long text
        long_text = "A" * 10000
        result = _hash_doc(long_text, "model")
        assert len(result) == 64
        
        # Special characters
        special_text = "Text with special chars: !@#$%^&*()_+-={}[]|\\:;\"'<>?,./"
        result = _hash_doc(special_text, "model")
        assert len(result) == 64
        
        # Unicode text
        unicode_text = "测试文本 🚀 émoji àccénts"
        result = _hash_doc(unicode_text, "model")
        assert len(result) == 64
    
    @patch('analysis.normalize_requirements.make_llm')
    def test_pickle_serialization_edge_cases(self, mock_make_llm):
        """Test pickle serialization with complex data"""
        mock_llm = Mock()
        complex_response = {
            "normalized": "Complex Response",
            "categories": ["Security", "Performance"],
            "extra_data": {"nested": True, "list": [1, 2, 3]}
        }
        mock_response = Mock()
        mock_response.content = json.dumps(complex_response)
        mock_llm.invoke.return_value = mock_response
        mock_make_llm.return_value = mock_llm
        
        temp_dir = tempfile.mkdtemp()
        try:
            docs = [{"text": "Test", "source": "test.pdf"}]
            
            result = normalize_requirements(docs, cache_dir=temp_dir)
            
            # Should handle extra data gracefully
            assert result[0]["normalized"] == "Complex Response"
            assert result[0]["categories"] == ["Security", "Performance"]
            
        finally:
            shutil.rmtree(temp_dir)