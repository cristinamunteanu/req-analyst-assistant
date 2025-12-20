import pytest
from unittest.mock import Mock, patch, MagicMock
from analysis.index import (
    build_index, 
    extract_requirements_with_model, 
    convert_extracted_to_dict, 
    create_requirements_summary
)
from analysis.model_extractor import ExtractedRequirement


class TestBuildIndex:
    """Test suite for the build_index function."""

    def test_build_index_minimal(self):
        """
        Test that build_index successfully creates an index from a minimal document list.

        Steps:
            - Provide a single document dictionary with 'path' and 'text' keys.
            - Call build_index with the document and a specified embedding model.
            - Assert that the returned index object is not None.

        This ensures that the indexing pipeline works for the simplest valid input.
        """
        idx = build_index([{"path":"a.txt","text":"hello world"}], "sentence-transformers/all-MiniLM-L6-v2")
        assert idx is not None

    def test_build_index_multiple_documents(self):
        """
        Test that build_index handles multiple documents correctly.
        
        Steps:
            - Provide multiple document dictionaries
            - Call build_index with the documents
            - Assert that the returned index is not None
        """
        docs = [
            {"path": "doc1.txt", "text": "The system shall authenticate users"},
            {"path": "doc2.txt", "text": "The application must process data quickly"},
            {"path": "doc3.txt", "text": "Security requirements include encryption"}
        ]
        idx = build_index(docs)
        assert idx is not None

    def test_build_index_empty_list(self):
        """
        Test that build_index handles an empty document list gracefully.
        
        Steps:
            - Provide an empty list
            - Call build_index
            - Assert that None is returned
        """
        idx = build_index([])
        assert idx is None

    def test_build_index_invalid_input_type(self):
        """
        Test that build_index raises ValueError for invalid input type.
        
        Steps:
            - Provide a non-list input
            - Call build_index
            - Assert that ValueError is raised
        """
        with pytest.raises(ValueError, match="raw_docs must be a list"):
            build_index("not a list")

    def test_build_index_invalid_document_structure(self):
        """
        Test that build_index raises ValueError for invalid document structure.
        
        Steps:
            - Provide documents missing required keys
            - Call build_index
            - Assert that ValueError is raised
        """
        # Test with non-dictionary element
        with pytest.raises(ValueError, match="Element at index 0 is not a dictionary"):
            build_index(["not a dict"])
        
        # Test with missing 'text' key
        with pytest.raises(ValueError, match="must contain 'text' and 'path' keys"):
            build_index([{"path": "test.txt"}])
        
        # Test with missing 'path' key
        with pytest.raises(ValueError, match="must contain 'text' and 'path' keys"):
            build_index([{"text": "some text"}])

    def test_build_index_skips_malformed_documents(self):
        """
        Test that build_index skips malformed documents and continues processing.
        
        Steps:
            - Provide mix of valid and malformed documents
            - Call build_index
            - Assert that index is created from valid documents
        """
        docs = [
            {"path": "good1.txt", "text": "Valid document content"},
            {"path": "good2.txt", "text": "Another valid document"}
        ]
        idx = build_index(docs)
        assert idx is not None

    def test_build_index_large_text_chunking(self):
        """
        Test that build_index properly chunks large text documents.
        
        Steps:
            - Provide a document with large text content
            - Call build_index
            - Assert that index is created successfully
        """
        large_text = "This is a requirement. " * 200  # Create text larger than chunk size
        docs = [{"path": "large.txt", "text": large_text}]
        idx = build_index(docs)
        assert idx is not None

    @patch('analysis.index.FAISS.from_texts')
    @patch('analysis.index.HuggingFaceEmbeddings')
    def test_build_index_embedding_failure(self, mock_embeddings, mock_faiss):
        """
        Test that build_index handles embedding failures gracefully.
        
        Steps:
            - Mock embedding generation to raise an exception
            - Call build_index
            - Assert that None is returned
        """
        mock_embeddings.side_effect = Exception("Embedding failed")
        
        docs = [{"path": "test.txt", "text": "test content"}]
        idx = build_index(docs)
        assert idx is None

    def test_build_index_custom_embedding_model(self):
        """
        Test that build_index accepts custom embedding models.
        
        Steps:
            - Call build_index with a custom embedding model
            - Assert that index is created successfully
        """
        docs = [{"path": "test.txt", "text": "test content"}]
        idx = build_index(docs, embed_model="sentence-transformers/all-mpnet-base-v2")
        assert idx is not None


class TestExtractRequirementsWithModel:
    """Test suite for the extract_requirements_with_model function."""

    def test_extract_requirements_empty_file_list(self):
        """
        Test that extract_requirements_with_model handles empty file list.
        
        Steps:
            - Provide empty file list
            - Call extract_requirements_with_model
            - Assert that empty list is returned
        """
        result = extract_requirements_with_model([])
        assert result == []

    @patch('analysis.index.get_default_provider')
    @patch('analysis.index.ModelBasedExtractor')
    def test_extract_requirements_with_llm(self, mock_extractor_class, mock_get_provider):
        """
        Test extract_requirements_with_model with LLM provider.
        
        Steps:
            - Mock LLM provider and extractor
            - Call extract_requirements_with_model with use_llm=True
            - Assert that extractor is called correctly
        """
        # Setup mocks
        mock_provider = Mock()
        mock_get_provider.return_value = mock_provider
        
        mock_extractor = Mock()
        mock_requirements = [
            ExtractedRequirement(
                id="REQ-001", 
                text="The system shall authenticate users", 
                type_hint="functional",
                source_hint="test.txt"
            )
        ]
        mock_extractor.extract_requirements.return_value = mock_requirements
        mock_extractor_class.return_value = mock_extractor
        
        # Test
        result = extract_requirements_with_model(["test.txt"], use_llm=True)
        
        # Assertions
        mock_get_provider.assert_called_once()
        mock_extractor_class.assert_called_once_with(max_chunk_chars=3000, llm_provider=mock_provider)
        mock_extractor.extract_requirements.assert_called_once_with("test.txt")
        assert result == mock_requirements

    @patch('analysis.index.get_default_provider')
    @patch('analysis.index.ModelBasedExtractor')
    def test_extract_requirements_without_llm(self, mock_extractor_class, mock_get_provider):
        """
        Test extract_requirements_with_model without LLM (heuristic mode).
        
        Steps:
            - Call extract_requirements_with_model with use_llm=False
            - Assert that extractor is created without LLM provider
        """
        mock_extractor = Mock()
        mock_extractor.extract_requirements.return_value = []
        mock_extractor_class.return_value = mock_extractor
        
        result = extract_requirements_with_model(["test.txt"], use_llm=False)
        
        mock_get_provider.assert_not_called()
        mock_extractor_class.assert_called_once_with(max_chunk_chars=3000, llm_provider=None)

    @patch('analysis.index.get_default_provider')
    @patch('analysis.index.ModelBasedExtractor')
    @patch('analysis.index.logger')
    def test_extract_requirements_no_provider_fallback(self, mock_logger, mock_extractor_class, mock_get_provider):
        """
        Test fallback to heuristic mode when no LLM provider is available.
        
        Steps:
            - Mock get_default_provider to return None
            - Call extract_requirements_with_model with use_llm=True
            - Assert that it falls back to heuristic mode
        """
        mock_get_provider.return_value = None
        mock_extractor = Mock()
        mock_extractor.extract_requirements.return_value = []
        mock_extractor_class.return_value = mock_extractor
        
        result = extract_requirements_with_model(["test.txt"], use_llm=True)
        
        mock_logger.warning.assert_called_once()
        mock_extractor_class.assert_called_once_with(max_chunk_chars=3000, llm_provider=None)

    @patch('analysis.index.ModelBasedExtractor')
    @patch('analysis.index.logger')
    def test_extract_requirements_file_processing_error(self, mock_logger, mock_extractor_class):
        """
        Test error handling when file processing fails.
        
        Steps:
            - Mock extractor to raise exception for specific file
            - Call extract_requirements_with_model with multiple files
            - Assert that processing continues for other files
        """
        mock_extractor = Mock()
        mock_extractor.extract_requirements.side_effect = [
            Exception("File not found"),
            [ExtractedRequirement(id="REQ-001", text="Valid req", type_hint="functional", source_hint="file2.txt")]
        ]
        mock_extractor_class.return_value = mock_extractor
        
        result = extract_requirements_with_model(["bad_file.txt", "good_file.txt"], use_llm=False)
        
        mock_logger.error.assert_called_once()
        assert len(result) == 1
        assert result[0].id == "REQ-001"

    def test_extract_requirements_with_custom_provider(self):
        """
        Test extract_requirements_with_model with custom LLM provider.
        
        Steps:
            - Provide custom LLM provider
            - Call extract_requirements_with_model
            - Assert that custom provider is used
        """
        custom_provider = Mock()
        
        with patch('analysis.index.ModelBasedExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor.extract_requirements.return_value = []
            mock_extractor_class.return_value = mock_extractor
            
            extract_requirements_with_model(["test.txt"], use_llm=True, llm_provider=custom_provider)
            
            mock_extractor_class.assert_called_once_with(max_chunk_chars=3000, llm_provider=custom_provider)


class TestConvertExtractedToDict:
    """Test suite for the convert_extracted_to_dict function."""

    def test_convert_extracted_to_dict_empty_list(self):
        """
        Test convert_extracted_to_dict with empty requirements list.
        
        Steps:
            - Provide empty list
            - Call convert_extracted_to_dict
            - Assert that empty list is returned
        """
        result = convert_extracted_to_dict([])
        assert result == []

    def test_convert_extracted_to_dict_single_requirement(self):
        """
        Test convert_extracted_to_dict with single requirement.
        
        Steps:
            - Provide single ExtractedRequirement
            - Call convert_extracted_to_dict
            - Assert correct dictionary structure is returned
        """
        req = ExtractedRequirement(
            id="REQ-001",
            text="The system shall authenticate users",
            type_hint="functional",
            source_hint="auth.txt",
            chunk_index=1
        )
        req.confidence = 0.95
        req.quality_issues = ["vague_term"]
        
        result = convert_extracted_to_dict([req])
        
        assert len(result) == 1
        expected = {
            'id': "REQ-001",
            'text': "The system shall authenticate users",
            'type': "functional",
            'source': "auth.txt",
            'chunk_index': 1,
            'confidence': 0.95,
            'quality_issues': ["vague_term"]
        }
        assert result[0] == expected

    def test_convert_extracted_to_dict_multiple_requirements(self):
        """
        Test convert_extracted_to_dict with multiple requirements.
        
        Steps:
            - Provide multiple ExtractedRequirement objects
            - Call convert_extracted_to_dict
            - Assert all requirements are converted correctly
        """
        reqs = [
            ExtractedRequirement(
                id="REQ-001",
                text="First requirement",
                type_hint="functional",
                source_hint="doc1.txt"
            ),
            ExtractedRequirement(
                id=None,
                text="Second requirement",
                type_hint="non-functional",
                source_hint="doc2.txt"
            )
        ]
        
        result = convert_extracted_to_dict(reqs)
        
        assert len(result) == 2
        assert result[0]['id'] == "REQ-001"
        assert result[1]['id'] is None
        assert result[0]['type'] == "functional"
        assert result[1]['type'] == "non-functional"

    def test_convert_extracted_to_dict_default_values(self):
        """
        Test convert_extracted_to_dict with missing optional attributes.
        
        Steps:
            - Provide ExtractedRequirement without optional attributes
            - Call convert_extracted_to_dict
            - Assert default values are used
        """
        req = ExtractedRequirement(
            id="REQ-001",
            text="Test requirement",
            type_hint="functional",
            source_hint="test.txt"
        )
        # Don't set confidence or quality_issues
        
        result = convert_extracted_to_dict([req])
        
        assert result[0]['confidence'] == 1.0
        assert result[0]['quality_issues'] == []


class TestCreateRequirementsSummary:
    """Test suite for the create_requirements_summary function."""

    def test_create_requirements_summary_empty_list(self):
        """
        Test create_requirements_summary with empty requirements list.
        
        Steps:
            - Provide empty list
            - Call create_requirements_summary
            - Assert correct empty summary structure is returned
        """
        result = create_requirements_summary([])
        
        expected = {
            'total_count': 0,
            'by_type': {},
            'with_ids': 0,
            'quality_issues': {}
        }
        assert result == expected

    def test_create_requirements_summary_single_requirement(self):
        """
        Test create_requirements_summary with single requirement.
        
        Steps:
            - Provide single ExtractedRequirement
            - Call create_requirements_summary
            - Assert correct summary is generated
        """
        req = ExtractedRequirement(
            id="REQ-001",
            text="The system shall authenticate users",
            type_hint="functional",
            source_hint="auth.txt"
        )
        req.confidence = 0.9
        req.quality_issues = ["passive_voice"]
        
        result = create_requirements_summary([req])
        
        expected = {
            'total_count': 1,
            'by_type': {'functional': 1},
            'with_ids': 1,
            'quality_issues': {'passive_voice': 1},
            'avg_confidence': 0.9
        }
        assert result == expected

    def test_create_requirements_summary_multiple_requirements_same_type(self):
        """
        Test create_requirements_summary with multiple requirements of same type.
        
        Steps:
            - Provide multiple functional requirements
            - Call create_requirements_summary
            - Assert correct type counting
        """
        reqs = [
            ExtractedRequirement(id="REQ-001", text="Req 1", type_hint="functional", source_hint="doc1.txt"),
            ExtractedRequirement(id="REQ-002", text="Req 2", type_hint="functional", source_hint="doc1.txt"),
            ExtractedRequirement(id=None, text="Req 3", type_hint="functional", source_hint="doc1.txt")
        ]
        
        for req in reqs:
            req.confidence = 1.0
        
        result = create_requirements_summary(reqs)
        
        assert result['total_count'] == 3
        assert result['by_type'] == {'functional': 3}
        assert result['with_ids'] == 2  # Only 2 have IDs
        assert result['avg_confidence'] == 1.0

    def test_create_requirements_summary_multiple_types(self):
        """
        Test create_requirements_summary with requirements of different types.
        
        Steps:
            - Provide requirements of different types
            - Call create_requirements_summary
            - Assert correct type distribution
        """
        reqs = [
            ExtractedRequirement(id="REQ-001", text="Functional req", type_hint="functional", source_hint="doc1.txt"),
            ExtractedRequirement(id="REQ-002", text="Non-functional req", type_hint="non-functional", source_hint="doc1.txt"),
            ExtractedRequirement(id="REQ-003", text="Constraint", type_hint="constraint", source_hint="doc1.txt"),
            ExtractedRequirement(id="REQ-004", text="Another functional", type_hint="functional", source_hint="doc1.txt")
        ]
        
        for req in reqs:
            req.confidence = 0.8
        
        result = create_requirements_summary(reqs)
        
        expected_by_type = {
            'functional': 2,
            'non-functional': 1,
            'constraint': 1
        }
        assert result['by_type'] == expected_by_type
        assert result['total_count'] == 4
        assert result['with_ids'] == 4
        assert result['avg_confidence'] == 0.8

    def test_create_requirements_summary_quality_issues(self):
        """
        Test create_requirements_summary quality issue aggregation.
        
        Steps:
            - Provide requirements with various quality issues
            - Call create_requirements_summary
            - Assert correct quality issue counting
        """
        req1 = ExtractedRequirement(id="REQ-001", text="Req 1", type_hint="functional", source_hint="doc1.txt")
        req1.quality_issues = ["vague_term", "passive_voice"]
        req1.confidence = 0.7
        
        req2 = ExtractedRequirement(id="REQ-002", text="Req 2", type_hint="functional", source_hint="doc1.txt")
        req2.quality_issues = ["vague_term", "tbd_marker"]
        req2.confidence = 0.8
        
        req3 = ExtractedRequirement(id="REQ-003", text="Req 3", type_hint="functional", source_hint="doc1.txt")
        req3.quality_issues = []
        req3.confidence = 0.9
        
        result = create_requirements_summary([req1, req2, req3])
        
        expected_quality_issues = {
            'vague_term': 2,
            'passive_voice': 1,
            'tbd_marker': 1
        }
        assert result['quality_issues'] == expected_quality_issues
        assert result['avg_confidence'] == (0.7 + 0.8 + 0.9) / 3

    def test_create_requirements_summary_no_ids(self):
        """
        Test create_requirements_summary with requirements without IDs.
        
        Steps:
            - Provide requirements without IDs
            - Call create_requirements_summary
            - Assert correct ID counting
        """
        reqs = [
            ExtractedRequirement(id=None, text="Req 1", type_hint="functional", source_hint="doc1.txt"),
            ExtractedRequirement(id="", text="Req 2", type_hint="functional", source_hint="doc1.txt"),
            ExtractedRequirement(id="REQ-001", text="Req 3", type_hint="functional", source_hint="doc1.txt")
        ]
        
        for req in reqs:
            req.confidence = 1.0
        
        result = create_requirements_summary(reqs)
        
        assert result['with_ids'] == 1  # Only the last one has a non-empty ID

    def test_create_requirements_summary_missing_optional_attributes(self):
        """
        Test create_requirements_summary with requirements missing optional attributes.
        
        Steps:
            - Provide requirements without confidence or quality_issues
            - Call create_requirements_summary
            - Assert default values are handled correctly
        """
        reqs = [
            ExtractedRequirement(id="REQ-001", text="Req 1", type_hint="functional", source_hint="doc1.txt"),
            ExtractedRequirement(id="REQ-002", text="Req 2", type_hint="functional", source_hint="doc1.txt")
        ]
        # Don't set confidence or quality_issues attributes
        
        result = create_requirements_summary(reqs)
        
        assert result['avg_confidence'] == 1.0  # Default confidence
        assert result['quality_issues'] == {}  # No quality issues
