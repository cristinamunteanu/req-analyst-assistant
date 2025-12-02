from typing import List, Optional, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .model_extractor import ModelBasedExtractor, ExtractedRequirement
from .llm_providers import get_default_provider
import logging

logger = logging.getLogger(__name__)

def build_index(raw_docs: List[dict], embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Builds a FAISS vector index from a list of document dictionaries.

    Each document dictionary must contain the keys:
        - "text": The document's plain text content.
        - "path": The file path or identifier for the document.

    The function splits each document's text into overlapping chunks, generates embeddings
    for each chunk using a HuggingFace model, and stores the chunks in a FAISS vector store
    along with their source metadata.

    Args:
        raw_docs (List[dict]): List of dictionaries, each with "text" and "path" keys.
        embed_model (str): Name of the HuggingFace embedding model to use.

    Returns:
        FAISS: A FAISS vector store containing the indexed document chunks and metadata,
               or None if embedding/indexing fails.

    Notes:
        - Documents missing "text" or "path" are skipped with a warning.
        - If embedding or indexing fails, an error is printed and None is returned.
    """
    if not isinstance(raw_docs, list):
        raise ValueError("raw_docs must be a list of dictionaries.")
    for i, d in enumerate(raw_docs):
        if not isinstance(d, dict):
            raise ValueError(f"Element at index {i} is not a dictionary.")
        if "text" not in d or "path" not in d:
            raise ValueError(f"Dictionary at index {i} must contain 'text' and 'path' keys.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts, metadatas = [], []
    for i, d in enumerate(raw_docs):
        try:
            text = d["text"]
            path = d["path"]
        except KeyError as e:
            print(f"Skipping document at index {i}: missing key {e}")
            continue
        for chunk in splitter.split_text(text):
            texts.append(chunk)
            metadatas.append({"source": path})
    if not texts:
        print("No text chunks to embed. Check your data and dependencies.")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        return FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    except Exception as e:
        print(f"Failed to build FAISS index or generate embeddings: {e}")
        return None


def extract_requirements_with_model(file_paths: List[str], 
                                   use_llm: bool = True,
                                   llm_provider: Optional[Any] = None) -> List[ExtractedRequirement]:
    """
    Extract requirements from documents using the model-based approach.
    
    This function implements the 3-step process:
    1. Document chunking with structure preservation
    2. LLM-based requirement extraction with JSON schema  
    3. Post-processing with regex helpers
    
    Args:
        file_paths: List of document file paths to process
        use_llm: Whether to use LLM for extraction (fallback to heuristics if False)
        llm_provider: Optional LLM provider instance. If None, auto-detects available provider.
        
    Returns:
        List of ExtractedRequirement objects
    """
    if not file_paths:
        return []
        
    # Auto-detect LLM provider if not provided and LLM is requested
    if use_llm and llm_provider is None:
        llm_provider = get_default_provider()
        if llm_provider is None:
            logger.warning("No LLM provider available, falling back to heuristic extraction")
            use_llm = False
    
    # Create extractor instance
    extractor = ModelBasedExtractor(
        max_chunk_chars=3000,
        llm_provider=llm_provider if use_llm else None
    )
    
    all_requirements = []
    
    for file_path in file_paths:
        try:
            logger.info(f"Extracting requirements from: {file_path}")
            requirements = extractor.extract_requirements(file_path)
            all_requirements.extend(requirements)
        except Exception as e:
            logger.error(f"Failed to extract requirements from {file_path}: {e}")
            continue
    
    return all_requirements


def convert_extracted_to_dict(requirements: List[ExtractedRequirement]) -> List[Dict[str, Any]]:
    """
    Convert ExtractedRequirement objects to dictionary format for compatibility.
    
    Args:
        requirements: List of ExtractedRequirement objects
        
    Returns:
        List of dictionaries with requirement data
    """
    result = []
    
    for req in requirements:
        req_dict = {
            'id': req.id,
            'text': req.text,
            'type': req.type_hint,
            'source': req.source_hint,
            'chunk_index': req.chunk_index,
            'confidence': getattr(req, 'confidence', 1.0),
            'quality_issues': getattr(req, 'quality_issues', [])
        }
        result.append(req_dict)
    
    return result


def create_requirements_summary(requirements: List[ExtractedRequirement]) -> Dict[str, Any]:
    """
    Create a summary of extracted requirements for dashboard display.
    
    Args:
        requirements: List of ExtractedRequirement objects
        
    Returns:
        Dictionary containing summary statistics
    """
    if not requirements:
        return {
            'total_count': 0,
            'by_type': {},
            'with_ids': 0,
            'quality_issues': {}
        }
    
    # Count by type
    by_type = {}
    for req in requirements:
        req_type = req.type_hint
        by_type[req_type] = by_type.get(req_type, 0) + 1
    
    # Count requirements with IDs
    with_ids = sum(1 for req in requirements if req.id)
    
    # Count quality issues
    quality_issues = {}
    for req in requirements:
        issues = getattr(req, 'quality_issues', [])
        for issue in issues:
            quality_issues[issue] = quality_issues.get(issue, 0) + 1
    
    return {
        'total_count': len(requirements),
        'by_type': by_type,
        'with_ids': with_ids,
        'quality_issues': quality_issues,
        'avg_confidence': sum(getattr(req, 'confidence', 1.0) for req in requirements) / len(requirements)
    }
