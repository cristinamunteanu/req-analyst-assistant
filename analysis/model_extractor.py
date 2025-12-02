"""
Model-based requirement extractor using LLM for semantic understanding.

This module implements a three-step process:
1. Document chunking with structure preservation
2. LLM-based requirement extraction with JSON schema
3. Post-processing with regex helpers
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedRequirement:
    """Data class for extracted requirements"""
    id: Optional[str]
    text: str
    type_hint: str
    source_hint: str
    chunk_index: int = 0
    confidence: float = 1.0

@dataclass
class DocumentChunk:
    """Data class for document chunks"""
    content: str
    source_hint: str
    chunk_index: int
    char_count: int

class ModelBasedExtractor:
    """
    Model-based requirement extractor that uses LLM for semantic understanding.
    """
    
    def __init__(self, max_chunk_chars: int = 3000, llm_provider=None):
        """
        Initialize the extractor.
        
        Args:
            max_chunk_chars: Maximum characters per chunk
            llm_provider: LLM provider instance (OpenAI, etc.)
        """
        self.max_chunk_chars = max_chunk_chars
        self.llm_provider = llm_provider
        
        # Regex patterns for post-processing
        self.id_patterns = [
            r'\b(SYS-\d+)\b',
            r'\b(CMP-\d+)\b', 
            r'\b(TST-\d+)\b',
            r'\b(REQ-\d+)\b',
            r'\b(FR-\d+)\b',
            r'\b(NFR-\d+)\b',
            r'\b([A-Z]{2,5}-\d{3,5})\b'
        ]
        
        # Quality issue patterns
        self.tbd_pattern = r'\b(TBD|TODO|XXX|FIXME|TBC)\b'
        self.passive_voice_patterns = [
            r'\bis\s+(?:being\s+)?(?:implemented|developed|tested|verified)',
            r'\bshall\s+be\s+(?:implemented|developed|tested|verified)',
            r'\bwill\s+be\s+(?:implemented|developed|tested|verified)'
        ]
        self.vague_terms = [
            r'\b(?:reasonable|appropriate|sufficient|adequate|proper|correct)\b',
            r'\b(?:user-friendly|intuitive|easy|simple)\b',
            r'\b(?:fast|slow|quick|responsive)\b'
        ]

    def extract_requirements(self, file_path: str) -> List[ExtractedRequirement]:
        """
        Main extraction method that orchestrates the three-step process.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of extracted requirements
        """
        logger.info(f"Starting requirement extraction for: {file_path}")
        
        try:
            # Step 1: Chunk the document
            chunks = self._chunk_document(file_path)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 2: Extract requirements using LLM
            all_requirements = []
            for chunk in chunks:
                requirements = self._extract_from_chunk(chunk)
                all_requirements.extend(requirements)
            
            logger.info(f"Extracted {len(all_requirements)} requirements")
            
            # Step 3: Apply regex helpers
            processed_requirements = self._post_process_requirements(all_requirements)
            
            logger.info(f"Post-processed {len(processed_requirements)} requirements")
            return processed_requirements
            
        except Exception as e:
            logger.error(f"Error extracting requirements: {e}")
            return []

    def _chunk_document(self, file_path: str) -> List[DocumentChunk]:
        """
        Step 1: Chunk the document while preserving structure.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks
        """
        try:
            # Use unstructured to parse the document
            elements = partition(filename=file_path)
            
            # Group elements by title/section for better chunking
            chunks = chunk_by_title(
                elements,
                max_characters=self.max_chunk_chars,
                new_after_n_chars=self.max_chunk_chars // 2,
                combine_text_under_n_chars=100
            )
            
            document_chunks = []
            for idx, chunk in enumerate(chunks):
                # Extract content and metadata
                content = str(chunk)
                
                # Try to identify the source section
                source_hint = self._extract_source_hint(chunk, idx)
                
                document_chunks.append(DocumentChunk(
                    content=content,
                    source_hint=source_hint,
                    chunk_index=idx,
                    char_count=len(content)
                ))
                
            return document_chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            return []

    def _extract_source_hint(self, chunk, chunk_index: int) -> str:
        """Extract source hint from chunk metadata or content."""
        try:
            # Look for title or heading in the chunk
            chunk_text = str(chunk)
            lines = chunk_text.split('\n')
            
            # Find potential headings (short lines, often uppercase/title case)
            for line in lines[:3]:  # Check first 3 lines
                line = line.strip()
                if len(line) < 80 and line and not line.endswith('.'):
                    # Could be a heading
                    return f"Section: {line}"
                    
            # Fallback to chunk position
            return f"Chunk {chunk_index + 1}"
            
        except Exception:
            return f"Chunk {chunk_index + 1}"

    def _extract_from_chunk(self, chunk: DocumentChunk) -> List[ExtractedRequirement]:
        """
        Step 2: Extract requirements from a chunk using LLM.
        
        Args:
            chunk: Document chunk to process
            
        Returns:
            List of extracted requirements
        """
        if not self.llm_provider:
            logger.warning("No LLM provider configured, using fallback extraction")
            return self._fallback_extraction(chunk)
            
        try:
            prompt = self._create_extraction_prompt(chunk.content)
            
            # Call LLM (this will depend on your LLM provider)
            response = self._call_llm(prompt)
            
            # Parse JSON response
            requirements_data = self._parse_llm_response(response)
            
            # Convert to ExtractedRequirement objects
            requirements = []
            for req_data in requirements_data.get('requirements', []):
                requirement = ExtractedRequirement(
                    id=req_data.get('id'),
                    text=req_data.get('text', ''),
                    type_hint=req_data.get('type_hint', 'Unknown'),
                    source_hint=req_data.get('source_hint', chunk.source_hint),
                    chunk_index=chunk.chunk_index
                )
                requirements.append(requirement)
                
            return requirements
            
        except Exception as e:
            logger.error(f"Error extracting from chunk {chunk.chunk_index}: {e}")
            return self._fallback_extraction(chunk)

    def _create_extraction_prompt(self, content: str) -> str:
        """Create the extraction prompt with JSON schema."""
        return f"""You are a requirements extraction assistant.
From the text below, extract only actual requirements.
A requirement is a statement that describes behavior, constraints, or qualities the system or component shall/should/must satisfy.

For each requirement, return:
- id: exact ID if present (e.g., SYS-007), otherwise null
- text: the requirement sentence(s) only, no headings or bullets, copied verbatim except you may join wrapped lines  
- type_hint: "System", "Component", "Test" or "Unknown"
- source_hint: a short string to help locate it (e.g., section/heading text or line number if mentioned)

Do not invent requirements. If you are not sure, omit it.
Return JSON only:
{{"requirements": [{{"id": "...", "text": "...", "type_hint": "...", "source_hint": "..."}}, ...]}}

TEXT TO ANALYZE:
{content}
"""

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM provider with the prompt.
        This method should be implemented based on your LLM provider.
        """
        if hasattr(self.llm_provider, 'chat') and hasattr(self.llm_provider.chat, 'completions'):
            # OpenAI-style interface
            try:
                response = self.llm_provider.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a requirements extraction assistant. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                raise
        else:
            # Generic interface
            try:
                return self.llm_provider.generate(prompt)
            except Exception as e:
                logger.error(f"LLM provider error: {e}")
                raise

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM JSON response."""
        try:
            # Clean the response (remove potential markdown formatting)
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response[:500]}...")
            return {"requirements": []}

    def _fallback_extraction(self, chunk: DocumentChunk) -> List[ExtractedRequirement]:
        """
        Fallback extraction method when LLM is not available.
        Uses basic heuristics to identify potential requirements.
        """
        requirements = []
        content = chunk.content
        
        # Split into sentences/lines
        sentences = re.split(r'[.!?]\s+', content)
        
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            
            # Look for requirement indicators
            req_indicators = [
                r'\b(?:shall|should|must|will|may)\b',
                r'\b(?:requires?|specifies?|defines?)\b',
                r'\b(?:system|component|software|application)\s+(?:shall|should|must|will)\b'
            ]
            
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in req_indicators):
                # Extract potential ID
                req_id = None
                for pattern in self.id_patterns:
                    match = re.search(pattern, sentence)
                    if match:
                        req_id = match.group(1)
                        break
                
                requirements.append(ExtractedRequirement(
                    id=req_id,
                    text=sentence,
                    type_hint="Unknown",
                    source_hint=chunk.source_hint,
                    chunk_index=chunk.chunk_index,
                    confidence=0.6  # Lower confidence for fallback
                ))
        
        return requirements

    def _post_process_requirements(self, requirements: List[ExtractedRequirement]) -> List[ExtractedRequirement]:
        """
        Step 3: Apply regex helpers for post-processing.
        
        Args:
            requirements: List of extracted requirements
            
        Returns:
            Post-processed requirements
        """
        processed = []
        
        for req in requirements:
            # Extract ID if missing
            if not req.id:
                for pattern in self.id_patterns:
                    match = re.search(pattern, req.text)
                    if match:
                        req.id = match.group(1)
                        break
            
            # Detect quality issues
            issues = []
            
            # TBD detection
            if re.search(self.tbd_pattern, req.text, re.IGNORECASE):
                issues.append("TBD")
            
            # Passive voice detection
            if any(re.search(pattern, req.text, re.IGNORECASE) for pattern in self.passive_voice_patterns):
                issues.append("PassiveVoice")
            
            # Vague terms detection
            if any(re.search(pattern, req.text, re.IGNORECASE) for pattern in self.vague_terms):
                issues.append("Ambiguous")
            
            # Add issues to the requirement (you might want to store this differently)
            req.quality_issues = issues
            
            processed.append(req)
        
        return processed

def create_extractor(llm_provider=None, max_chunk_chars: int = 3000) -> ModelBasedExtractor:
    """
    Factory function to create a ModelBasedExtractor instance.
    
    Args:
        llm_provider: LLM provider instance
        max_chunk_chars: Maximum characters per chunk
        
    Returns:
        ModelBasedExtractor instance
    """
    return ModelBasedExtractor(max_chunk_chars=max_chunk_chars, llm_provider=llm_provider)