import re
import json

def split_into_requirements(text: str) -> list:
    """
    Splits a requirements document into individual requirements.
    Splits on lines starting with a bullet and a requirement ID (e.g., • SYS-001: ...).
    """
    parts = re.split(r'(?m)^\s*(?:•\s*)?(?=[A-Z]{2,5}(?:-[A-Z]{2,5})?-\d{3,})', text)
    return [p.strip() for p in parts if len(p.strip()) > 10]

def is_requirement(chunk: str) -> bool:
    """
    Returns True if the chunk looks like a real requirement, False if it's a section header or metadata.
    """
    # Exclude if chunk is very short or matches a known section header
    if len(chunk.strip()) < 20:
        return False
    lowered = chunk.strip().lower()
    non_req_keywords = [
        "scope", "introduction", "overview", "purpose", "document id", "version", "date", "table of contents",
        "requirements", "system requirements", "non-functional requirements", "functional requirements"
    ]
    # Exclude if chunk is just a section header
    if any(lowered.startswith(word) for word in non_req_keywords):
        return False
    return True

def parse_llm_content(llm_obj):
    """
    Extracts normalized text and categories from an LLM response object or string.
    Handles both AIMessage and str, and removes code block markers if present.
    Returns (normalized, categories).
    """
    content = getattr(llm_obj, "content", llm_obj)
    try:
        content = content.strip()
        # Remove code block markers if present
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        data = json.loads(content)
        normalized = data.get("normalized", "")
        categories = data.get("categories", [])
        return normalized, categories
    except Exception:
        return str(content), []