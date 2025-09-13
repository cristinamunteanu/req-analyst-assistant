import re
import json
from typing import List

REQ_ID = r'[A-Z]{2,5}(?:-[A-Z]{2,10})?-\d{3,}'

# Requirement start: allow bullets and optional (...) before colon
REQ_START_RE = re.compile(rf'(?m)^\s*(?:[•\-\*]\s*)?({REQ_ID})(?:\s*\([^)]+\))*\s*:', re.MULTILINE)

# Numbered header at start of a line only (hard stop)
HDR_START_LINE_RE = re.compile(r'(?m)^\s*\d+(?:\.\d+)*\s+[A-Z]')

# Inline header tail stripper: only after a sentence terminator, to avoid nuking "≥ 10 FPS", "≤ 300 ms", etc.
INLINE_HEADER_AFTER_PUNCT_RE = re.compile(
    r'(?<=[:.;])\s+\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z][^\n]*$'
)

def _find_starts(text: str):
    """
    Identifies all hard boundaries in the text that mark the start of either a requirement
    or a numbered section header.

    This function scans the input text for:
      - Requirement lines (using REQ_START_RE), such as "SYS-001: ..." or "• CMP-APP-204: ..."
      - Numbered section headers (using HDR_START_LINE_RE), such as "3. Component Requirements"

    Returns a sorted list of tuples, each containing:
      (start_index, kind, match_object)
    where:
      - start_index: the character index in the text where the match starts
      - kind: "req" for requirement, "hdr" for header
      - match_object: the regex match object for further inspection

    Args:
        text (str): The full document text to scan.

    Returns:
        List[Tuple[int, str, re.Match]]: Sorted list of all requirement and header start positions.
    """
    starts = []
    # Find all requirement line starts
    for m in REQ_START_RE.finditer(text):
        starts.append((m.start(), "req", m))
    # Find all header line starts
    for m in HDR_START_LINE_RE.finditer(text):
        starts.append((m.start(), "hdr", m))
    # Sort all found starts by their position in the text
    starts.sort(key=lambda x: x[0])
    return starts

def _clean_block(block: str) -> str:
    """
    Cleans a block of requirement text by:
      - Joining lines that may have been wrapped in the source document
      - Collapsing multiple spaces and newlines into a single space
      - Stripping leading and trailing whitespace

    Args:
        block (str): The raw requirement block.

    Returns:
        str: The cleaned, single-line requirement text.
    """
    block = re.sub(r'\s*\n\s*', ' ', block)   # join line wraps
    block = re.sub(r'\s+', ' ', block).strip()
    return block

def _strip_inline_header_tail(cleaned: str) -> str:
    """
    Removes a trailing inline section header from a requirement line, but only if it appears
    after a sentence terminator (period, semicolon, or colon).

    This is useful for cleaning up lines where a section header (e.g., "3.2 Navigation Engine")
    was appended to the end of a requirement statement, such as:
        "The system shall log all actions. 3.2 Navigation Engine"
    The function ensures that only the requirement text remains, improving clarity and downstream processing.
    It does NOT remove units like "10 FPS" or "300 ms".

    Args:
        cleaned (str): The cleaned, single-line requirement text.

    Returns:
        str: The text with any trailing inline section header removed, or the original text if no match is found.
    """
    # Remove inline section title ONLY if it comes after ., ;, or :
    m = INLINE_HEADER_AFTER_PUNCT_RE.search(cleaned)
    if m:
        return cleaned[:m.start()].rstrip()
    return cleaned

def split_into_requirements(text: str) -> List[str]:
    """
    From each requirement start, capture until the next requirement or the next
    line-start header; then trim any inline header tail that appears after a
    sentence terminator. Safe for units like '10 FPS', '300 ms', '5 Hz'.
    """
    starts = _find_starts(text)
    if not starts:
        return []

    reqs: List[str] = []
    for i, (pos, kind, m) in enumerate(starts):
        if kind != "req":
            continue
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        block = text[pos:end]
        cleaned = _clean_block(block)
        cleaned = _strip_inline_header_tail(cleaned)
        if REQ_START_RE.match(cleaned):
            reqs.append(cleaned)
    return reqs

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