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

def parse_requirement(raw_text):
    """
    Parse a requirement string into its components: ID, status, and main text.

    Args:
        raw_text (str): The raw requirement text.

    Returns:
        dict: {
            "id": str or None,
            "status": str ("missing", "defined", "undefined"),
            "text": str
        }

    Example:
        parse_requirement("CMP-SYNC-999: [MISSING] Legacy sync shim for offline mode. (Not defined in this document.)")
        # {'id': 'CMP-SYNC-999', 'status': 'missing', 'text': '[MISSING] Legacy sync shim for offline mode. (Not defined in this document.)'}
    """
    # Match ID at the start (e.g., CMP-SYNC-999:)
    match = re.match(r"^\s*([A-Z]+(?:-[A-Z]+)*-\d+)(?:\s*\([^)]+\))?\s*:\s*(.*)", raw_text)
    if match:
        req_id = match.group(1)
        rest = match.group(2)
        # Detect missing/placeholder status
        if "[MISSING]" in rest or "Not defined in this document" in rest:
            status = "missing"
        else:
            status = "defined"
        return {"id": req_id, "status": status, "text": rest}
    else:
        # No ID found, treat as undefined
        return {"id": None, "status": "undefined", "text": raw_text}

VALID_PREFIXES = {"CMP", "SYS", "AUTH", "OPS", "TST"}  # Add all valid prefixes

def is_valid_req_id(req_id):
    prefix = req_id.split("-")[0]
    return prefix in VALID_PREFIXES

def analyze_dependencies(requirement_rows):
    """
    Analyze requirement dependencies for missing and circular references.

    This function scans a list of requirement records, each expected to have a "Requirement" field
    containing text with possible references to other requirements by their IDs (e.g., CMP-SEC-601).
    It performs two main checks:
      1. Detects referenced requirement IDs that are missing from the dataset.
      2. Detects circular references, where two requirements reference each other.

    Args:
        requirement_rows (list of dict): Each dict should have a "Requirement" key containing the requirement text.

    Returns:
        missing_refs (set): Requirement IDs that are referenced but not defined in the dataset.
        circular_refs (set of tuples): Tuples of (referenced_id, requirement_id) indicating circular references.

    Example:
        missing_refs, circular_refs = analyze_dependencies(requirement_rows)
        # missing_refs: {'CMP-SEC-999'}
        # circular_refs: {('CMP-SEC-601', 'CMP-SEC-602'), ...}
    """
    from collections import defaultdict
    
    id_to_reqs = defaultdict(list)
    for r in requirement_rows:
        req_text = r.get("Requirement", "")
        parsed = parse_requirement(req_text)
        if parsed["id"] and parsed["status"] == "defined":
            id_to_reqs[parsed["id"]].append(r)

    missing_refs = set()
    id_pattern = re.compile(r"\b([A-Z]+(?:-[A-Z]+)*-\d+)\b")
    
    dependency_map = defaultdict(set)
    for r in requirement_rows:
        req_text = r.get("Requirement", "")
        parsed = parse_requirement(req_text)
        if parsed["id"] and parsed["status"] == "defined":
            refs = id_pattern.findall(req_text)
            refs = [ref for ref in refs if is_valid_req_id(ref) and ref != parsed["id"]]
            dependency_map[parsed["id"]].update(refs)
            for ref in refs:
                if ref not in id_to_reqs:
                    missing_refs.add(ref)

    circular_refs = set()
    for req_id, refs in dependency_map.items():
        for ref in refs:
            if ref in dependency_map and req_id in dependency_map[ref]:
                circular_refs.add(tuple(sorted([req_id, ref])))

    return missing_refs, circular_refs