import os
import hashlib
import pickle
import re
import json
import traceback
from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from analysis.qa import make_llm

NORMALIZE_PROMPT = """You are a requirements analyst assistant.
Your task is to:
1. Normalize the requirement into a short, abstract statement of maximum 5 words.(remove IDs, numbers, device specifics).
2. Categorize the requirement into one or more categories: 
   [Functional, Performance, Security, Usability, Reliability, Compliance, Integration, Other].

Requirement:
{chunk}

Return JSON with keys: "normalized", "categories".
"""


def _hash_doc(text: str, embed_model: str) -> str:
    """
    Generate a SHA256 hash for a document chunk, combining the embedding model name and the text.

    Args:
        text (str): The text content of the document chunk.
        embed_model (str): The embedding model name (used to ensure cache separation by model).

    Returns:
        str: A hexadecimal SHA256 hash string uniquely identifying the (embed_model, text) pair.

    Raises:
        ValueError: If either text or embed_model is not a string or is empty.

    Notes:
        This hash is used as a cache key to avoid redundant LLM summarization for the same chunk
        and embedding model combination.
    """
    if not isinstance(text, str) or not isinstance(embed_model, str):
        raise ValueError("Both 'text' and 'embed_model' must be strings.")
    if not text.strip():
        raise ValueError("'text' must be a non-empty string.")
    if not embed_model.strip():
        raise ValueError("'embed_model' must be a non-empty string.")

    h = hashlib.sha256()
    h.update(embed_model.encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()

def split_into_requirements(text: str) -> list:
    # Example: splits on lines starting with a number or bullet
    # Adjust the regex to match your requirements format
    parts = re.split(r'(?m)^\s*(?:\d+\.|\-|\*)\s+', text)
    # Remove empty and very short parts
    return [p.strip() for p in parts if len(p.strip()) > 10]


def parse_normalized_requirement_response(s):
    """
    Attempts to robustly parse a normalized requirement and its categories from a string
    returned by an LLM. Handles plain JSON, Markdown code blocks (```json ... ```), and
    legacy debug formats (e.g., content='```json ... ```').

    Args:
        s (str): The string to parse, typically the raw response from an LLM.

    Returns:
        dict: A dictionary with keys "normalized" (str) and "categories" (list).
              If parsing fails, returns {"normalized": s, "categories": ["Other"]} as a fallback.

    Examples:
        >>> parse_normalized_requirement_response('{"normalized": "Login", "categories": ["Functional"]}')
        {'normalized': 'Login', 'categories': ['Functional']}

        >>> parse_normalized_requirement_response('```json\\n{"normalized": "Login", "categories": ["Functional"]}\\n```')
        {'normalized': 'Login', 'categories': ['Functional']}

        >>> parse_normalized_requirement_response("content='```json {\"normalized\": \"Logout\", \"categories\": [\"Functional\"]} ```'")
        {'normalized': 'Logout', 'categories': ['Functional']}

        >>> parse_normalized_requirement_response('Some random text')
        {'normalized': 'Some random text', 'categories': ['Other']}
    """
    s = s.strip()

    def _strip_json_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[len("```json"):]
        elif text.startswith("```"):
            text = text[len("```"):]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    # 1) Try direct JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) Try to find any ```json ... ``` code block inside the text
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", s, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    # 3) Strip fences if the whole response is fenced
    stripped = _strip_json_fence(s)
    if stripped != s:
        try:
            return json.loads(stripped)
        except Exception:
            pass

    # 4) Try regex fallback (legacy, for content='```json ... ```')
    match = re.search(r"content='(```json.*?```)'", s, re.DOTALL)
    if match:
        content = _strip_json_fence(match.group(1))
        try:
            return json.loads(content)
        except Exception:
            pass

    return {"normalized": s, "categories": ["Other"]}


def normalize_requirements(
    docs: List[Dict[str, Any]],
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: str = ".normalize_cache"
) -> List[Dict[str, Any]]:
    """
    Normalize and categorize requirements using an LLM.
    Caches outputs by (embed_model, doc_hash) to avoid redundant LLM calls.
    Returns: List of dicts: [{id, source, text, normalized, categories}]
    """
    cache_enabled = True
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        cache_enabled = False

    llm = make_llm()
    prompt = PromptTemplate(input_variables=["chunk"], template=NORMALIZE_PROMPT)
    results = []

    for i, d in enumerate(docs):
        try:
            text = d.get("text", "")
            source = d.get("source") or d.get("path", "unknown")
            doc_hash = _hash_doc(text, embed_model)
            cache_path = os.path.join(cache_dir, f"{doc_hash}.pkl")

            result = None
            if cache_enabled:
                try:
                    if os.path.exists(cache_path):
                        with open(cache_path, "rb") as f:
                            result = pickle.load(f)
                except OSError:
                    result = None

            if result is None:
                chain_input = prompt.format(chunk=text)
                response = llm.invoke(chain_input, temperature=0)
                raw_response = (
                    response
                    if isinstance(response, str)
                    else getattr(response, "content", str(response)).strip()
                )
                result = parse_normalized_requirement_response(raw_response)

                if cache_enabled:
                    try:
                        with open(cache_path, "wb") as f:
                            pickle.dump(result, f)
                    except OSError:
                        pass

            results.append({
                "id": doc_hash,
                "source": source,
                "text": text,
                "normalized": result.get("normalized", ""),
                "categories": result.get("categories", []),
            })

        except Exception as e:
            print(f"[normalize_requirements] Fatal error processing doc {i}: {e}")
            print(traceback.format_exc())
            results.append({
                "id": f"error_{i}",
                "source": d.get("source") or d.get("path", "unknown"),
                "text": d.get("text", ""),
                "normalized": f"ERROR: {e}",
                "categories": []
            })

    return results
