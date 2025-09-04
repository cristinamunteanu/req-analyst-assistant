import os
import hashlib
import pickle
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from analysis.qa import make_llm

SUMMARY_PROMPT = """You are a requirements analyst assistant.
Summarize the following requirement or section in 1-2 sentences, focusing on the main intent and any key constraints.
Text:
{chunk}
Summary:"""

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

def summarize_requirements(
    docs: List[Dict[str, Any]],
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: str = ".summarize_cache"
) -> List[Dict[str, Any]]:
    """
    Summarize requirements or document chunks using an LLM.
    Caches summaries by (embed_model, doc_hash) to avoid redundant LLM calls.

    Args:
        docs: List of dicts with at least 'text' and 'source' keys.
        embed_model: Embedding model name (used for cache key).
        cache_dir: Directory for summary cache.

    Returns:
        List of dicts: [{id, source, text, summary}]
    """
    import traceback

    os.makedirs(cache_dir, exist_ok=True)
    llm = make_llm()
    prompt = PromptTemplate(input_variables=["chunk"], template=SUMMARY_PROMPT)
    results = []

    for i, d in enumerate(docs):
        try:
            text = d.get("text", "")
            source = d.get("source") or d.get("path", "unknown")
            doc_hash = _hash_doc(text, embed_model)
            cache_path = os.path.join(cache_dir, f"{doc_hash}.pkl")

            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        summary = pickle.load(f)
                except Exception as e:
                    print(f"[summarize_requirements] Error loading cache for {source}: {e}")
                    summary = None
            else:
                summary = None

            if summary is None:
                try:
                    chain_input = prompt.format(chunk=text)
                    summary = llm.invoke(chain_input, temperature=0)
                    with open(cache_path, "wb") as f:
                        pickle.dump(summary, f)
                except Exception as e:
                    print(f"[summarize_requirements] Error summarizing chunk from {source}: {e}")
                    print(traceback.format_exc())
                    summary = f"ERROR: {e}"

            results.append({
                "id": doc_hash,
                "source": source,
                "text": text,
                "summary": summary.strip() if isinstance(summary, str) else summary,
            })
        except Exception as e:
            print(f"[summarize_requirements] Fatal error processing doc {i}: {e}")
            print(traceback.format_exc())
            results.append({
                "id": f"error_{i}",
                "source": d.get("source") or d.get("path", "unknown"),
                "text": d.get("text", ""),
                "summary": f"ERROR: {e}"
            })

    return results