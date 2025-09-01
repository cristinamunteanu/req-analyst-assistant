from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
