from analysis.index import build_index

def test_build_index_minimal():
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
