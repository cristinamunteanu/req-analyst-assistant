from ingestion.loader import load_documents

def test_load_documents_empty(tmp_path, monkeypatch):
    """
    Test that load_documents returns an empty list when the input directory is empty.

    Steps:
        - Create a temporary 'data' directory using pytest's tmp_path fixture.
        - Change the working directory to the temporary path.
        - Call load_documents on the empty 'data' directory.
        - Assert that the result is a list and is empty.

    Args:
        tmp_path: pytest fixture providing a temporary directory unique to the test invocation.
        monkeypatch: pytest fixture for safely patching and restoring objects.
    """
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.chdir(tmp_path)
    docs = load_documents("data")
    assert isinstance(docs, list)
    assert len(docs) == 0
