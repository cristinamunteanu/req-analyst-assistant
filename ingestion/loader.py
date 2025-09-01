from pathlib import Path
from typing import List, Dict, Any
from unstructured.partition.auto import partition

def load_documents(input_dir: str = "data") -> List[Dict[str, Any]]:
    """
    Parses all files under the specified input directory into plain text with simple metadata.

    For each file found recursively in the input directory, attempts to extract text content
    using the `partition` function from the unstructured library. If text is successfully
    extracted, a dictionary containing the file path and extracted text is added to the result list.

    Args:
        input_dir (str): Path to the directory containing files to parse. Defaults to "data".

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each with:
            - "path": the file path as a string
            - "text": the extracted plain text from the file

    Notes:
        - Files that cannot be processed or have no extractable text are skipped.
        - Errors during processing are printed to the console.
    """
    docs: List[Dict[str, Any]] = []
    for path in Path(input_dir).glob("**/*"):
        if not path.is_file():
            continue
        try:
            elements = partition(filename=str(path))
            text = "\n".join([getattr(el, "text", "") for el in elements if getattr(el, "text", "")])
            if text.strip():
                docs.append({"path": str(path), "text": text})
        except Exception as e:
            print(f"Error processing {path}: {e}")
    return docs
