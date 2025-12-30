from types import SimpleNamespace
from unittest.mock import patch

from ingestion.loader import load_documents


class TestLoadDocuments:
    def test_load_documents_collects_text(self, tmp_path):
        file_path = tmp_path / "doc.txt"
        file_path.write_text("ignored by partition")

        elements = [
            SimpleNamespace(text="Line one."),
            SimpleNamespace(text=""),
            SimpleNamespace(text="Line two."),
        ]

        with patch("ingestion.loader.partition", return_value=elements) as mock_partition:
            docs = load_documents(str(tmp_path))

        assert docs == [{"path": str(file_path), "text": "Line one.\nLine two."}]
        mock_partition.assert_called_once_with(filename=str(file_path))

    def test_load_documents_skips_hidden_and_temp_files(self, tmp_path):
        (tmp_path / ".hidden.txt").write_text("hidden")
        (tmp_path / "~temp.txt").write_text("temp")
        (tmp_path / "file#").write_text("tmp")
        (tmp_path / "~$office.tmp").write_text("office")
        (tmp_path / "real.txt").write_text("real")

        with patch("ingestion.loader.partition", return_value=[SimpleNamespace(text="ok")]) as mock_partition:
            docs = load_documents(str(tmp_path))

        assert docs == [{"path": str(tmp_path / "real.txt"), "text": "ok"}]
        mock_partition.assert_called_once_with(filename=str(tmp_path / "real.txt"))

    def test_load_documents_skips_empty_text(self, tmp_path):
        file_path = tmp_path / "empty.txt"
        file_path.write_text("empty")

        with patch("ingestion.loader.partition", return_value=[SimpleNamespace(text="   ")]) as mock_partition:
            docs = load_documents(str(tmp_path))

        assert docs == []
        mock_partition.assert_called_once_with(filename=str(file_path))

    def test_load_documents_prints_error_and_continues(self, tmp_path, capsys):
        file_path = tmp_path / "bad.txt"
        file_path.write_text("bad")

        with patch("ingestion.loader.partition", side_effect=Exception("boom")):
            docs = load_documents(str(tmp_path))

        assert docs == []
        captured = capsys.readouterr()
        assert f"Error processing {file_path}: boom" in captured.out
