import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


class StopExecution(Exception):
    """Used to short-circuit Streamlit execution during module import."""


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


class DummyBlock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyStreamlit:
    def __init__(self):
        self.session_state = SessionState()
        self.sidebar = DummyBlock()
        self._errors = []
        self._warnings = []
        self._infos = []
        self._writes = []

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop

    def set_page_config(self, **kwargs):
        return None

    def cache_data(self, **kwargs):
        def decorator(func):
            func.clear = lambda: None
            return func
        return decorator

    def tabs(self, labels):
        return [DummyBlock() for _ in labels]

    def columns(self, spec):
        return [DummyBlock() for _ in range(len(spec))]

    def form(self, **kwargs):
        return DummyBlock()

    def expander(self, *args, **kwargs):
        return DummyBlock()

    def container(self):
        return DummyBlock()

    def spinner(self, *args, **kwargs):
        return DummyBlock()

    def button(self, *args, **kwargs):
        return False

    def form_submit_button(self, *args, **kwargs):
        return False

    def file_uploader(self, *args, **kwargs):
        return []

    def text_input(self, *args, **kwargs):
        return ""

    def stop(self):
        raise StopExecution()

    def error(self, msg):
        self._errors.append(msg)

    def warning(self, msg):
        self._warnings.append(msg)

    def info(self, msg):
        self._infos.append(msg)

    def write(self, *args):
        self._writes.append(args)

    def empty(self):
        return DummyBlock()

    def rerun(self):
        raise StopExecution()


@pytest.fixture
def app_module(monkeypatch):
    stub_streamlit = DummyStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", stub_streamlit)

    app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
    spec = importlib.util.spec_from_file_location("ui.app", app_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "ui.app", module)

    try:
        spec.loader.exec_module(module)
    except StopExecution:
        pass

    return module


class TestLoadCss:
    def test_load_css_reads_file(self, app_module, tmp_path):
        css_path = tmp_path / "style.css"
        css_path.write_text("body { color: black; }")

        assert app_module.load_css(str(css_path)) == "body { color: black; }"

    def test_load_css_missing_file(self, app_module, tmp_path):
        missing_path = tmp_path / "missing.css"
        assert app_module.load_css(str(missing_path)) == ""
        assert app_module.st._errors


class TestLog:
    def test_log_respects_debug(self, app_module, monkeypatch):
        mock_print = Mock()
        monkeypatch.setattr("builtins.print", mock_print)

        app_module.DEBUG = False
        app_module.log("nope")
        mock_print.assert_not_called()

        app_module.DEBUG = True
        app_module.log("yes")
        mock_print.assert_called()


class TestProcessUploadedFiles:
    def test_process_uploaded_files_success(self, app_module, monkeypatch, tmp_path):
        def partition(filename):
            return [
                SimpleNamespace(text="Line one."),
                SimpleNamespace(text="Line two."),
            ]

        unstructured = types.ModuleType("unstructured")
        partition_pkg = types.ModuleType("unstructured.partition")
        partition_auto = types.ModuleType("unstructured.partition.auto")
        partition_auto.partition = partition
        monkeypatch.setitem(sys.modules, "unstructured", unstructured)
        monkeypatch.setitem(sys.modules, "unstructured.partition", partition_pkg)
        monkeypatch.setitem(sys.modules, "unstructured.partition.auto", partition_auto)

        class DummyUpload:
            def __init__(self, name, content):
                self.name = name
                self._content = content

            def getvalue(self):
                return self._content

        uploaded = DummyUpload("doc.txt", b"content")
        docs = app_module.process_uploaded_files([uploaded])

        assert len(docs) == 1
        assert docs[0]["name"] == "doc.txt"
        assert docs[0]["text"] == "Line one.\nLine two."
        assert docs[0]["size"] == 7

    def test_process_uploaded_files_no_files(self, app_module):
        assert app_module.process_uploaded_files([]) == []


class TestSessionDocHelpers:
    def test_load_documents_from_session(self, app_module):
        app_module.st.session_state.uploaded_docs = [{"name": "a", "text": "x"}]
        app_module.st.session_state.use_uploaded = False
        assert app_module.load_documents_from_session() == []

        app_module.st.session_state.use_uploaded = True
        assert app_module.load_documents_from_session() == [{"name": "a", "text": "x"}]

    def test_get_uploaded_docs_hash_order_insensitive(self, app_module):
        docs = [
            {"name": "a.txt", "size": 1, "text": "alpha"},
            {"name": "b.txt", "size": 2, "text": "beta"},
        ]
        app_module.st.session_state.uploaded_docs = docs
        first_hash = app_module.get_uploaded_docs_hash()

        app_module.st.session_state.uploaded_docs = list(reversed(docs))
        second_hash = app_module.get_uploaded_docs_hash()

        assert first_hash == second_hash


class TestProviders:
    def test_available_llm_providers_env(self, app_module, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.setenv("HF_TOKEN", "y")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "z")
        monkeypatch.setattr(app_module, "is_installed", lambda pkg: False)

        assert app_module.available_llm_providers() == ["openai", "huggingface", "anthropic"]


class TestShowSources:
    def test_show_sources_ignores_duplicates(self, app_module):
        app_module.show_sources(["doc1", "doc2", "doc2"])
        assert len(app_module.st._writes) == 2
