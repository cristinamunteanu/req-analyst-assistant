from types import SimpleNamespace
from unittest.mock import Mock, patch

from analysis.rewrites import REWRITE_PROMPT, suggest_rewrites


class TestSuggestRewrites:
    """Tests for suggest_rewrites LLM prompt construction and return handling."""

    @patch("analysis.rewrites.make_llm")
    def test_suggest_rewrites_builds_notes_and_invokes_llm(self, mock_make_llm):
        """Uses sorted unique notes from mixed issue types and calls LLM with prompt."""
        llm = Mock()
        llm.invoke.return_value = "Improved requirement"
        mock_make_llm.return_value = llm

        class Issue:
            def __init__(self, note):
                self.note = note

        text = "The system shall respond quickly."
        issues = [
            Issue("Ambiguous time"),
            {"note": "Missing metric"},
            Issue(""),
            {"note": None},
            {"other": "ignored"},
            Issue("Ambiguous time"),
        ]

        result = suggest_rewrites(text, issues)

        assert result == "Improved requirement"
        expected_notes = "Ambiguous time; Missing metric"
        expected_prompt = REWRITE_PROMPT.format(
            issue_notes=expected_notes,
            requirement=text
        )
        llm.invoke.assert_called_once_with(expected_prompt, temperature=0)

    @patch("analysis.rewrites.make_llm")
    def test_suggest_rewrites_handles_no_notes_and_content_result(self, mock_make_llm):
        """Falls back to N/A notes and returns content attribute when provided."""
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content="Refined requirement")
        mock_make_llm.return_value = llm

        text = "TBD: define latency requirement."
        result = suggest_rewrites(text, issues=[])

        assert result == "Refined requirement"
        expected_prompt = REWRITE_PROMPT.format(issue_notes="N/A", requirement=text)
        llm.invoke.assert_called_once_with(expected_prompt, temperature=0)
