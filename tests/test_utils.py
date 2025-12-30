from types import SimpleNamespace

from analysis.utils import (
    _find_starts,
    _clean_block,
    _strip_inline_header_tail,
    split_into_requirements,
    is_requirement,
    parse_llm_content,
    parse_requirement,
    is_valid_req_id,
    analyze_dependencies,
)


class TestFindStarts:
    def test_find_starts_orders_requirements_and_headers(self):
        text = "\n".join(
            [
                "1.1 System Requirements",
                "SYS-001: The system shall boot.",
                "• CMP-002: The component shall log.",
            ]
        )
        starts = _find_starts(text)
        kinds = [kind for _, kind, _ in starts]

        assert kinds == ["hdr", "req", "req"]


class TestCleanBlock:
    def test_clean_block_joins_wrapped_lines(self):
        block = "SYS-001: The system shall\n   boot quickly.\n\n"
        assert _clean_block(block) == "SYS-001: The system shall boot quickly."


class TestStripInlineHeaderTail:
    def test_strips_header_after_punctuation(self):
        cleaned = "SYS-001: Log events. 3.2 Navigation Engine"
        assert _strip_inline_header_tail(cleaned) == "SYS-001: Log events."

    def test_does_not_strip_units(self):
        cleaned = "SYS-002: Render at 60 FPS"
        assert _strip_inline_header_tail(cleaned) == cleaned


class TestSplitIntoRequirements:
    def test_split_into_requirements_trims_headers(self):
        text = "\n".join(
            [
                "2. Component Requirements",
                "SYS-010: Store events. 2.1 Logging",
                "CMP-020 (draft): Persist data",
            ]
        )
        reqs = split_into_requirements(text)

        assert reqs == [
            "SYS-010: Store events.",
            "CMP-020 (draft): Persist data",
        ]

    def test_split_into_requirements_handles_no_matches(self):
        assert split_into_requirements("No requirements here.") == []


class TestIsRequirement:
    def test_is_requirement_rejects_headers_and_short_text(self):
        assert not is_requirement("Overview")
        assert not is_requirement("Short line")

    def test_is_requirement_accepts_requirement_like_text(self):
        assert is_requirement("SYS-001: The system shall validate inputs.")


class TestParseLlmContent:
    def test_parse_llm_content_from_string_json(self):
        content = """```json
{"normalized": "Text", "categories": ["functional"]}
```"""
        normalized, categories = parse_llm_content(content)

        assert normalized == "Text"
        assert categories == ["functional"]

    def test_parse_llm_content_from_object(self):
        llm_obj = SimpleNamespace(content='{"normalized": "A", "categories": []}')
        normalized, categories = parse_llm_content(llm_obj)

        assert normalized == "A"
        assert categories == []

    def test_parse_llm_content_fallback(self):
        normalized, categories = parse_llm_content("not-json")

        assert normalized == "not-json"
        assert categories == []


class TestParseRequirement:
    def test_parse_requirement_with_missing_status(self):
        raw = "CMP-SYNC-999: [MISSING] Legacy sync shim."
        parsed = parse_requirement(raw)

        assert parsed["id"] == "CMP-SYNC-999"
        assert parsed["status"] == "missing"
        assert parsed["text"].startswith("[MISSING]")

    def test_parse_requirement_without_id(self):
        parsed = parse_requirement("No requirement ID here.")
        assert parsed == {
            "id": None,
            "status": "undefined",
            "text": "No requirement ID here.",
        }


class TestValidateReqId:
    def test_is_valid_req_id(self):
        assert is_valid_req_id("SYS-100")
        assert not is_valid_req_id("FOO-100")


class TestAnalyzeDependencies:
    def test_analyze_dependencies_missing_and_circular(self):
        requirement_rows = [
            {"Requirement": "SYS-001: Uses CMP-002 and CMP-999."},
            {"Requirement": "CMP-002: Depends on SYS-001."},
            {"Requirement": "TST-003: References FOO-100."},
            {"Requirement": "CMP-004: [MISSING] Placeholder."},
        ]
        missing_refs, circular_refs = analyze_dependencies(requirement_rows)

        assert missing_refs == {"CMP-999"}
        assert circular_refs == {("CMP-002", "SYS-001")}
