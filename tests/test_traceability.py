import pandas as pd

from analysis.traceability import (
    extract_req_ids,
    extract_req_id,
    parse_dependencies,
    parse_covers,
    get_req_type,
    build_trace_matrix,
    export_trace_matrix_csv,
)


class TestExtractReqIds:
    def test_extract_req_ids_finds_all(self):
        text = "SYS-123 depends on CMP-456 and TST-789."
        assert extract_req_ids(text) == ["SYS-123", "CMP-456", "TST-789"]

    def test_extract_req_id_from_start_only(self):
        assert extract_req_id("SYS-001 The system shall boot.") == "SYS-001"
        assert extract_req_id("The system SYS-001 shall boot.") is None


class TestParseRelationships:
    def test_parse_dependencies_excludes_self(self):
        text = "SYS-100 depends on SYS-100 and CMP-200."
        assert parse_dependencies(text) == ["CMP-200"]

    def test_parse_dependencies_requires_keyword(self):
        text = "SYS-101 references CMP-201."
        assert parse_dependencies(text) == []

    def test_parse_covers_parses_group_and_excludes_self(self):
        text = "TST-300 verifies behavior (covers SYS-100, CMP-200, TST-300)."
        assert parse_covers(text) == ["SYS-100", "CMP-200"]

    def test_parse_covers_no_group(self):
        text = "TST-301 verifies behavior without coverage."
        assert parse_covers(text) == []


class TestGetReqType:
    def test_get_req_type(self):
        assert get_req_type("TST-001") == "Test"
        assert get_req_type("SYS-001") == "System"
        assert get_req_type("CMP-001") == "Component"
        assert get_req_type("") == ""


class TestBuildTraceMatrix:
    def test_build_trace_matrix_fields_and_covered_by(self):
        rows = [
            {"Requirement": "SYS-100 System requirement.", "Source": "spec.md"},
            {"Requirement": "CMP-200 depends on SYS-100.", "Source": "spec.md"},
            {"Requirement": "TST-300 verifies feature (covers SYS-100, CMP-200).", "Source": "tests.md"},
        ]

        df = build_trace_matrix(rows)

        assert list(df.columns) == [
            "ReqID",
            "Type",
            "Requirement",
            "Source",
            "DependsOn",
            "Covers",
            "CoveredBy",
        ]
        assert df.loc[0, "ReqID"] == "SYS-100"
        assert df.loc[1, "DependsOn"] == "SYS-100"
        assert df.loc[2, "Covers"] == "SYS-100, CMP-200"
        assert df.loc[0, "CoveredBy"] == "TST-300"
        assert df.loc[1, "CoveredBy"] == "TST-300"


class TestExportTraceMatrixCsv:
    def test_export_trace_matrix_csv(self, tmp_path):
        df = pd.DataFrame(
            [
                {
                    "ReqID": "SYS-100",
                    "Type": "System",
                    "Requirement": "SYS-100 System requirement.",
                    "Source": "spec.md",
                    "DependsOn": "",
                    "Covers": "",
                    "CoveredBy": "",
                }
            ]
        )
        path = tmp_path / "trace.csv"
        exported = export_trace_matrix_csv(df, str(path))

        assert exported == str(path)
        assert path.exists()
