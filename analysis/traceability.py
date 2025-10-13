"""
traceability.py

This module provides functions to extract requirement relationships and build a traceability matrix
from a list of requirement records. It uses regular expressions to identify requirement IDs,
dependencies, and coverage links in requirement text.

Features:
- Extracts requirement IDs from text.
- Parses "Depends on" relationships between requirements.
- Parses "(covers ...)" relationships in test requirements.
- Builds a pandas DataFrame representing the traceability matrix.
- Exports the traceability matrix to CSV.

Functions:
    extract_req_id(text: str) -> str | None
        Extracts the requirement ID from the start of a requirement text.

    parse_dependencies(text: str) -> list[str]
        Finds all requirement IDs referenced in "Depends on" clauses, excluding the current requirement's own ID.

    parse_covers(text: str) -> list[str]
        Extracts IDs from coverage clauses in test requirements, again excluding the current requirement's own ID.

    build_trace_matrix(requirement_rows: List[Dict[str, Any]]) -> pd.DataFrame
        Builds a pandas DataFrame representing the traceability matrix. Each row contains the requirement ID,
        type (Test/System/Component), full text, source, dependencies, coverage links, and reverse coverage ("CoveredBy").

    export_trace_matrix_csv(df: pd.DataFrame, path: str) -> str
        Exports the traceability matrix DataFrame to a CSV file at the given path.
"""

import re
import pandas as pd
from typing import List, Dict, Any

REQ_ID_RE = re.compile(r'^([A-Z]{2,5}(?:-[A-Z]{2,10})?-\d{3,})')
ANY_ID_RE = re.compile(r'[A-Z]{2,5}(?:-[A-Z]{2,10})?-\d{3,}')
COVERS_RE = re.compile(r'\(covers ([^)]+)\)', re.IGNORECASE)

def extract_req_id(text: str) -> str | None:
    """
    Extracts the requirement ID from the start of a requirement text.

    Args:
        text (str): The requirement text.

    Returns:
        str | None: The extracted requirement ID, or None if not found.
    """
    m = REQ_ID_RE.match(text.strip())
    return m.group(1) if m else None

def parse_dependencies(text: str) -> list[str]:
    """
    Finds all requirement IDs referenced in "Depends on" clauses, excluding the current requirement's own ID.

    Args:
        text (str): The requirement text.

    Returns:
        list[str]: List of referenced requirement IDs.
    """
    if "depends on" in text.lower():
        ids = [x for x in ANY_ID_RE.findall(text)]
        rid = extract_req_id(text)
        return [i for i in ids if i != rid]
    return []

def parse_covers(text: str) -> list[str]:
    """
    Extracts IDs from coverage clauses in test requirements, again excluding the current requirement's own ID.

    Args:
        text (str): The requirement text.

    Returns:
        list[str]: List of covered requirement IDs.
    """
    m = COVERS_RE.search(text)
    if not m: return []
    ids = [x.strip() for x in ANY_ID_RE.findall(m.group(1))]
    rid = extract_req_id(text)
    return [i for i in ids if i != rid]

def build_trace_matrix(requirement_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Builds a pandas DataFrame representing the traceability matrix.

    Each row contains:
        - ReqID: Requirement ID
        - Type: "Test", "System", or "Component"
        - Requirement: Full requirement text
        - Source: Source document or file
        - DependsOn: Comma-separated list of dependencies
        - Covers: Comma-separated list of covered requirements
        - CoveredBy: Comma-separated list of requirements that cover this one

    Args:
        requirement_rows (List[Dict[str, Any]]): List of requirement records, each with at least "Requirement" and "Source".

    Returns:
        pd.DataFrame: The traceability matrix.
    """
    rows = []
    for r in requirement_rows:
        txt = r["Requirement"]
        rid = extract_req_id(txt) or ""
        typ = ("Test" if rid.startswith("TST-") else ("System" if rid.startswith("SYS-") else "Component"))
        deps = parse_dependencies(txt)
        covers = parse_covers(txt)
        rows.append({
            "ReqID": rid,
            "Type": typ,
            "Requirement": txt,
            "Source": r.get("Source",""),
            "DependsOn": ", ".join(deps),
            "Covers": ", ".join(covers),
        })
    df = pd.DataFrame(rows)

    # Compute CoveredBy (reverse of Covers)
    covered_by = {}
    for _, row in df.iterrows():
        for k in [x.strip() for x in row["Covers"].split(",") if x.strip()]:
            covered_by.setdefault(k, set()).add(row["ReqID"])

    df["CoveredBy"] = df["ReqID"].map(lambda k: ", ".join(sorted(covered_by.get(k, []))))
    return df

def export_trace_matrix_csv(df: pd.DataFrame, path: str) -> str:
    """
    Exports the traceability matrix DataFrame to a CSV file at the given path.

    Args:
        df (pd.DataFrame): The traceability matrix DataFrame.
        path (str): The file path to export to.

    Returns:
        str: The file path of the exported CSV.
    """
    df.to_csv(path, index=False)
    return path
