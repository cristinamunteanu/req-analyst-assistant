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
"""

import re
import pandas as pd
from typing import List, Dict, Any

REQ_ID_RE = re.compile(r'^([A-Z]{2,5}(?:-[A-Z]{2,10})?-\d{3,})')
ANY_ID_RE = re.compile(r'[A-Z]{2,5}(?:-[A-Z]{2,10})?-\d{3,}')
COVERS_RE = re.compile(r'\(covers ([^)]+)\)', re.IGNORECASE)

def extract_req_ids(text: str) -> list[str]:
    """
    Extracts all requirement IDs from the given text.
    """
    return [x for x in ANY_ID_RE.findall(text)]

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

def parse_relationships(text: str, keyword: str, group: str = None) -> list[str]:
    """
    Generalized function to extract requirement IDs based on a keyword or regex group.

    Args:
        text (str): The requirement text.
        keyword (str): Keyword to look for (e.g., "depends on").
        group (str): Optional regex group to extract IDs from.

    Returns:
        list[str]: List of referenced requirement IDs, excluding the current requirement's own ID.
    """
    rid = extract_req_id(text)
    if group:
        m = COVERS_RE.search(text)
        if not m:
            return []
        ids = extract_req_ids(m.group(1))
    else:
        if keyword.lower() not in text.lower():
            return []
        ids = extract_req_ids(text)
    return [i for i in ids if i != rid]

def parse_dependencies(text: str) -> list[str]:
    """
    Finds all requirement IDs referenced in "Depends on" clauses, excluding the current requirement's own ID.
    """
    return parse_relationships(text, "depends on")

def parse_covers(text: str) -> list[str]:
    """
    Extracts IDs from coverage clauses in test requirements, again excluding the current requirement's own ID.
    """
    return parse_relationships(text, "", group="covers")

def get_req_type(rid: str) -> str:
    """
    Determines the requirement type based on its ID prefix.
    """
    if rid.startswith("TST-"):
        return "Test"
    elif rid.startswith("SYS-"):
        return "System"
    elif rid:
        return "Component"
    return ""

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
        typ = get_req_type(rid)
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
