import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# Set of words/phrases that are considered non-verifiable quantifiers in requirements.
NON_VERIFIABLE_QUANTIFIERS = {
    "fast", "quick", "minimal", "as needed", "as appropriate",
    "acceptable", "reasonable", "best effort", "soon"
}

# Set of markers indicating something is to be decided or determined.
TBD_MARKERS = {"tbd", "t.b.d", "to be decided", "to be determined", "xxx", "???", "tba"}

# Regex pattern for detecting passive voice (simple heuristic).
PASSIVE_VERB = r"\b(?:is|are|was|were|be|been|being)\s+\w+ed\b"

@dataclass
class Issue:
    """
    Represents a clarity issue found in a requirement.
    Attributes:
        type (str): The type/category of the issue (e.g., 'TBD', 'Ambiguous').
        span (str): The text span where the issue was found.
        note (str): Additional note or explanation about the issue.
    """
    type: str
    span: str
    note: str

def _find_terms(text: str, terms: set, issue_type: str, note: str) -> List[Issue]:
    """
    Helper function to find occurrences of specific terms in the text.
    Args:
        text (str): The text to search.
        terms (set): Set of terms to look for.
        issue_type (str): The type of issue to record.
        note (str): Note to include with each issue.
    Returns:
        List[Issue]: List of found issues.
    """
    issues = []
    lowered = text.lower()
    for t in terms:
        # Only flag whole words
        for m in re.finditer(rf"\b{re.escape(t)}\b", lowered):
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 30)
            issues.append(Issue(issue_type, text[start:end], f"{note}: '{t}'"))
    return issues

def _find_tbd(text: str) -> List[Issue]:
    """
    Finds 'to be determined' markers in the text.
    Returns:
        List[Issue]: List of TBD issues found.
    """
    return _find_terms(text, TBD_MARKERS, "TBD", "To be determined marker")

def _find_vague(text: str) -> List[Issue]:
    """
    Finds vague or ambiguous terms in the text.
    Returns:
        List[Issue]: List of ambiguous issues found.
    """
    vague_terms = {"some", "various", "appropriate", "etc", "possibly", "may", "might"}
    return _find_terms(text, vague_terms, "Ambiguous", "Ambiguous/vague term")

def _find_nonverifiable(text: str) -> List[Issue]:
    """
    Finds non-verifiable quantifiers in the text.
    Returns:
        List[Issue]: List of non-verifiable issues found.
    """
    return _find_terms(text, NON_VERIFIABLE_QUANTIFIERS, "NonVerifiable", "Non-verifiable quantifier")

def _find_passive(text: str) -> List[Issue]:
    """
    Finds passive voice constructions in the text.
    Returns:
        List[Issue]: List of passive voice issues found.
    """
    issues = []
    for m in re.finditer(PASSIVE_VERB, text):
        # Check for the presence of an agent ("by ...") after the verb
        tail = text[m.end():m.end() + 30].lower()
        if " by " not in tail:
            start = max(0, m.start() - 20)
            end = min(len(text), m.end() + 20)
            issues.append(Issue("PassiveVoice", text[start:end], "Use active voice with clear actor"))
    return issues

def analyze_clarity(text: str) -> Dict[str, Any]:
    """
    Analyzes the clarity of a requirement text using several heuristics.
    Finds issues such as TBD markers, vague terms, non-verifiable quantifiers, and passive voice.
    Returns a list of issues and a clarity score (0-100).
    Args:
        text (str): The requirement text to analyze.
    Returns:
        dict: {
            "issues": [Issue, ...],
            "clarity_score": int (0..100)
        }
    """
    found: List[Issue] = []
    found += _find_tbd(text)
    found += _find_vague(text)
    found += _find_nonverifiable(text)
    found += _find_passive(text)

    # Scoring (start at 100; subtract weighted penalties)
    score = 100
    weights = {
        "TBD": 25,
        "Ambiguous": 15,
        "NonVerifiable": 15,
        "PassiveVoice": 10
    }
    # De-duplicate same term occurrences in close proximity (less noisy)
    uniques: Dict[Tuple[str, str], Issue] = {}
    for i in found:
        key = (i.type, i.span.lower())
        uniques[key] = i
    for i in uniques.values():
        score -= weights.get(i.type, 5)
    score = max(0, score)
    return {
        "issues": list(uniques.values()),
        "clarity_score": score
    }
