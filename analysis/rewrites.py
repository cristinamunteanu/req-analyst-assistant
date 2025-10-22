from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from analysis.qa import make_llm 

REWRITE_PROMPT = """You are a senior requirements engineer.
Rewrite the requirement to be concise, testable, and unambiguous while preserving intent.
Apply these issue notes if relevant:
{issue_notes}

Requirement:
{requirement}

If the requirement contains TBD/XXX/etc., 
do NOT invent details. Instead, rewrite it to clearly flag that a specification is missing 
and provide an example of how to replace it (but mark as illustrative only).

Return ONLY the improved requirement. Do not add commentary.
"""

def suggest_rewrites(text: str, issues: list) -> str:
    """
    Suggests an improved version of a requirement using an LLM, based on detected issues.
    If a TBD marker is present, the LLM is instructed not to invent details, but to rewrite the rest.
    """
    llm = make_llm()
    notes = "; ".join(sorted({
        getattr(i, "note", None) if hasattr(i, "note") else i.get("note", "")
        for i in issues if (getattr(i, "note", None) if hasattr(i, "note") else i.get("note", ""))
    }))
    prompt = PromptTemplate(
        input_variables=["issue_notes", "requirement"],
        template=REWRITE_PROMPT
    )
    completion = llm.invoke(prompt.format(issue_notes=notes or "N/A", requirement=text), temperature=0)
    return completion if isinstance(completion, str) else getattr(completion, "content", str(completion))
