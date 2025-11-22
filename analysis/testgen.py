# analysis/testgen.py
import re
from typing import List, Dict, Any, Optional, Tuple

# --- Lightweight classifier ---------------------------------------------------

C_FUN = {"shall", "must", "provide", "verify", "expose", "process", "scan", "decode", "authenticate", "route", "recompute"}
C_PERF = {"ms", "s ", "seconds", "latency", "throughput", "p95", "p99", "uptime", "fps", "hz", "rps", "per second"}
C_UX   = {"tap", "taps", "one-handed", "usability", "sus", "accessible", "ui", "interface", "gesture"}
C_SAFE = {"safety", "hazard", "fail-safe", "block", "prevent", "mismatch", "unauthorized", "compliance", "enforce", "lock"}

def _classify(text: str) -> str:
    """
    Classifies a requirement text into one of four categories: functional, performance,
    interface/ux, or safety, based on the presence of category-specific keywords.

    The function counts how many keywords from each category appear in the text and
    assigns the category with the highest count. If no keywords are found, it defaults
    to "functional".

    Args:
        text (str): The requirement text to classify.

    Returns:
        str: The detected category ("functional", "performance", "interface/ux", or "safety").
    """
    t = text.lower()
    score = {"functional": 0, "performance": 0, "interface/ux": 0, "safety": 0}
    score["functional"] += sum(k in t for k in C_FUN)
    score["performance"] += sum(k in t for k in C_PERF)
    score["interface/ux"] += sum(k in t for k in C_UX)
    score["safety"] += sum(k in t for k in C_SAFE)
    # choose max; default functional
    return max(score, key=score.get) if max(score.values()) > 0 else "functional"

# --- Metric extraction (best-effort) ------------------------------------------

NUM_UNIT = re.compile(
    r'(?P<op>≤|>=|≤|≥|<|>|=|==|<=|>=)?\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>ms|s|sec|seconds|FPS|fps|Hz|hz|m|meters|%)',
    flags=re.IGNORECASE
)

def _extract_metrics(text: str) -> List[Tuple[str, str, str]]:
    """
    Extracts all metric expressions with units from the given text.

    This function searches for patterns like "≤ 300 ms", "99.5%", or ">= 60 FPS" and returns
    a list of tuples, each containing the operator (if present), the numeric value, and the unit.

    Args:
        text (str): The input text to search for metric expressions.

    Returns:
        List[Tuple[str, str, str]]: A list of (operator, number, unit) tuples found in the text.
            - operator (str): The comparison operator (e.g., "≤", ">=", "<", etc.), or an empty string if not present.
            - number (str): The numeric value as a string (e.g., "300", "99.5").
            - unit (str): The unit of measurement (e.g., "ms", "%", "FPS").
    
    Example:
        >>> _extract_metrics("The system must respond in ≤ 300 ms and maintain > 99.5% uptime.")
        [('≤', '300', 'ms'), ('>', '99.5', '%')]
    """
    out = []
    for m in NUM_UNIT.finditer(text):
        op = m.group("op") or ""
        num = m.group("num")
        unit = m.group("unit")
        out.append((op, num, unit))
    return out

# --- Template builders ---------------------------------------------------------

def _functional_templates(text: str) -> List[Dict[str, Any]]:
    return [
        {
            "title": "Nominal scenario with valid inputs",
            "steps": [
                "Given a valid precondition or input set",
                "When the user/system performs the described action",
                "Then the expected outcome is produced and persisted (no errors)"
            ],
            "acceptance": [
                "Given valid inputs, When action is performed, Then result matches specification and is visible in downstream interfaces/logs."
            ]
        },
        {
            "title": "Negative path blocks incorrect inputs or states",
            "steps": [
                "Given invalid/mismatched input or unauthorized state",
                "When the action is attempted",
                "Then the system blocks the action and shows a clear error"
            ],
            "acceptance": [
                "Given a mismatch, When confirmation is attempted, Then action is blocked and an explanatory error is shown; no side effects occur."
            ]
        },
        {
            "title": "Resilience / offline or retry behavior (if applicable)",
            "steps": [
                "Given loss of connectivity or dependent service",
                "When the action is performed",
                "Then the event is queued and synced automatically later (no data loss)"
            ],
            "acceptance": [
                "Given no network, When user completes the flow, Then events are queued; When connectivity resumes, Then all events sync within a defined window."
            ]
        }
    ]

def _performance_templates(text: str) -> List[Dict[str, Any]]:
    t = text.lower()
    metrics = _extract_metrics(text)

    # Detect uptime / availability style requirements
    has_uptime = "uptime" in t or "availability" in t
    percent_metrics = [(op, num, unit) for op, num, unit in metrics if unit == "%"]

    if has_uptime and percent_metrics:
        # Use the first percentage metric, ignore the operator in wording
        _, num, unit = percent_metrics[0]
        threshold = f"{num}{unit}"

        return [
            {
                "title": "Monthly uptime SLO",
                "steps": [
                    "Given production monitoring and logging are enabled with a defined uptime SLO",
                    "When uptime is calculated over a full calendar month (excluding approved maintenance windows)",
                    "Then the measured uptime meets or exceeds the agreed threshold"
                ],
                "acceptance": [
                    f"Measured monthly uptime is at least {threshold} over the last full month, excluding approved maintenance windows."
                ],
            },
            {
                "title": "Outage budget / downtime limit",
                "steps": [
                    "Given an agreed error budget derived from the uptime target",
                    "When all outages for a calendar month are summed",
                    "Then total unplanned downtime does not exceed the allowed error budget"
                ],
                "acceptance": [
                    f"Total unplanned downtime per calendar month is within the limit implied by {threshold} uptime."
                ],
            },
            {
                "title": "Failure and recovery monitoring",
                "steps": [
                    "Given a simulated set of service failures during the month",
                    "When the monitoring and alerting system records incidents",
                    "Then each outage is detected, alerted, and included in uptime calculations"
                ],
                "acceptance": [
                    "All outages are recorded with timestamps; calculated uptime from monitoring matches the SLO report within an agreed tolerance."
                ],
            },
        ]

    # ---- Default performance templates (latency / throughput etc.) ----
    # Only consider latency-like metrics for pretty printing (ms / seconds)
    latency_metrics = [
        (op, num, unit)
        for op, num, unit in metrics
        if unit.lower() in ("ms", "s", "sec", "seconds")
    ]
    pretty = (
        ", ".join(
            [f"{num} {unit}" for op, num, unit in latency_metrics]
        )
        if latency_metrics
        else "<target>"
    )

    return [
        {
            "title": "Latency / response-time budget",
            "steps": [
                "Given a reference device/test rig",
                "When the operation is triggered 50–100 times under nominal load",
                "Then measured latency meets or outperforms the threshold",
            ],
            "acceptance": [
                f"p95 latency ≤ {pretty} under nominal load"
                if latency_metrics
                else "p95 latency meets the specified target under nominal load"
            ],
        },
        {
            "title": "Cold/warm start performance",
            "steps": [
                "Given the app/service is cold-started",
                "When the first actionable screen or instruction is requested",
                "Then time-to-first-instruction meets target",
            ],
            "acceptance": [
                "Time-to-first-instruction ≤ target on the reference device (specify device/OS)."
            ],
        },
        {
            "title": "Sustained throughput / stability",
            "steps": [
                "Given sustained activity for N minutes",
                "When operations are executed continuously",
                "Then throughput and error rate stay within bounds",
            ],
            "acceptance": [
                "Throughput ≥ target; error rate ≤ target; no memory leaks or crashes observed."
            ],
        },
    ]

def _ux_templates(text: str) -> List[Dict[str, Any]]:
    taps = None
    m = re.search(r'≤?\s*(\d+)\s*taps?', text, flags=re.IGNORECASE)
    if m: taps = m.group(1)
    return [
        {
            "title": "Task completion efficiency (taps/steps)",
            "steps": [
                "Given a representative user and task",
                "When the user completes the core flow unaided",
                "Then total interactions are within the budget"
            ],
            "acceptance": [
                f"Avg taps per stop ≤ {taps}" if taps else "Avg taps/steps are within the specified budget."
            ]
        },
        {
            "title": "One-handed operability (if specified)",
            "steps": [
                "Given the reference device",
                "When the user performs the task one-handed",
                "Then all targets are reachable and flow is completable"
            ],
            "acceptance": [
                "Reachability and target sizes meet platform HIG; no two-handed requirements."
            ]
        },
        {
            "title": "Usability score",
            "steps": [
                "Given a moderated usability session with N users",
                "When SUS or equivalent survey is conducted post-task",
                "Then mean score meets threshold"
            ],
            "acceptance": [
                "SUS ≥ 75 (or the specified target)."
            ]
        }
    ]

def _safety_templates(text: str) -> List[Dict[str, Any]]:
    return [
        {
            "title": "Incorrect action is blocked",
            "steps": [
                "Given a mismatched or hazardous input/condition",
                "When the user attempts to proceed",
                "Then the system prevents the action and informs the user"
            ],
            "acceptance": [
                "Incorrect confirmation is blocked; a clear, actionable message is shown; no unsafe side effects."
            ]
        },
        {
            "title": "Recovery / fail-safe path",
            "steps": [
                "Given a failure or deviation condition",
                "When the system detects the failure",
                "Then a safe fallback is executed and state remains consistent"
            ],
            "acceptance": [
                "On failure, system transitions to a defined safe state; audit/logs show cause and recovery."
            ]
        },
        {
            "title": "Security/compliance guard (if relevant)",
            "steps": [
                "Given an unauthorized attempt or expired session",
                "When a protected action is requested",
                "Then access is denied and re-authentication is required"
            ],
            "acceptance": [
                "Access control enforced; tokens are handled per policy; sensitive data not exposed."
            ]
        }
    ]

TEMPLATES = {
    "functional": _functional_templates,
    "performance": _performance_templates,
    "interface/ux": _ux_templates,
    "safety": _safety_templates,
}

# --- Public API ----------------------------------------------------------------

def generate_test_ideas(text: str, type_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate high-level test ideas and acceptance criteria skeletons for a requirement.

    This function classifies the requirement into one of four categories
    ("functional", "performance", "interface/ux", or "safety") using either an explicit
    type_hint or a keyword-based classifier. It then generates a list of test ideas
    (with steps and acceptance criteria) using category-specific templates.

    Args:
        text (str): The requirement text for which to generate test ideas.
        type_hint (Optional[str]): Optional explicit category ("functional", "performance",
            "interface/ux", or "safety"). If not provided, the category is inferred.

    Returns:
        Dict[str, Any]: A dictionary with:
            - "type": The detected or provided category.
            - "ideas": A list of dictionaries, each containing:
                - "title": The test idea/scenario.
                - "steps": List of Gherkin-style steps for the test.
                - "acceptance": List of acceptance criteria for the test.
    """
    bucket = (type_hint or _classify(text)).lower()
    bucket = bucket if bucket in TEMPLATES else "functional"
    ideas = TEMPLATES[bucket](text)
    return {"type": bucket, "ideas": ideas}
