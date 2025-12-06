# 📘 Req-Analyst-Assistant

An AI-powered assistant for requirements analysis: clarity checks, ambiguity detection, test-case suggestions, and traceability support.  
This tool helps engineers, analysts, and auditors improve the quality of requirements early in the development lifecycle.

---

## 🧾 Overview

Req-Analyst-Assistant streamlines the review of system and software requirements.  
Upload or paste your requirement set, and the assistant will:

- Detect ambiguous, unclear, or incomplete requirements  
- Flag vague or risky wording  
- Suggest test cases (nominal, edge, and failure scenarios)  
- Highlight missing conditions, missing actors, or circular references  
- Support traceability between requirements when applicable  

This improves requirement quality, reduces rework, and accelerates compliance workflows.

---

## ✨ Features

### 🔍 Clarity & Ambiguity Checks
- Identifies vague terms, missing actors, unclear conditions, weak verbs, etc.  
- Flags requirements likely to cause misinterpretation.

### 🧪 Test Case Suggestion
- Automatically generates structured test scenarios:
  - Nominal cases  
  - Edge cases  
  - Failure scenarios  
- Includes Gherkin-style *Given / When / Then* options.

### 🧷 Traceability Support
- Detects requirement cross-references.  
- Warns about circular dependencies.

### 📄 Multi-Format Ingestion
- Supports requirements entered as text or uploaded via document ingestion pipelines (depending on implementation).

### 🖥️ Simple, Interactive UI
- Clean interface for requirement submission and results review.

---

## 📁 Project Structure

```text
req-analyst-assistant/
│
├── ingestion/                 # Parsing and loading of requirements
├── analysis/                  # Ambiguity checks, clarity scoring, test suggestion logic
├── ui/                        # Application UI (e.g., Streamlit components)
├── tests/                     # Unit and integration tests
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
