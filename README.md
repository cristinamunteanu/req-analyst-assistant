# Req-Analyst-Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()
[![Contributions
Welcome](https://img.shields.io/badge/Contributions-Welcome-orange.svg)]()

An AI-powered assistant for analyzing system & software requirements.\
Helps identify ambiguity, generate test cases, and support traceability
--- ideal for engineering teams working under standards like
**DO-178C**, **ISO 26262**, **IEC 61508**, and similar.

## Overview

Req-Analyst-Assistant streamlines the review and validation of textual
requirements.\
It helps engineers, auditors, and analysts detect issues early by
applying structured clarity checks and AI-powered insights.

Upload or paste your requirement set, and the assistant will
automatically:

-   Detect ambiguous, vague, or incomplete requirements\
-   Highlight unclear actors, weak verbs, untestable statements\
-   Suggest test cases: nominal, edge, and failure scenarios\
-   Flag missing preconditions or unclear outputs\
-   Identify potential circular or missing dependencies\
-   Provide traceability hints

This reduces rework, improves requirement quality, and accelerates
documentation workflows.

## Key Features

### Clarity & Ambiguity Detection

-   Flags weak terms
-   Finds missing conditions or unclear actors
-   Warns about unverifiable statements

### Automated Test Case Generation

Generates structured **Given / When / Then** scenarios.

### Traceability Support

Identifies requirement cross-references.

### Simple UI

Clean interface built with Streamlit.

## Architecture

    +------------------------+
    |   User Interface (UI)  |
    |   Streamlit App        |
    +-----------+------------+
                |
                v
    +------------------------+
    |  Ingestion Layer       |
    +-----------+------------+
                |
                v
    +------------------------+
    |  Analysis Engine       |
    +-----------+------------+
                |
                v
    +------------------------+
    |  Results Export        |
    +------------------------+

## Project Structure

    req-analyst-assistant/
    ├── ingestion/
    ├── analysis/
    ├── ui/
    ├── tests/
    ├── requirements.txt
    └── README.md

## Installation

``` bash
git clone https://github.com/cristinamunteanu/req-analyst-assistant.git
cd req-analyst-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

``` bash
streamlit run ui/app.py
```

## Example Output

    Ambiguity: term 'robust' is vague
    Missing detail: encryption method unspecified
    Suggested Test: Given valid data → When encrypting → Then system uses defined algorithm

## Roadmap

-   CSV/JSON export\
-   Requirement clustering\
-   Local LLM support

## Contributing

Fork → Branch → PR.

## License

MIT License.

## Acknowledgements

OpenAI, Streamlit, and the requirements engineering community.
