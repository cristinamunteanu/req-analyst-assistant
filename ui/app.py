import os
import io
import json
import importlib
import re

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from ingestion.loader import load_documents
from analysis.index import build_index
from analysis.qa import make_qa
from analysis.normalize_requirements import normalize_requirements
from analysis.utils import split_into_requirements, is_requirement, parse_llm_content, analyze_dependencies
from analysis.heuristics import analyze_clarity
from analysis.rewrites import suggest_rewrites
from analysis.testgen import generate_test_ideas


st.set_page_config(page_title="RAG MVP", page_icon="🔎", layout="wide")
load_dotenv()

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if "embed_model" not in st.session_state:
    st.session_state["embed_model"] = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def _get_embed_model():
    return os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_index():
    with st.spinner("Parsing & indexing documents…"):
        docs = load_documents("data")
        print(f"Loaded documents: {docs}")
        return build_index(docs, embed_model=_get_embed_model())

def is_installed(pkg):
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False

def available_llm_providers():
    providers = []
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("HF_TOKEN") or is_installed("transformers"):
        providers.append("huggingface")
    if os.getenv("ANTHROPIC_API_KEY") or is_installed("anthropic"):
        providers.append("anthropic")
    if is_installed("ollama"):
        providers.append("ollama")
    return providers

# --- Add a log buffer ---
log_buffer = io.StringIO()

def log(msg):
    print(msg)
    log_buffer.write(str(msg) + "\n")


st.title("🔎 RAG MVP")
st.caption("Streamlit UI • LangChain • Unstructured • OpenAI/HF")

# At the very top of your Streamlit app (after st.title or st.caption), add an anchor:
st.markdown('<div id="top"></div>', unsafe_allow_html=True)

st.markdown("""
    <style>
    pre {
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Logs expander in sidebar ---
with st.sidebar.expander("Logs", expanded=False):
    st.code(log_buffer.getvalue() or "No logs yet.", language="text")

# --- Back to top button in sidebar ---
with st.sidebar:
    st.markdown(
        """
        <a href="#top" style="text-decoration: none;">
            <button style="background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px; padding: 0.5em 1em; font-size: 1em; cursor: pointer; width: 100%;">
                ⬆️ Back to top
            </button>
        </a>
        """,
        unsafe_allow_html=True,
    )

# --- Tabbed layout ---
tab_search, tab_summaries, tab_quality, tab_tests = st.tabs(["Search", "Summaries", "Quality", "Test ideas"])


with tab_search:
    st.markdown("## 🔍 Search")
    col1, col2 = st.columns([3, 1])
    with col2:
        llm_options = available_llm_providers()
        if not llm_options:
            st.warning("No LLM providers available. Please check your environment variables and dependencies.")

    index = get_index()
    if index is None:
        st.error("Failed to build the document index. Please check your embedding model, input data, and logs for errors.")
        log("Failed to build the document index. Please check your embedding model, input data, and logs for errors.")
        st.stop()

    try:
        retriever = index.as_retriever(search_kwargs={"k": 4})
        log(f"Retriever created: {retriever}")
    except Exception as e:
        st.error(f"Failed to create retriever: {e}")
        log(f"Failed to create retriever: {e}")
        st.stop()

    qa = make_qa(retriever)
    log(f"QA chain created: {qa}")
    if qa is None:
        st.error("QA chain was not created. Please check your retriever and LLM setup.")
        log("QA chain was not created. Please check your retriever and LLM setup.")
        st.stop()

    query = col1.text_input("Ask a question about the files in `data/`")

    # Set the flag to update the answer only if the query is new
    if query and st.session_state.get("last_query") != query:
        st.session_state["should_update_answer"] = True
        st.session_state["last_query"] = query

    if query:
        try:
            with st.spinner("Thinking…"):
                log(f"Calling QA chain with: {query}")
                if st.session_state.get("should_update_answer", True):
                    # Generate the answer
                    out = qa({"query": query})
                    st.session_state["answer"] = out.get("result", "No answer returned.")
                    st.session_state["sources"] = [d.metadata.get("source", "unknown") for d in out.get("source_documents", [])]
                    st.session_state["should_update_answer"] = False  # Reset after generating
                    log(f"QA chain output: {out}")  # <-- Only log here
                # No else: don't use 'out' outside this block

            # Before your answer section, reset feedback if a new query is entered
            if "last_query" not in st.session_state or st.session_state["last_query"] != query:
                st.session_state["last_query"] = query

            with st.container():
                st.markdown("### 🟩 Answer")
                answer = st.session_state.get("answer", "No answer returned.")

                # Revert to plain markdown bullets for answer display
                if isinstance(answer, list):
                    for req in answer:
                        desc = req.get("description", "")
                        st.markdown(f"- {desc}")
                else:
                    st.markdown(answer)

            with st.container():
                st.markdown("### 📄 Sources")
                sources = st.session_state.get("sources", [])
                if len(sources) > 3:
                    with st.expander("View Sources"):
                        for i, src in enumerate(sources):
                            if (src.startswith("data/") and os.path.exists(src)):
                                with open(src, "rb") as f:
                                    st.download_button(
                                        label=f"Download {os.path.basename(src)}",
                                        data=f,
                                        file_name=os.path.basename(src),
                                        mime="application/octet-stream",
                                        key=f"download_{i}_{os.path.basename(src)}"  # <-- unique key
                                    )
                            else:
                                st.write("•", src)
                else:
                    for i, src in enumerate(sources):
                        if (src.startswith("data/") and os.path.exists(src)):
                            with open(src, "rb") as f:
                                st.download_button(
                                    label=f"Download {os.path.basename(src)}",
                                    data=f,
                                    file_name=os.path.basename(src),
                                    mime="application/octet-stream",
                                    key=f"download_{i}_{os.path.basename(src)}"  # <-- unique key
                                )
                        else:
                            st.write("•", src)
        except Exception as e:
            st.error(f"An error occurred while processing your question: {e}")
            import traceback
            tb = traceback.format_exc()
            log(f"Exception in QA call: {e}\n{tb}")
            st.text(tb)

with tab_summaries:
    st.markdown("## 📋 Requirement Normalization & Categorization")
    try:
        docs = load_documents("data")
        requirement_chunks = []
        for doc in docs:
            requirements = split_into_requirements(doc["text"])
            for req in requirements:
                if is_requirement(req):
                    requirement_chunks.append({
                        "text": req,
                        "source": doc.get("source") or doc.get("path", "unknown")
                    })

        results = normalize_requirements(requirement_chunks)
        if results:
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Source": r["source"],
                    "Requirement": r["text"],
                    "Normalized": r["normalized"],
                    "Categories": ", ".join(r["categories"]),
                }
                for r in results
            ])
            st.dataframe(df, use_container_width=True)
            for r in results:
                print("NORMALIZED FIELD:", repr(r["normalized"]))
                break
        else:
            st.info("No requirements processed.")
    except Exception as e:
        st.error(f"Failed to process requirements: {e}")

with tab_quality:
    st.markdown("## 🧹 Clarity & Ambiguity Checks")
    try:
        docs = load_documents("data")
        requirement_rows = []
        for doc in docs:
            for req in split_into_requirements(doc["text"]):
                if is_requirement(req):
                    result = analyze_clarity(req)
                    requirement_rows.append({
                        "Source": doc.get("source") or doc.get("path", "unknown"),
                        "Requirement": req,
                        "ClarityScore": result["clarity_score"],
                        "Issues": result["issues"]
                    })

        if not requirement_rows:
            st.info("No requirements detected.")
            st.stop()

        st.divider()
        st.caption("### Filters")
        colA, colB, colC, colD = st.columns(4)
        f_amb = colA.toggle("Ambiguous", value=True)
        f_pass = colB.toggle("Passive voice", value=True)
        f_tbd = colC.toggle("TBD", value=True)
        f_nonv = colD.toggle("Non-verifiable", value=True)

        def pass_filters(issues):
            types = {i.type for i in issues}
            if not f_amb and "Ambiguous" in types: return False
            if not f_pass and "PassiveVoice" in types: return False
            if not f_tbd and "TBD" in types: return False
            if not f_nonv and "NonVerifiable" in types: return False
            return True

        filtered = [r for r in requirement_rows if pass_filters(r["Issues"])]

        # --- Dependency analysis BEFORE details loop ---
        missing_refs, circular_refs = analyze_dependencies(requirement_rows)

        st.divider()
        st.markdown("### Table View")
        df = pd.DataFrame([
            {
                "Clarity": r["ClarityScore"],
                "Requirement": r["Requirement"],
                "Issues": ", ".join(sorted({i.type for i in r["Issues"]})) or "—",
                "Source": r["Source"],
                "Details": (
                    f'<a href="#req-{abs(hash(r["Requirement"]))}">🔎 Details & Rewrite</a>'
                    if r["ClarityScore"] < 100 else ""
                ),
            }
            for r in filtered
        ]).sort_values(by=["Clarity", "Issues"], ascending=[True, True])

        st.write(
            df.to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("### Details & Suggested Rewrites")
        for idx, r in enumerate(filtered):
            st.markdown(f'<div id="req-{abs(hash(r["Requirement"]))}"></div>', unsafe_allow_html=True)
            with st.expander(
                f"{r['Requirement'][:100]}{'...' if len(r['Requirement'])>100 else ''}  •  Clarity {r['ClarityScore']}",
                expanded=False
            ):
                req_text = r["Requirement"]
                refs = re.findall(r"\b([A-Z]+-[A-Z]+-\d+)\b", req_text)
                for ref in refs:
                    if ref in missing_refs:
                        st.markdown(
                            f"<span style='color: #d9534f;'>🔗 Reference to missing requirement: <b>{ref}</b></span>",
                            unsafe_allow_html=True
                        )
                    for circ_a, circ_b in circular_refs:
                        if ref == circ_a or ref == circ_b:
                            st.markdown(
                                f"<span style='color: #f0ad4e;'>🔄 Circular reference detected involving <b>{ref}</b></span>",
                                unsafe_allow_html=True
                            )
                if r["ClarityScore"] == 100:
                    st.markdown(
                        '<span style="color: #28a745; font-weight: bold; font-size: 1.1em;">✅ No issues detected</span>',
                        unsafe_allow_html=True
                    )
                else:
                    if r["Issues"]:
                        for i in r["Issues"]:
                            st.markdown(
                                f"- **{i.type}** — {i.note}\n\n    ⟶ _“…{i.span}…”_"
                            )
                    if st.button("💡 Suggest rewrite", key=f"rewrite_{idx}_{hash(r['Requirement'])}"):
                        has_tbd = any(i.type == "TBD" for i in r["Issues"])
                        if has_tbd:
                            st.markdown(
                                '<span style="color: #d9534f; font-weight: bold; font-size: 1.1em;">🚩 TBD — requires clarification</span>',
                                unsafe_allow_html=True
                            )
                            st.info("This is not resolvable by AI. You must fill in the blank.")
                        with st.spinner("Proposing rewrite…"):
                            rewrite = suggest_rewrites(r["Requirement"], r["Issues"])
                        st.markdown(f"**Rewrite:** {rewrite}")

        
        st.divider()
        st.markdown("## 🔗 Dependency & Consistency Check")
        # You can still show summary here if you want
        if missing_refs:
            st.warning(f"Referenced but missing requirement IDs: {', '.join(sorted(missing_refs))}")
        else:
            st.success("No missing requirement references detected.")

        if circular_refs:
            st.error(f"Circular references detected: {', '.join([f'{a} ↔ {b}' for a, b in circular_refs])}")
        else:
            st.success("No circular references detected.")

    except Exception as e:
        st.error(f"Failed to analyze clarity: {e}")

st.divider()
with tab_tests:
    st.markdown("## 🧪 Test Ideas for Requirements")
    try:
        docs = load_documents("data")
        requirement_rows = []
        for doc in docs:
            for req in split_into_requirements(doc["text"]):
                if is_requirement(req):
                    requirement_rows.append({
                        "Source": doc.get("source") or doc.get("path", "unknown"),
                        "Requirement": req,
                    })

        if not requirement_rows:
            st.info("No requirements detected.")
            st.stop()

        st.divider()
        for r in requirement_rows:
            with st.expander(f"{r['Requirement'][:100]}{'...' if len(r['Requirement'])>100 else ''}"):
                status = "ready"
                issues = {i["type"] for i in r.get("Issues", [])}
                if "TBD" in issues:
                    status = "blocked"
                elif {"Ambiguous", "NonVerifiable", "PassiveVoice"} & issues:
                    status = "provisional"

                badge = {"ready": "✅ Ready", "provisional": "🟡 Provisional", "blocked": "🚩 Blocked"}[status]
                st.markdown(f"**Status:** {badge}")

                ideas = generate_test_ideas(r["Requirement"])
                st.caption(f"Requirement type: **{ideas['type']}**")

                if status == "blocked":
                    st.warning("REQUIRES SPECIFICATION — replace TBD/XXX before finalizing.")
                elif status == "provisional":
                    st.info("Provisional — refine vague terms or add measurable thresholds.")

                for idea in ideas["ideas"]:
                    st.markdown(f"**{idea['title']}**")
                    st.markdown("- **Steps:**")
                    for step in idea["steps"]:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;- {step}", unsafe_allow_html=True)
                    st.markdown("- **Acceptance:**")
                    for acc in idea["acceptance"]:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;- {acc}", unsafe_allow_html=True)
                    st.markdown("---")
    except Exception as e:
        st.error(f"Failed to generate test ideas: {e}")



