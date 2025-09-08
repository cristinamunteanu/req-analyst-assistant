import os
import streamlit as st
from dotenv import load_dotenv
from ingestion.loader import load_documents
from analysis.index import build_index
from analysis.qa import make_qa
from analysis.normalize_requirements import normalize_requirements
from analysis.utils import split_into_requirements, is_requirement, parse_llm_content
import importlib
import io
import json

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

# --- Logs expander in sidebar ---
with st.sidebar.expander("Logs", expanded=False):
    st.code(log_buffer.getvalue() or "No logs yet.", language="text")

# --- Tabbed layout ---
tab_search, tab_summaries, tab_quality = st.tabs(["Search", "Summaries", "Quality"])

with tab_search:
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
    if query:
        try:
            with st.spinner("Thinking…"):
                log(f"Calling QA chain with: {query}")
                out = qa({"query": query})
                log(f"QA chain output: {out}")
            st.subheader("Answer")
            st.write(out.get("result", "No answer returned."))
            st.subheader("Sources")
            sources = {d.metadata.get("source", "unknown") for d in out.get("source_documents", [])}
            for src in sources:
                st.write("•", src)
        except Exception as e:
            st.error(f"An error occurred while processing your question: {e}")
            import traceback
            tb = traceback.format_exc()
            log(f"Exception in QA call: {e}\n{tb}")
            st.text(tb)

with tab_summaries:
    st.subheader("Requirement Normalization & Categorization")
    try:
        docs = load_documents("data")
        requirement_chunks = []
        for doc in docs:
            requirements = split_into_requirements(doc["text"])
            for req in requirements[1:]:  # skip preamble
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
    st.info("Quality metrics and analysis coming soon.")


