import os
import io
import json
import importlib
import re

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# --- Project-specific imports ---
from ingestion.loader import load_documents
from analysis.index import build_index
from analysis.qa import make_qa
from analysis.normalize_requirements import normalize_requirements
from analysis.utils import (
    split_into_requirements,
    is_requirement,
    parse_llm_content,
    analyze_dependencies,
)
from analysis.heuristics import analyze_clarity
from analysis.rewrites import suggest_rewrites
from analysis.testgen import generate_test_ideas
from analysis.traceability import build_trace_matrix, export_trace_matrix_csv


DEBUG = True  # Set to False to disable debug prints

def load_css(file_path: str) -> str:
    """Load CSS content from external file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"CSS file not found: {file_path}")
        return ""
    except Exception as e:
        st.error(f"Error loading CSS: {e}")
        return ""

def log(msg: str):
    """Simple logging helper used across the UI.

    Prints to stdout when DEBUG is True. We avoid using Streamlit writers here
    to keep logging side-effects minimal and safe during import-time execution.
    """
    if not DEBUG:
        return
    try:
        # Prefer stdout so logs appear in the terminal running Streamlit
        print(f"[LOG] {msg}")
    except Exception:
        # Last-resort fallback: ignore logging errors
        pass

st.set_page_config(page_title="Requirements Analyst Assistant", page_icon=None, layout="wide")

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
        if DEBUG:
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

# --- Utility: Get requirements from docs ---
def get_requirement_rows():
    docs = load_documents("data")
    requirement_rows = []
    for doc in docs:
        for req in split_into_requirements(doc["text"]):
            if is_requirement(req):
                requirement_rows.append({
                    "Source": doc.get("source") or doc.get("path", "unknown"),
                    "Requirement": req,
                })
    return requirement_rows

# --- Cache normalized requirements for reuse across tabs ---
@st.cache_data(show_spinner=True)
def get_normalized_requirements():
    requirement_rows = get_requirement_rows()
    requirement_chunks = [
        {"text": r["Requirement"], "source": r["Source"]}
        for r in requirement_rows
    ]
    return normalize_requirements(requirement_chunks)

@st.cache_data(show_spinner=True)
def get_clarity_results():
    results = get_normalized_requirements()
    clarity_rows = []
    for r in results:
        clarity = analyze_clarity(r["text"])
        clarity_rows.append({
            "Source": r["source"],
            "Requirement": r["text"],
            "ClarityScore": clarity["clarity_score"],
            "Issues": clarity["issues"]
        })
    return clarity_rows

# --- UI code starts here ---
# Enhanced header section
st.markdown("""
    <div class="content-card" style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none;">
        <h1 style="color: white; margin-bottom: 0.5rem; font-size: 3rem;">Requirements Analyst Assistant</h1>
        <p style="font-size: 1.4rem; opacity: 0.9; margin-bottom: 0;">
            AI-powered analysis for software requirements - clarity, quality, and compliance at scale
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div id="top"></div>', unsafe_allow_html=True)

# Load external CSS
css_content = load_css('ui/styles.css')
if css_content:
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

with st.sidebar:
    # Sidebar Header - simplified
    st.markdown("""
        <div class="sidebar-header">
            <h2 style="margin: 0; font-size: 1.6rem;">Filter Requirements</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">
                Filter and search requirements across all tabs
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Search input section
    st.markdown("**Enter keywords to filter requirements:**")
    
    # Search input with enhanced styling
    col1, col2 = st.columns([3.5, 1])
    
    # Get or initialize search instance counter
    search_instance = st.session_state.get('search_instance', 0)
    
    with col1:
        search_query = st.text_input(
            "Filter all requirements:",
            placeholder="e.g., performance, security, user...",
            key=f"unified_search_{search_instance}",
            label_visibility="collapsed",
            help="Enter keywords to filter requirements across all tabs"
        )
    with col2:
        if st.button("🗑️", key="clear_unified_search", help="Clear filter"):
            # Increment search instance to create a new text input widget
            st.session_state['search_instance'] = st.session_state.get('search_instance', 0) + 1
            # Clear any old search states
            keys_to_clear = [k for k in st.session_state.keys() if k.startswith('unified_search_')]
            for key in keys_to_clear:
                del st.session_state[key]
            st.rerun()
    
    # Search status with enhanced styling
    if search_query:
        st.markdown(f"""
            <div class="search-stats">
                <strong>&#128269; Active Search:</strong><br>
                <em>"{search_query}"</em>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background: #dbeafe; border: 1px solid #3b82f6; border-radius: 6px; padding: 0.75rem; margin-top: 0.5rem; font-size: 0.9rem;">
                <strong>🔵 Showing all requirements</strong>
            </div>
        """, unsafe_allow_html=True)
    
    # Document information - showing only file names
    st.markdown("---")
    st.markdown("**📄 Documents Loaded:**")
    
    # Show only document names
    try:
        docs = load_documents("data")
        if docs:
            doc_names = []
            for doc in docs:
                source = doc.get("source", "") or doc.get("path", "")
                if source:
                    # Remove "data/" prefix if present
                    doc_name = source.replace("data/", "") if source.startswith("data/") else source
                    doc_names.append(doc_name)
            
            if doc_names:
                for name in sorted(doc_names):
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {name}", unsafe_allow_html=True)
            else:
                st.markdown("*No document names found*")
        else:
            st.markdown("*No documents loaded*")
    except Exception as e:
        st.markdown(f"*Error loading documents: {str(e)}*")

# Define the unified search query for all tabs to use
search_instance = st.session_state.get('search_instance', 0)
unified_search_query = st.session_state.get(f'unified_search_{search_instance}', "")
search_trigger = st.session_state.get('search_trigger', 0)  # Used to force re-execution
search_summaries = unified_search_query
search_quality = unified_search_query
search_tests = unified_search_query
search_traceability = unified_search_query

tab_search, tab_summaries_traceability, tab_quality, tab_tests, tab_dashboard = st.tabs(
    ["Search", "Summaries & Traceability", "Quality", "Test Scenarios", "Dashboard"]
)

def show_sources(sources):
    unique_sources = []
    seen = set()
    for src in sources:
        if src not in seen:
            unique_sources.append(src)
            seen.add(src)
    if len(unique_sources) > 3:
        with st.expander("View Sources"):
            for i, src in enumerate(unique_sources):
                if (src.startswith("data/") and os.path.exists(src)):
                    with open(src, "rb") as f:
                        st.download_button(
                            label=f"Download {os.path.basename(src)}",
                            data=f,
                            file_name=os.path.basename(src),
                            mime="application/octet-stream",
                            key=f"download_{i}_{os.path.basename(src)}"
                        )
                else:
                    st.write("•", src)
    else:
        for i, src in enumerate(unique_sources):
            if (src.startswith("data/") and os.path.exists(src)):
                with open(src, "rb") as f:
                    st.download_button(
                        label=f"Download {os.path.basename(src)}",
                        data=f,
                        file_name=os.path.basename(src),
                        mime="application/octet-stream",
                        key=f"download_{i}_{os.path.basename(src)}"
                    )
            else:
                st.write("•", src)

with tab_search:
    llm_options = available_llm_providers()
    if not llm_options:
        st.warning("No LLM providers available. Please check your environment variables and dependencies.")

    index = get_index()
    if index is None:
        st.error("Failed to build the document index. Please check your embedding model, input data, and logs for errors.")
        if DEBUG:
            print("Failed to build the document index. Please check your embedding model, input data, and logs for errors.")
        st.stop()

    try:
        retriever = index.as_retriever(search_kwargs={"k": 4})
        if DEBUG:
            print(f"Retriever created: {retriever}")
    except Exception as e:
        st.error(f"Failed to create retriever: {e}")
        if DEBUG:
            print(f"Failed to create retriever: {e}")
        st.stop()

    qa = make_qa(retriever)
    if DEBUG:
        print(f"QA chain created: {qa}")
    if qa is None:
        st.error("QA chain was not created. Please check your retriever and LLM setup.")
        print("QA chain was not created. Please check your retriever and LLM setup.")
        st.stop()

    # Search interface anchor
    st.markdown('<div id="search-input-section"></div>', unsafe_allow_html=True)
    
    # Enhanced search interface with consolidated question section
    st.markdown("""
        <div class="content-card" style="position: relative; z-index: 10;">
            <h3 style="margin-top: 0;">&#128172; Ask a Question</h3>
            <p style="color: #64748b; margin-bottom: 1rem; font-size: 0.775rem;">Ask anything about your requirements documents. I can help with analysis, dependencies, quality checks, and more.</p>
        </div>
        
        <script>
            // Scroll to search input when Search tab is clicked
            setTimeout(function() {
                const searchSection = document.getElementById('search-input-section');
                if (searchSection) {
                    searchSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 100);
        </script>
    """, unsafe_allow_html=True)

    # Column layout for text input and clear button
    st.markdown("""
        <style>
        /* Make Ask a Question header and description slightly smaller for better balance */
        .content-card h3 {
            font-size: 1.5rem !important;
            margin-top: 0;
            margin-bottom: 0.2rem;
            font-weight: 600;
        }
        .content-card p {
            font-size: 1.1rem !important;
            color: #64748b;
            margin-bottom: 0.75rem;
        }

        /* Reduce input/textarea font size in the search area */
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea,
        input[type="text"],
        textarea {
            font-size: 1.25rem !important;
            padding: 12px 16px !important;
        }

        /* Align button text with text input text */
        button[kind="secondary"] {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            height: 38px !important;
            padding: 0 12px !important;
            line-height: 1 !important;
        }
        
        /* Align primary button with text input */
        button[kind="primary"] {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            height: 38px !important;
            padding: 0 12px !important;
            line-height: 1 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Your question:", 
            placeholder="e.g., What are the performance requirements? Which requirements have dependencies?",
            label_visibility="collapsed",
            key="search_tab_query"
        )
    
    with col2:
        if st.button("🗑️ Clear", key="clear_search", help="Clear the search input", type="primary"):
            # Increment counter to create new text input with empty value
            st.session_state.clear_counter += 1
            # Clear any stored answers
            if "answer" in st.session_state:
                del st.session_state["answer"]
            if "sources" in st.session_state:
                del st.session_state["sources"]
            if "last_query" in st.session_state:
                del st.session_state["last_query"]
            st.rerun()
    
    # Get the current query value
    query = st.session_state.get("search_tab_query", "")

    if query and st.session_state.get("last_query") != query:
        st.session_state["should_update_answer"] = True
        st.session_state["last_query"] = query

    if query:
        try:
            with st.spinner("Thinking…"):
                print(f"[LOG] Calling QA chain with: {query}")
                if st.session_state.get("should_update_answer", True):
                    out = qa({"query": query})
                    st.session_state["answer"] = out.get("result", "No answer returned.")
                    st.session_state["sources"] = [d.metadata.get("source", "unknown") for d in out.get("source_documents", [])]
                    st.session_state["should_update_answer"] = False
                    print(f"[LOG] QA chain output: {out}")

            if "last_query" not in st.session_state or st.session_state["last_query"] != query:
                st.session_state["last_query"] = query

            with st.container():
                answer = st.session_state.get("answer", "No answer returned.")
                if isinstance(answer, list):
                    # Handle list answers
                    answer_list = ""
                    for req in answer:
                        desc = req.get("description", "")
                        answer_list += f"- {desc}<br>"
                    
                    st.markdown(f"""
                        <div class="content-card" style="background-color: #dbeafe; border-left: 4px solid #3b82f6;">
                            <h3 style="margin-top: 0; margin-bottom: 1rem;">&#129001; Answer</h3>
                            <div style="font-size: 1rem; line-height: 1.5;">
                                {answer_list}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Add CSS for consistent text sizing
                    st.markdown("""
                        <style>
                        .content-card p, .content-card div, .content-card span, .content-card li, .content-card ul, .content-card ol {
                            font-size: 1rem !important;
                            line-height: 1.5 !important;
                        }
                        .content-card h1, .content-card h2, .content-card h4, .content-card h5, .content-card h6 {
                            font-size: 1rem !important;
                            font-weight: normal !important;
                            margin: 0.5rem 0 !important;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="content-card" style="background-color: #dbeafe; border-left: 4px solid #3b82f6;">
                            <h3 style="margin-top: 0; margin-bottom: 1rem;">&#129001; Answer</h3>
                            <div style="font-size: 1rem; line-height: 1.5;">
                                {answer}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Add CSS for consistent text sizing
                    st.markdown("""
                        <style>
                        .content-card p, .content-card div, .content-card span, .content-card li, .content-card ul, .content-card ol {
                            font-size: 1rem !important;
                            line-height: 1.5 !important;
                        }
                        .content-card h1, .content-card h2, .content-card h4, .content-card h5, .content-card h6 {
                            font-size: 1rem !important;
                            font-weight: normal !important;
                            margin: 0.5rem 0 !important;
                        }
                        </style>
                    """, unsafe_allow_html=True)

            with st.container():
                sources = st.session_state.get("sources", [])
                unique_sources = []
                seen = set()
                for src in sources:
                    if src not in seen:
                        unique_sources.append(src)
                        seen.add(src)
                
                # Create sources content
                sources_content = ""
                if len(unique_sources) > 3:
                    sources_content = f"<p><strong>Sources ({len(unique_sources)} documents):</strong></p>"
                    for src in unique_sources:
                        if not src.startswith("data/"):
                            sources_content += f"<p>• {src}</p>"
                else:
                    for src in unique_sources:
                        if not src.startswith("data/"):
                            sources_content += f"<p>• {src}</p>"
                
                st.markdown(f"""
                    <div class="content-card" style="background-color: #f8fafc; border-left: 4px solid #64748b;">
                        <h3 style="margin-top: 0; margin-bottom: 1rem;">&#128196; Sources</h3>
                        {sources_content}
                    </div>
                """, unsafe_allow_html=True)
                
                # Handle downloadable files separately if any exist
                downloadable_sources = [src for src in unique_sources if src.startswith("data/") and os.path.exists(src)]
                if downloadable_sources:
                    for i, src in enumerate(downloadable_sources):
                        with open(src, "rb") as f:
                            st.download_button(
                                label=f"Download {os.path.basename(src)}",
                                data=f,
                                file_name=os.path.basename(src),
                                mime="application/octet-stream",
                                key=f"download_{i}_{os.path.basename(src)}"
                            )
        except Exception as e:
            st.error(f"An error occurred while processing your question: {e}")
            import traceback
            tb = traceback.format_exc()
            print(f"[ERROR] Exception in QA call: {e}\n{tb}")
            st.text(tb)

with tab_summaries_traceability:
    # Create sub-tabs for Summaries and Traceability
    subtab_summaries, subtab_traceability = st.tabs(["Summaries", "Traceability"])
    
    with subtab_summaries:
        try:
            results = get_normalized_requirements()
            if results:
                # Filter results based on search query
                filtered_results = results
                if search_summaries:
                    search_terms = search_summaries.lower().split()
                    filtered_results = []
                    for r in results:
                        text_to_search = f"{r['text']} {r['normalized']} {' '.join(r['categories'])}".lower()
                        if all(term in text_to_search for term in search_terms):
                            filtered_results.append(r)
                
                if filtered_results:
                    df = pd.DataFrame([
                        {
                            "Requirement": r["text"],
                            "Summary": r["normalized"],
                            "Type": ", ".join(r["categories"]),
                            "Source document": r["source"].replace("data/", "") if r["source"].startswith("data/") else r["source"],
                        }
                        for r in filtered_results
                    ])
                    
                    # Show search results count
                    if search_summaries:
                        st.info(f"Found {len(filtered_results)} requirement(s) matching '{search_summaries}'")
                    
                    # Custom HTML table so long requirements wrap instead of being truncated
                    st.markdown("""
                        <style>
                        .summaries-table-container {
                            width: 100%;
                            max-height: 600px;
                            overflow-y: auto;
                            border: 1px solid #e2e8f0;
                            border-radius: 8px;
                            background: #ffffff;
                        }
                        .summaries-table {
                            width: 100%;
                            border-collapse: collapse;
                            font-size: 0.95rem;
                        }
                        .summaries-table th {
                            position: sticky;
                            top: 0;
                            background: #f8fafc;
                            border-bottom: 1px solid #e2e8f0;
                            padding: 8px 10px;
                            text-align: left;
                            font-weight: 600;
                        }
                        .summaries-table td {
                            border-top: 1px solid #e5e7eb;
                            padding: 6px 10px;
                            vertical-align: top;
                            /* ✅ allow long requirement text to wrap */
                            white-space: normal !important;
                            word-wrap: break-word;
                        }
                        .summaries-table tr:nth-child(even) {
                            background: #f9fafb;
                        }
                        </style>
                    """, unsafe_allow_html=True)

                    html_table = df.to_html(
                        classes="summaries-table",
                        index=False,
                        escape=False,
                    )

                    st.markdown(
                        f'<div class="summaries-table-container">{html_table}</div>',
                        unsafe_allow_html=True,
                    )
                    
                    for r in filtered_results:
                        if DEBUG:
                            print("NORMALIZED FIELD:", repr(r["normalized"]))
                        break
                else:
                    if search_summaries:
                        st.warning(f"No requirements found matching '{search_summaries}'")
                    else:
                        st.info("No requirements processed.")
            else:
                st.info("No requirements processed.")
        except Exception as e:
            st.error(f"Failed to process requirements: {e}")

with tab_quality:
    # Add anchor for back to top functionality
    st.markdown('<div id="quality-top"></div>', unsafe_allow_html=True)
    
    try:
        requirement_rows = get_clarity_results()
        if not requirement_rows:
            st.info("No requirements detected.")
            st.stop()

        # Filter requirements based on search query
        if search_quality:
            search_terms = search_quality.lower().split()
            filtered_requirement_rows = []
            
            # Get the original normalized data for more comprehensive search
            normalized_results = get_normalized_requirements()
            
            for r in requirement_rows:
                # Find the corresponding normalized data for this requirement
                normalized_data = None
                for norm_r in normalized_results:
                    if norm_r["text"] == r["Requirement"] and norm_r["source"] == r["Source"]:
                        normalized_data = norm_r
                        break
                
                # Search in requirement text, source, normalized text, and categories
                text_to_search = f"{r['Requirement']} {r['Source']}".lower()
                if normalized_data:
                    text_to_search += f" {normalized_data['normalized']} {' '.join(normalized_data['categories'])}".lower()
                
                if all(term in text_to_search for term in search_terms):
                    filtered_requirement_rows.append(r)
            
            requirement_rows = filtered_requirement_rows
            
            if not requirement_rows:
                st.warning(f"No requirements found matching '{search_quality}'")
                st.stop()
            else:
                st.info(f"Found {len(requirement_rows)} requirement(s) matching '{search_quality}'")

        # Calculate dependencies once for all subtabs
        missing_refs, circular_refs = analyze_dependencies(requirement_rows)

        # Add custom styling for subtabs
        st.markdown("""
            <style>
            /* Quality tab subtabs styling - more specific targeting */
            div[data-testid="stTabs"] div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                gap: 8px;
                background-color: #f1f5f9;
                border-radius: 6px;
                padding: 4px;
            }
            
            div[data-testid="stTabs"] div[data-testid="stTabs"] [data-baseweb="tab"] {
                height: 32px !important;
                padding: 0px 12px !important;
                background-color: #cbd5e1 !important;
                border-radius: 4px;
                border: none;
                font-size: 0.85rem !important;
                font-weight: 500;
                color: #475569 !important;
                min-width: auto !important;
            }
            
            div[data-testid="stTabs"] div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
                background-color: #1e40af !important;
                color: white !important;
                box-shadow: 0 1px 3px rgba(30, 64, 175, 0.3);
            }
            
            div[data-testid="stTabs"] div[data-testid="stTabs"] [data-baseweb="tab"]:not([aria-selected="true"]):hover {
                background-color: #3b82f6 !important;
                color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Create sub-tabs for the Quality tab
        subtab_analysis, subtab_dependency = st.tabs(
            ["Analysis", "Dependencies"]
        )

        with subtab_analysis:
            st.caption("Filters")
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

            # Sort filtered requirements by clarity score (lowest first) then alphabetically
            filtered_sorted = sorted(filtered, key=lambda x: (x["ClarityScore"], x["Requirement"]))
            
            df = pd.DataFrame([
                {
                    "Clarity": (
                        f'<a href="#req-{abs(hash(r["Requirement"]))}" style="text-decoration: none; color: #dc2626; font-weight: bold;">{r["ClarityScore"]}</a>'
                        if r["ClarityScore"] < 100 
                        else f'<span style="color: #059669; font-weight: bold;">{r["ClarityScore"]}</span>'
                    ),
                    "Requirement": r["Requirement"],
                    "Issues": ", ".join(sorted({i.type for i in r["Issues"]})) or "",
                    "Source": r["Source"].replace("data/", "") if r["Source"].startswith("data/") else r["Source"],
                }
                for r in filtered_sorted
            ])

            # Reduced heading size and consistent capitalization
            st.markdown(
                f"""
                <div style="background: #f7f7f9; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); padding: 1.5em 1em 1em 1em; margin-bottom: 1em;">
                    {df.to_html(escape=False, index=False)}
                </div>
                <style>
                /* Center table headers */
                table.dataframe thead th {{
                    text-align: center !important;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
            
            st.divider()

            # Details & Suggested Rewrites Section (now in same tab)
            st.markdown("### Details & Suggested Rewrites")

            # Ensure Show details button text is visible (white) across themes
            st.markdown('''
                <style>
                /* Target the Show details button more specifically */
                button:has-text("🔍 Show details"), 
                button[kind="secondary"]:contains("Show details"),
                .stButton > button:contains("🔍 Show details"),
                div[data-testid="stButton"] button:contains("Show details") {
                    color: #ffffff !important;
                }
                /* More aggressive targeting for all secondary buttons with magnifying glass */
                button[kind="secondary"] {
                    color: #ffffff !important;
                }
                </style>
            ''', unsafe_allow_html=True)

            # Show all filtered requirements by default
            for idx, r in enumerate(filtered_sorted):
                # Create anchor target for the table links
                anchor_id = f"req-{abs(hash(r['Requirement']))}"
                st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)
                
                show_details_key = f"show_details_{idx}_{hash(r['Requirement'])}"
                details_state_key = f"{show_details_key}_state"
                rewrite_btn_key = f"rewrite_btn_{idx}_{hash(r['Requirement'])}"
                rewrite_state_key = f"rewrite_state_{idx}_{hash(r['Requirement'])}"

                expander_label = f"📝 {r['Requirement'][:100]}{'...' if len(r['Requirement'])>100 else ''} • Clarity {r['ClarityScore']}"
                with st.expander(expander_label, expanded=False):
                    # Show status badge and main issues only
                    if r["ClarityScore"] == 100:
                        st.markdown(
                            '<span style="color: #28a745; font-weight: bold; font-size: 1.1em;">✅ No issues detected</span>',
                            unsafe_allow_html=True
                        )
                    else:
                        main_issues = [i.type for i in r["Issues"]]
                        st.markdown(f"**🔎 Main issues:** <span style='color:#d9534f'>{', '.join(main_issues)}</span>", unsafe_allow_html=True)

                        # Show details and rewrite only when button is pressed
                        if st.button("🔍 Show details", key=show_details_key):
                            st.session_state[details_state_key] = True

                        if st.session_state.get(details_state_key, False):
                            st.markdown("---")
                            st.markdown("#### 🗂️ Issue Details")
                            for i in r["Issues"]:
                                st.markdown(
                                    f"- <span style='color:#f0ad4e'><b>{i.type}</b></span> — {i.note}<br>&nbsp;&nbsp;&nbsp;&nbsp;_“…{i.span}…”_", unsafe_allow_html=True
                                )
                            st.markdown("---")
                            if st.button("✨ Suggest rewrite", key=rewrite_btn_key):
                                has_tbd = any(i.type == "TBD" for i in r["Issues"])
                                if has_tbd:
                                    st.markdown(
                                        '<span style="color: #d9534f; font-weight: bold; font-size: 1.1em;">🚩 TBD — requires clarification</span>',
                                        unsafe_allow_html=True
                                    )
                                    st.info("This is not resolvable by AI. You must fill in the blank.")
                                with st.spinner("Proposing rewrite…"):
                                    rewrite = suggest_rewrites(r["Requirement"], r["Issues"])
                                st.session_state[rewrite_state_key] = rewrite

                            if rewrite_state_key in st.session_state:
                                st.markdown("#### ✏️ <span style='color:#0072B2'>Rewrite</span>", unsafe_allow_html=True)
                                st.info(st.session_state[rewrite_state_key])

            # Back to top button
            st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
            st.markdown("""
                <a href="#quality-top" style="
                    display: inline-block;
                    background-color: #ffffff;
                    border: 1px solid #cccccc;
                    color: #262730;
                    padding: 0.5rem 1rem;
                    text-decoration: none;
                    border-radius: 0.5rem;
                    font-size: 1rem;
                    font-weight: 400;
                    line-height: 1.6;
                    cursor: pointer;
                    transition: all 0.15s ease-in-out;
                    min-height: 2.5rem;
                    min-width: 180px;
                    box-sizing: border-box;
                    text-align: center;
                    vertical-align: middle;
                    white-space: nowrap;
                " onmouseover="this.style.borderColor='#ff4b4b'; this.style.color='#ff4b4b'" onmouseout="this.style.borderColor='#cccccc'; this.style.color='#262730'">
                    ⬆️ Back to top
                </a>
            """, unsafe_allow_html=True)

        with subtab_dependency:
            # Add professional styling for Dependencies tab
            st.markdown("""
                <style>
                .dependency-section {
                    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin: 1rem 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                    border: 1px solid #e2e8f0;
                }
                
                .status-card {
                    background: white;
                    border-radius: 8px;
                    padding: 1.25rem;
                    margin: 0.75rem 0;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
                    border: 1px solid #e5e7eb;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                }
                
                .status-card:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
                }
                
                .status-icon {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    width: 2.5rem;
                    height: 2.5rem;
                    border-radius: 50%;
                    font-size: 1.25rem;
                    margin-right: 1rem;
                }
                
                .status-success {
                    background-color: #dcfce7;
                    color: #16a34a;
                    border-left: 4px solid #22c55e;
                }
                
                .status-warning {
                    background-color: #fef3c7;
                    color: #d97706;
                    border-left: 4px solid #f59e0b;
                }
                
                .status-error {
                    background-color: #fee2e2;
                    color: #dc2626;
                    border-left: 4px solid #ef4444;
                }
                
                .circular-refs {
                    background: linear-gradient(45deg, #fef2f2 0%, #fecaca 100%);
                    border: 1px solid #f87171;
                    border-radius: 6px;
                    padding: 1rem;
                    margin: 0.75rem 0;
                    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                }
                
                .ref-item {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    margin: 0.5rem 0;
                    padding: 0.5rem;
                    background: rgba(255, 255, 255, 0.7);
                    border-radius: 4px;
                }
                
                .section-title {
                    color: #1e293b;
                    font-weight: 600;
                    font-size: 1.1rem;
                    margin: 0 0 0.5rem 0;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                </style>
            """, unsafe_allow_html=True)

            if missing_refs:
                st.markdown(f"""
                    <div class="dependency-section">
                        <div class="status-card status-warning">
                            <div style="display: flex; align-items: flex-start;">
                                <div class="status-icon" style="background-color: #fef3c7; color: #d97706;">
                                    ⚠️
                                </div>
                                <div style="flex: 1;">
                                    <h3 class="section-title" style="color: #d97706;">
                                        Missing References Detected
                                    </h3>
                                    <p style="margin: 0 0 1rem 0; color: #92400e; line-height: 1.5;">
                                        The following requirement IDs are referenced but do not exist in the current dataset:
                                    </p>
                                    <div class="circular-refs" style="background: linear-gradient(45deg, #fef3c7 0%, #fed7aa 100%); border-color: #f59e0b;">
                                        {''.join([f'<div class="ref-item"><span style="background: #f59e0b; color: white; padding: 0.25rem 0.5rem; border-radius: 3px; font-size: 0.8rem; font-weight: 600;">REF</span><code style="color: #92400e; font-weight: 600;">{ref}</code></div>' for ref in sorted(missing_refs)])}
                                    </div>
                                    <p style="margin: 0.75rem 0 0 0; color: #92400e; font-size: 0.9rem; font-style: italic;">
                                        💡 <strong>Recommendation:</strong> Verify these requirement IDs exist or update the references.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="dependency-section">
                        <div class="status-card status-success">
                            <div style="display: flex; align-items: center;">
                                <div class="status-icon" style="background-color: #dcfce7; color: #16a34a;">
                                    ✅
                                </div>
                                <div>
                                    <h3 class="section-title" style="color: #16a34a;">
                                        All References Valid
                                    </h3>
                                    <p style="margin: 0; color: #065f46; line-height: 1.5;">
                                        No missing requirement references detected. All dependency links are valid.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            if circular_refs:
                st.markdown(f"""
                    <div class="dependency-section">
                        <div class="status-card status-error">
                            <div style="display: flex; align-items: flex-start;">
                                <div class="status-icon" style="background-color: #fee2e2; color: #dc2626;">
                                    🚨
                                </div>
                                <div style="flex: 1;">
                                    <h3 class="section-title" style="color: #dc2626;">
                                        Circular Dependencies Detected
                                    </h3>
                                    <p style="margin: 0 0 1rem 0; color: #dc2626; line-height: 1.5;">
                                        The following circular reference patterns were found:
                                    </p>
                                    <div class="circular-refs">
                                        {''.join([f'<div class="ref-item"><span style="background: #dc2626; color: white; padding: 0.25rem 0.5rem; border-radius: 3px; font-size: 0.8rem; font-weight: 600;">CIRCULAR</span><strong style="color: #dc2626;">{a}</strong> <span style="color: #6b7280;">↔</span> <strong style="color: #dc2626;">{b}</strong></div>' for a, b in circular_refs])}
                                    </div>
                                    <p style="margin: 0.75rem 0 0 0; color: #dc2626; font-size: 0.9rem; font-style: italic;">
                                        💡 <strong>Critical:</strong> Review these requirements immediately to break the circular dependency chain.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="dependency-section">
                        <div class="status-card status-success">
                            <div style="display: flex; align-items: center;">
                                <div class="status-icon" style="background-color: #dcfce7; color: #16a34a;">
                                    ✅
                                </div>
                                <div>
                                    <h3 class="section-title" style="color: #16a34a;">
                                        No Circular Dependencies
                                    </h3>
                                    <p style="margin: 0; color: #065f46; line-height: 1.5;">
                                        No circular reference patterns detected. Dependency structure is clean and well-organized.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to analyze clarity: {e}")

def render_compact_test_card(requirement, clarity_score, status):
    """Render a very compact test card with minimal vertical space"""
    with st.container():
        # Mini header
        col1, col2 = st.columns([4,1])
        with col1:
            st.markdown(f"**{requirement['Requirement']}**")
        with col2:
            score_text = str(clarity_score) if clarity_score is not None else 'n/a'
            st.caption(f"Score: {score_text}")
        
        # Generate and show test scenarios in compact format
        ideas = generate_test_ideas(requirement["Requirement"])
        
        with st.expander(f"{len(ideas['ideas'])} test scenario(s)", expanded=False):
            for i, idea in enumerate(ideas['ideas'][:3]):  # Limit to 3 for space
                st.markdown(f"**{idea.get('title')}**")
                
                # Show all steps/acceptance (no truncation)
                steps = list(idea.get('steps', []))
                acceptance = list(idea.get('acceptance', []))
                
                # Mask for non-ready status
                if status in ('provisional','blocked'):
                    def _mask(s):
                        return re.sub(r"\b\d+(?:\.\d+)?\b", "<target>", s)
                    steps = [_mask(s) for s in steps]
                    acceptance = [_mask(a) for a in acceptance]
                
                # Format acceptance criteria with proper capitalization
                def format_acceptance(text):
                    # Split by common sentence starters and rejoin with proper capitalization
                    parts = re.split(r'\b(Given|When|Then|And|But)\b', text)
                    formatted_parts = []
                    for i, part in enumerate(parts):
                        if part in ['Given', 'When', 'Then', 'And', 'But']:
                            if i == 0:  # First word stays capitalized
                                formatted_parts.append(part)
                            else:  # Subsequent instances become lowercase
                                formatted_parts.append(part.lower())
                        else:
                            formatted_parts.append(part)
                    result = ''.join(formatted_parts)
                    # Ensure the first character is capitalized
                    if result and result[0].islower():
                        result = result[0].upper() + result[1:]
                    return result
                
                formatted_acceptance = [format_acceptance(a) for a in acceptance]
                
                if steps:
                    steps_html = "<br>".join([f"{j}. {step}" for j, step in enumerate(steps, 1)])
                    st.markdown(f'<div style="font-size:0.9rem; line-height:1.1;"><strong>Steps:</strong><br>{steps_html}</div>', unsafe_allow_html=True)
                if formatted_acceptance:
                    accept_html = "<br>".join([f"{accept}" for accept in formatted_acceptance])
                    st.markdown(f'<div style="font-size:0.9rem; line-height:1.1;"><br><strong>Accept:</strong><br>{accept_html}</div>', unsafe_allow_html=True)
                
                if i < len(ideas['ideas']) - 1:
                    st.divider()
        
        # Compact export status
        if status == 'blocked':
            st.caption("🚫 Export disabled")
        elif status == 'provisional':
            confirm_key = f"confirm_prov_{abs(hash(requirement['Requirement']))}"
            st.checkbox('✓ Confirm for export', key=confirm_key, help="Confirm export for provisional item")
        else:
            st.caption("✅ Ready for export")
        
        st.divider()

st.divider()
with tab_tests:
    try:
        requirement_rows = get_requirement_rows()
        if not requirement_rows:
            st.info("No requirements detected.")
            st.stop()

        # Filter requirements based on search query
        if search_tests:
            search_terms = search_tests.lower().split()
            filtered_requirement_rows = []
            
            # Get the original normalized data for more comprehensive search
            normalized_results = get_normalized_requirements()
            
            for r in requirement_rows:
                # Find the corresponding normalized data for this requirement
                normalized_data = None
                for norm_r in normalized_results:
                    if norm_r["text"] == r["Requirement"] and norm_r["source"] == r["Source"]:
                        normalized_data = norm_r
                        break
                
                # Search in requirement text, source, normalized text, and categories
                text_to_search = f"{r['Requirement']} {r['Source']}".lower()
                if normalized_data:
                    text_to_search += f" {normalized_data['normalized']} {' '.join(normalized_data['categories'])}".lower()
                
                if all(term in text_to_search for term in search_terms):
                    filtered_requirement_rows.append(r)
            
            requirement_rows = filtered_requirement_rows
            
            if not requirement_rows:
                st.warning(f"No requirements found matching '{search_tests}'")
                st.stop()
            else:
                st.info(f"Found {len(requirement_rows)} requirement(s) matching '{search_tests}'")

        # Precompute clarity lookup to gate test exports and UI
        clarity_rows = get_clarity_results()
        clarity_map = { (c["Requirement"], c["Source"]): c for c in clarity_rows }

        # Add CSS for better download button visibility
        st.markdown("""
        <style>
        div[data-testid="stDownloadButton"] > button {
            background-color: #0066cc !important;
            color: white !important;
            border: 1px solid #0066cc !important;
            padding: 0.25rem 0.75rem !important;
            border-radius: 6px !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            opacity: 1 !important;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            background-color: #0052a3 !important;
            border-color: #0052a3 !important;
            color: white !important;
        }
        div[data-testid="stDownloadButton"] > button > div {
            color: white !important;
        }
        div[data-testid="stDownloadButton"] > button > div > span {
            color: white !important;
        }
        div[data-testid="stDownloadButton"] > button p {
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Status-based organization for space efficiency
        ready_reqs = []
        provisional_reqs = []
        blocked_reqs = []
        
        for r in requirement_rows:
            req_key = (r["Requirement"], r["Source"])
            clarity = clarity_map.get(req_key, None)
            clarity_score = None
            issues_set = set()
            if clarity:
                clarity_score = clarity.get("ClarityScore") if isinstance(clarity, dict) else getattr(clarity, "ClarityScore", None)
                raw_issues = clarity.get("Issues", []) if isinstance(clarity, dict) else getattr(clarity, "Issues", [])
                for i in raw_issues:
                    if isinstance(i, dict):
                        issues_set.add(i.get("type"))
                    else:
                        issues_set.add(getattr(i, "type", None))

            # Determine status
            if "TBD" in issues_set or (clarity_score is not None and clarity_score < 80):
                blocked_reqs.append((r, clarity_score))
            elif clarity_score is not None and 80 <= clarity_score < 100:
                provisional_reqs.append((r, clarity_score))
            else:
                ready_reqs.append((r, clarity_score))

        # Visibility controls with counts
        st.markdown("**Show sections:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            show_ready = st.checkbox(f"Ready ({len(ready_reqs)})", value=True, key="show_ready_tests")
        with col2:
            show_provisional = st.checkbox(f"Provisional ({len(provisional_reqs)})", value=True, key="show_provisional_tests")
        with col3:
            show_blocked = st.checkbox(f"Blocked ({len(blocked_reqs)})", value=False, key="show_blocked_tests")
        
        st.divider()
        
        # Single download button for all test scenarios
        if st.button("Download Test Scenarios", type="primary", help="Download test scenarios for all visible requirements"):
            with st.spinner("Generating test scenarios..."):
                all_export_rows = []
                
                # Process all requirements based on current visibility settings
                for req_data in ready_reqs:
                    requirement, clarity_score = req_data
                    if show_ready:
                        ideas = generate_test_ideas(requirement["Requirement"])
                        for idea in ideas['ideas']:
                            steps = list(idea.get('steps', []))
                            acceptance = list(idea.get('acceptance', []))
                            
                            # Format acceptance criteria
                            def format_acceptance(text):
                                parts = re.split(r'\b(Given|When|Then|And|But)\b', text)
                                formatted_parts = []
                                for j, part in enumerate(parts):
                                    if part in ['Given', 'When', 'Then', 'And', 'But']:
                                        if j == 0:
                                            formatted_parts.append(part)
                                        else:
                                            formatted_parts.append(part.lower())
                                    else:
                                        formatted_parts.append(part)
                                result = ''.join(formatted_parts)
                                if result and result[0].islower():
                                    result = result[0].upper() + result[1:]
                                return result
                            
                            formatted_acceptance = [format_acceptance(a) for a in acceptance]
                            
                            all_export_rows.append({
                                'requirement': requirement['Requirement'],
                                'status': 'ready',
                                'test_scenario': idea.get('title'),
                                'test_steps': ' | '.join(steps),
                                'acceptance_criteria': ' | '.join(formatted_acceptance)
                            })
                
                # Process provisional requirements
                for req_data in provisional_reqs:
                    requirement, clarity_score = req_data
                    if show_provisional:
                        confirm_key = f"confirm_prov_{abs(hash(requirement['Requirement']))}"
                        if st.session_state.get(confirm_key, False):
                            ideas = generate_test_ideas(requirement["Requirement"])
                            for idea in ideas['ideas']:
                                steps = list(idea.get('steps', []))
                                acceptance = list(idea.get('acceptance', []))
                                
                                # Mask for provisional
                                def _mask(s):
                                    return re.sub(r"\b\d+(?:\.\d+)?\b", "<target>", s)
                                steps = [_mask(s) for s in steps]
                                acceptance = [_mask(a) for a in acceptance]
                                
                                # Format acceptance criteria
                                def format_acceptance(text):
                                    parts = re.split(r'\b(Given|When|Then|And|But)\b', text)
                                    formatted_parts = []
                                    for j, part in enumerate(parts):
                                        if part in ['Given', 'When', 'Then', 'And', 'But']:
                                            if j == 0:
                                                formatted_parts.append(part)
                                            else:
                                                formatted_parts.append(part.lower())
                                        else:
                                            formatted_parts.append(part)
                                    result = ''.join(formatted_parts)
                                    if result and result[0].islower():
                                        result = result[0].upper() + result[1:]
                                    return result
                                
                                formatted_acceptance = [format_acceptance(a) for a in acceptance]
                                
                                all_export_rows.append({
                                    'requirement': requirement['Requirement'],
                                    'status': 'provisional',
                                    'test_scenario': idea.get('title'),
                                    'test_steps': ' | '.join(steps),
                                    'acceptance_criteria': ' | '.join(formatted_acceptance)
                                })
                
                # Generate and provide download
                if all_export_rows:
                    out_df = pd.DataFrame(all_export_rows)
                    csv_bytes = out_df.to_csv(index=False).encode('utf-8')
                    st.success(f"✅ Generated {len(all_export_rows)} test scenarios!")
                    st.download_button('⬇️ Click to Download CSV', 
                                     data=csv_bytes, 
                                     file_name="all_test_scenarios.csv", 
                                     mime='text/csv',
                                     type="primary")
                else:
                    st.warning("No test scenarios available for download. Please confirm provisional exports or select different sections.")
        
        # Dynamic column layout based on visible sections
        visible_sections = []
        if show_ready and ready_reqs:
            visible_sections.append(("ready", ready_reqs, "✅ Ready"))
        if show_provisional and provisional_reqs:
            visible_sections.append(("provisional", provisional_reqs, "🟡 Provisional"))
        if show_blocked and blocked_reqs:
            visible_sections.append(("blocked", blocked_reqs, "🚩 Blocked"))
        
        if not visible_sections:
            st.info("No sections selected or no requirements match the current criteria.")
        else:
            # Create columns based on number of visible sections
            if len(visible_sections) == 1:
                cols = [st.container()]
            elif len(visible_sections) == 2:
                cols = st.columns(2)
            else:
                cols = st.columns(3)
            
            for i, (status, reqs, title) in enumerate(visible_sections):
                with cols[i]:
                    expanded = (status == "ready") or (status == "provisional" and not show_ready)
                    with st.expander(f"{title} ({len(reqs)})", expanded=expanded):
                        if reqs:
                            for r_data in reqs:
                                render_compact_test_card(r_data[0], r_data[1], status)
                        else:
                            st.info(f"No {status} requirements")

    except Exception as e:
        st.error(f"Failed to generate test scenarios: {e}")

    with subtab_traceability:
        st.markdown("## 📊 Traceability Matrix")
        
        try:
            requirement_rows = get_requirement_rows()
            if not requirement_rows:
                st.info("No requirements detected.")
                st.stop()

            # Filter requirements based on search query
            if search_traceability:
                search_terms = search_traceability.lower().split()
                filtered_requirement_rows = []
                
                # Get the original normalized data for more comprehensive search
                normalized_results = get_normalized_requirements()
                
                for r in requirement_rows:
                    # Find the corresponding normalized data for this requirement
                    normalized_data = None
                    for norm_r in normalized_results:
                        if norm_r["text"] == r["Requirement"] and norm_r["source"] == r["Source"]:
                            normalized_data = norm_r
                            break
                    
                    # Search in requirement text, source, normalized text, and categories
                    text_to_search = f"{r['Requirement']} {r['Source']}".lower()
                    if normalized_data:
                        text_to_search += f" {normalized_data['normalized']} {' '.join(normalized_data['categories'])}".lower()
                    
                    if all(term in text_to_search for term in search_terms):
                        filtered_requirement_rows.append(r)
                
                requirement_rows = filtered_requirement_rows
                
                if not requirement_rows:
                    st.warning(f"No requirements found matching '{search_traceability}'")
                    st.stop()
                else:
                    st.info(f"Found {len(requirement_rows)} requirement(s) matching '{search_traceability}'")

            trace_df = build_trace_matrix(requirement_rows)
            # Reorder and rename columns for display/export per user's requested order
            display_df = trace_df.rename(columns={
                "ReqID": "Requirement ID",
                "DependsOn": "Depends On",
                "Covers": "Verifies",
                "CoveredBy": "Verified By",
                "Source": "Source document"
            })
            
            # Clean up source document names - remove "data/" and keep only filename
            if "Source document" in display_df.columns:
                display_df["Source document"] = display_df["Source document"].apply(
                    lambda x: x.replace("data/", "").split("/")[-1] if isinstance(x, str) else x
                )
            
            desired_cols = [
                "Requirement ID",
                "Type",
                "Requirement",
                "Depends On",
                "Verifies",
                "Verified By",
                "Source document",
            ]
            # Only keep columns that exist (defensive) and preserve order
            cols_present = [c for c in desired_cols if c in display_df.columns]
            display_df = display_df.reindex(columns=cols_present)

            # Custom HTML table so long requirements wrap instead of being truncated
            st.markdown("""
                <style>
                .traceability-table-container {
                    width: 100%;
                    max-height: 600px;
                    overflow-y: auto;
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                    background: #ffffff;
                }
                .traceability-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.95rem;
                }
                .traceability-table th {
                    position: sticky;
                    top: 0;
                    background: #f8fafc;
                    border-bottom: 1px solid #e2e8f0;
                    padding: 8px 10px;
                    text-align: left;
                    font-weight: 600;
                }
                .traceability-table td {
                    border-top: 1px solid #e5e7eb;
                    padding: 6px 10px;
                    vertical-align: top;
                    /* ✅ allow long requirement text to wrap */
                    white-space: normal !important;
                    word-wrap: break-word;
                }
                .traceability-table tr:nth-child(even) {
                    background: #f9fafb;
                }
                </style>
            """, unsafe_allow_html=True)

            html_table = display_df.to_html(
                classes="traceability-table",
                index=False,
                escape=False,
            )

            st.markdown(
                f'<div class="traceability-table-container">{html_table}</div>',
                unsafe_allow_html=True,
            )
            csv_bytes = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Traceability Matrix CSV",
                data=csv_bytes,
                file_name="traceability_matrix.csv",
                mime="text/csv",
                key="trace_matrix_download"
            )
        except Exception as e:
            st.error(f"Failed to generate traceability matrix: {e}")

with tab_dashboard:
    st.markdown("## 📊 Requirements Dashboard")
    
    try:
        results = get_normalized_requirements()
        total_reqs = len(results) if results else 0
        
        if total_reqs == 0:
            st.warning("⚠️ No requirements data available. Please load requirements in the Search tab first.")
        else:
            # === KPI CARDS ROW ===
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    label="📋 Total Requirements", 
                    value=total_reqs,
                    help="Total number of requirements processed"
                )
            
            # Calculate clarity metrics
            clarity_rows = get_clarity_results()
            issue_counts = {"TBD": 0, "Ambiguous": 0, "NonVerifiable": 0, "PassiveVoice": 0}
            clarity_scores = []
            
            for r in clarity_rows:
                if r.get("ClarityScore") is not None:
                    clarity_scores.append(r["ClarityScore"])
                types = {i.type for i in r.get("Issues", [])}
                if "TBD" in types: issue_counts["TBD"] += 1
                if "Ambiguous" in types: issue_counts["Ambiguous"] += 1
                if "NonVerifiable" in types: issue_counts["NonVerifiable"] += 1
                if "PassiveVoice" in types: issue_counts["PassiveVoice"] += 1
            
            avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0
            
            with col2:
                st.metric(
                    label="🎯 Avg Clarity Score", 
                    value=f"{avg_clarity:.1f}",
                    delta=None,  # Remove confusing delta calculation
                    help="Average clarity score (1-10 scale, higher is better)"
                )
            
            with col3:
                tbd_pct = 100 * issue_counts['TBD'] / total_reqs if total_reqs else 0
                st.metric(
                    label="🚨 TBD Issues", 
                    value=f"{tbd_pct:.1f}%",
                    help="Percentage of requirements with TBD (To Be Determined) content"
                )
            
            with col4:
                ambiguous_pct = 100 * issue_counts['Ambiguous'] / total_reqs if total_reqs else 0
                st.metric(
                    label="⚠️ Ambiguous", 
                    value=f"{ambiguous_pct:.1f}%",
                    help="Percentage of requirements with ambiguous language"
                )
            
            # Calculate test coverage
            requirement_rows = [
                {"Source": r["source"], "Requirement": r["text"]}
                for r in results
            ]
            trace_df = build_trace_matrix(requirement_rows)
            sys_rows = trace_df[trace_df["Type"] == "System"]
            sys_covered = sys_rows[sys_rows["CoveredBy"].apply(lambda x: any(tid.startswith("TST-") for tid in x.split(",") if tid.strip()))] if len(sys_rows) > 0 else pd.DataFrame()
            coverage_pct = 100 * len(sys_covered) / len(sys_rows) if len(sys_rows) else 0
            
            with col5:
                st.metric(
                    label="✅ Test Coverage", 
                    value=f"{coverage_pct:.1f}%",
                    help="Percentage of system requirements covered by tests"
                )
            
            st.markdown("---")
            
            # === COVERAGE VISUALIZATION BAR ===
            st.markdown("### 📊 Test Coverage Overview")
            
            if len(sys_rows) > 0:
                covered_count = len(sys_covered)
                uncovered_count = len(sys_rows) - covered_count
                
                # Create coverage data
                coverage_data = pd.DataFrame({
                    'Status': ['Covered by Tests', 'Not Covered'],
                    'Count': [covered_count, uncovered_count],
                    'Percentage': [coverage_pct, 100 - coverage_pct]
                })
                
                # Visual coverage bar using progress bar
                col_bar1, col_bar2 = st.columns([3, 1])
                with col_bar1:
                    st.progress(coverage_pct / 100, text=f"Test Coverage: {coverage_pct:.1f}% ({covered_count}/{len(sys_rows)} system requirements)")
                with col_bar2:
                    if coverage_pct >= 80:
                        st.success("Excellent!")
                    elif coverage_pct >= 60:
                        st.warning("Good")
                    else:
                        st.error("Needs Improvement")
                        
                # Coverage breakdown chart
                st.bar_chart(coverage_data.set_index('Status')['Count'])
            else:
                st.info("💡 No system requirements found for coverage analysis")
            
            st.markdown("---")
            
            # === SIDE-BY-SIDE CHARTS ===
            chart_col1, chart_col2 = st.columns(2)
            
            # Left chart: Quality Issues Distribution
            with chart_col1:
                st.markdown("### 🔍 Quality Issues Distribution")
                
                if any(count > 0 for count in issue_counts.values()):
                    issue_data = pd.DataFrame({
                        'Issue Type': list(issue_counts.keys()),
                        'Count': list(issue_counts.values()),
                        'Percentage': [100 * count / total_reqs for count in issue_counts.values()]
                    })
                    issue_data = issue_data[issue_data['Count'] > 0]  # Only show non-zero counts
                    
                    # Rename for better display
                    issue_data['Issue Type'] = issue_data['Issue Type'].replace({
                        'TBD': 'TBD Content',
                        'Ambiguous': 'Ambiguous Language', 
                        'NonVerifiable': 'Non-Verifiable',
                        'PassiveVoice': 'Passive Voice'
                    })
                    
                    st.bar_chart(issue_data.set_index('Issue Type')['Count'])
                    
                    # Show percentages in a small table
                    st.caption("Issue Breakdown:")
                    for _, row in issue_data.iterrows():
                        st.caption(f"• {row['Issue Type']}: {row['Count']} ({row['Percentage']:.1f}%)")
                else:
                    st.success("🎉 No quality issues detected!")
            
            # Right chart: Requirement Types Distribution  
            with chart_col2:
                st.markdown("### 📋 Requirement Types Distribution")
                
                all_categories = [cat for r in results for cat in r.get("categories", [])]
                if all_categories:
                    cat_series = pd.Series(all_categories)
                    cat_counts = cat_series.value_counts()
                    
                    st.bar_chart(cat_counts)
                    
                    # Show percentages
                    st.caption("Type Breakdown:")
                    for cat_type, count in cat_counts.head(5).items():  # Top 5
                        pct = 100 * count / total_reqs
                        st.caption(f"• {cat_type}: {count} ({pct:.1f}%)")
                    
                    if len(cat_counts) > 5:
                        with st.expander(f"... and {len(cat_counts) - 5} more types"):
                            for cat_type, count in cat_counts.iloc[5:].items():  # Remaining types
                                pct = 100 * count / total_reqs
                                st.caption(f"• {cat_type}: {count} ({pct:.1f}%)")
                else:
                    st.info("💡 No requirement type categories available")
            
            # === SUMMARY INSIGHTS ===
            st.markdown("---")
            st.markdown("### 💡 Key Insights")
            
            insights = []
            
            if avg_clarity < 5:
                insights.append("🚨 **Low clarity scores** - Consider reviewing requirement definitions")
            elif avg_clarity > 7:
                insights.append("**Good clarity scores** - Requirements are well-defined")
                
            if tbd_pct > 20:
                insights.append("**High TBD content** - Many requirements need further definition")
            elif tbd_pct == 0:
                insights.append("✅ **No TBD content** - All requirements are fully defined")
                
            if coverage_pct < 50:
                insights.append("**Low test coverage** - Consider adding more test scenarios")
            elif coverage_pct > 80:
                insights.append("🎯 **Excellent test coverage** - Most requirements are tested")
                
            if len(insights) == 0:
                insights.append("📊 **Good overall status** - Requirements are in decent shape")
            
            for insight in insights:
                st.markdown(insight)
                
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        if DEBUG:
            st.exception(e)



