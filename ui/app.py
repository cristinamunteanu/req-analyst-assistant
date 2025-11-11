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

st.set_page_config(page_title="Requirements Analyst Assistant", page_icon="🔎", layout="wide")

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
        <h1 style="color: white; margin-bottom: 0.5rem; font-size: 3rem;">🔎 Requirements Analyst Assistant</h1>
        <p style="font-size: 1.4rem; opacity: 0.9; margin-bottom: 0;">
            AI-powered analysis for software requirements - clarity, quality, and compliance at scale
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div id="top"></div>', unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Import Google Fonts for better typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling improvements */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        background-color: #fafbfc;
    }
    
    /* Smooth scrolling for anchor links */
    html {
        scroll-behavior: smooth;
    }
    
    /* Anchor target styling for better visibility */
    div[id^="req-"] {
        scroll-margin-top: 100px;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    h1 {
        color: #1e293b;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #334155;
        font-weight: 600;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }
    
    h3 {
        color: #475569;
        font-weight: 500;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Enhanced tab styling */
    [data-testid="stTabs"] {
        background-color: white;
        padding: 0.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    [data-testid="stTabs"] button {
        font-size: 6rem !important;
        font-weight: 600 !important;
        padding: 2rem 3.5rem !important;
        border-radius: 8px !important;
        margin: 0 0.25rem !important;
        border: none !important;
        background-color: transparent !important;
        color: #64748b !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stTabs"] button:hover {
        background-color: #f1f5f9 !important;
        color: #475569 !important;
    }
    
    [data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Sidebar improvements */
    .stSidebar {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    .stSidebar .stSelectbox, .stSidebar .stSlider {
        background-color: white;
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    }
    
    /* Enhanced input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: border-color 0.2s ease;
        background-color: white;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Enhanced dataframe styling */
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Info/warning/error message improvements */
    .stAlert {
        border-radius: 8px;
        border: none;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .stAlert[data-baseweb="notification"] {
        padding: 1rem 1.5rem;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        color: #166534;
    }
    
    /* Warning messages */
    .stWarning {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        color: #92400e;
    }
    
    /* Error messages */
    .stError {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        color: #dc2626;
    }
    
    /* Info messages */
    .stInfo {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        color: #1e40af;
    }
    
    /* Enhanced metric styling */
    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Enhanced expander styling */
    .streamlit-expanderHeader {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        font-weight: 500;
        color: #374151;
    }
    
    .streamlit-expanderContent {
        background-color: #f9fafb;
        border: 1px solid #e2e8f0;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1.5rem;
    }
    
    /* Enhanced code block styling */
    .stCodeBlock {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        background-color: #1e293b;
    }
    
    pre {
        white-space: pre-wrap !important;
        word-break: break-word !important;
        background-color: #1e293b;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
    }
    
    /* Loading spinner improvements */
    .stSpinner {
        color: #3b82f6;
    }
    
    /* Column spacing improvements */
    .row-widget {
        margin-bottom: 1rem;
    }
    
    /* Custom card styling for content sections */
    .content-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Improved spacing for sections */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0f172a;
        }
        
        .content-card {
            background-color: #1e293b;
            border-color: #334155;
        }
        
        h1, h2, h3 {
            color: #f1f5f9;
        }
        
        [data-testid="stTabs"] {
            background-color: #1e293b;
            border-color: #334155;
        }
    }
    
    /* Make content text bigger for better readability */
    .stApp p, .stApp li, .stApp span, .stApp div:not([data-testid="stTabs"]) {
        font-size: 1.1rem !important;
    }
    
    /* Keep specific elements at normal or larger size */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        font-size: inherit !important;
    }
    
    /* Sidebar text should be readable */
    .stSidebar p, .stSidebar span, .stSidebar div {
        font-size: 1rem !important;
    }
    
    /* DataFrames should be readable */
    .stDataFrame, .stDataFrame * {
        font-size: 1rem !important;
    }
    
    /* Override small text for important headers */
    .content-card h1 {
        font-size: 2.6rem !important;
    }
    
    .content-card p {
        font-size: 1.4rem !important;
    }
    
    .sidebar-header h2 {
        font-size: 1.6rem !important;
    }
    
    .sidebar-header p {
        font-size: 1.1rem !important;
    }
    
    /* Make text inputs taller but keep Enter key behavior */
    .stTextInput > div > div > input {
        height: 2.8rem !important;
        min-height: 2.8rem !important;
        padding: 0.8rem 0.75rem !important;
        font-size: 1.1rem !important;
        line-height: 1.2 !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    # Enhanced sidebar styling
    st.markdown("""
        <style>
        /* Sidebar-specific enhancements */
        .stSidebar {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        }
        
        .sidebar-header {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.1);
        }
        
        .sidebar-section {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .sidebar-section h3 {
            color: #1e293b;
            margin-top: 0;
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .search-stats {
            background: #eff6ff;
            border: 1px solid #3b82f6;
            border-radius: 6px;
            padding: 0.75rem;
            margin-top: 0.5rem;
            font-size: 0.9rem;
        }
        
        .sidebar-button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            font-size: 0.9rem;
            cursor: pointer;
            width: 100%;
            transition: all 0.2s ease;
            text-decoration: none;
            display: inline-block;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .sidebar-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(16, 185, 129, 0.3);
        }
        
        .clear-btn {
            background: #ef4444;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem;
            font-size: 0.8rem;
            cursor: pointer;
            width: 100%;
            height: 100%;
        }
        
        .clear-btn:hover {
            background: #dc2626;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar Header - simplified
    st.markdown("""
        <div class="sidebar-header">
            <h2 style="margin: 0; font-size: 1.6rem;">&#128269; Filter Requirements</h2>
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
            <div style="background: #f0fdf4; border: 1px solid #22c55e; border-radius: 6px; padding: 0.75rem; margin-top: 0.5rem; font-size: 0.9rem;">
                <strong>✅ Showing all requirements</strong>
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
    
    # Help Section
    st.markdown("""
        <div class="sidebar-section">
            <h3>❓ How to Use</h3>
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.4;">
                <p><strong>�🔍 Search:</strong> Enter keywords to filter requirements</p>
                <p><strong>📋 Tabs:</strong> Switch between different analysis views</p>
                <p><strong>🗑️ Clear:</strong> Reset search to see all requirements</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Define the unified search query for all tabs to use
search_instance = st.session_state.get('search_instance', 0)
unified_search_query = st.session_state.get(f'unified_search_{search_instance}', "")
search_trigger = st.session_state.get('search_trigger', 0)  # Used to force re-execution
search_summaries = unified_search_query
search_quality = unified_search_query
search_tests = unified_search_query
search_traceability = unified_search_query

tab_search, tab_summaries, tab_quality, tab_tests, tab_traceability, tab_dashboard = st.tabs(
    ["Search", "Summaries", "Quality", "Test ideas", "Traceability", "Dashboard"]
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
                        <div class="content-card" style="background-color: #f0fdf4; border-left: 4px solid #22c55e;">
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
                        <div class="content-card" style="background-color: #f0fdf4; border-left: 4px solid #22c55e;">
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

with tab_summaries:
    
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
                        "Source document": r["source"].replace("data/", "") if r["source"].startswith("data/") else r["source"],
                        "Requirement": r["text"],
                        "Summary": r["normalized"],
                        "Type": ", ".join(r["categories"]),
                    }
                    for r in filtered_results
                ])
                
                # Show search results count
                if search_summaries:
                    st.info(f"Found {len(filtered_results)} requirement(s) matching '{search_summaries}'")
                
                # Display table with centered headers using HTML
                st.markdown("""
                    <style>
                    .summaries-table-container {
                        width: 100%;
                        overflow-x: auto;
                        overflow-y: auto;
                        max-height: 600px;
                        border: 1px solid #dee2e6;
                        border-radius: 8px;
                        margin: 1rem 0;
                    }
                    .summaries-table {
                        width: 100%;
                        border-collapse: collapse;
                        margin: 0;
                    }
                    .summaries-table th {
                        background-color: #f8f9fa;
                        border: 1px solid #dee2e6;
                        padding: 12px;
                        text-align: center !important;
                        font-weight: 600;
                        color: #495057;
                        position: sticky;
                        top: 0;
                        z-index: 10;
                    }
                    .summaries-table td {
                        border: 1px solid #dee2e6;
                        padding: 12px;
                        vertical-align: top;
                    }
                    .summaries-table tr:nth-child(even) {
                        background-color: #f8f9fa;
                    }
                    .summaries-table tr:hover {
                        background-color: #e9ecef;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                # Convert to HTML table with scrollable container
                html_table = df.to_html(classes='summaries-table', escape=False, index=False)
                st.markdown(f'<div class="summaries-table-container">{html_table}</div>', unsafe_allow_html=True)
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
    
    st.markdown("""
        <div class="section-header">
            <h2 style="margin: 0;">&#129529; Clarity & Ambiguity Checks</h2>
        </div>
    """, unsafe_allow_html=True)
    
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

        # Create sub-tabs for the Quality tab
        subtab_analysis, subtab_dependency = st.tabs(
            ["Table View & Details", "Dependency & Consistency Check"]
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

            if missing_refs:
                st.markdown(f"""
                    <div class="content-card" style="background-color: #fef3cd; border-left: 4px solid #f59e0b; margin-bottom: 1rem;">
                        <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                            <span style="font-size: 1.5rem; color: #f59e0b;">⚠️</span>
                            <div>
                                <h5 style="margin: 0 0 0.5rem 0; color: #92400e;">Missing References Found</h5>
                                <p style="margin: 0; color: #92400e; font-size: 0.9rem;">
                                    The following requirement IDs are referenced but do not exist in the current dataset:
                                </p>
                                <ul style="margin: 0.5rem 0 0 0; color: #92400e;">
                                    {''.join([f'<li><code>{ref}</code></li>' for ref in sorted(missing_refs)])}
                                </ul>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="content-card" style="background-color: #d1fae5; border-left: 4px solid #10b981;">
                        <div style="display: flex; align-items: center; gap: 0.75rem;">
                            <span style="font-size: 1.5rem; color: #10b981;">✅</span>
                            <div>
                                <h5 style="margin: 0; color: #065f46;">All References Valid</h5>
                                <p style="margin: 0; color: #065f46; font-size: 0.9rem;">
                                    No missing requirement references detected in the current dataset.
                                </p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            if circular_refs:
                st.markdown(f"""
                    <div class="content-card" style="background-color: #fee2e2; border-left: 4px solid #ef4444; margin-bottom: 1rem;">
                        <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                            <span style="font-size: 1.5rem; color: #ef4444;">🚨</span>
                            <div>
                                <h5 style="margin: 0 0 0.5rem 0; color: #dc2626;">Circular Dependencies Detected</h5>
                                <p style="margin: 0 0 0.75rem 0; color: #dc2626; font-size: 0.9rem;">
                                    The following circular reference patterns were found:
                                </p>
                                <div style="background-color: #fef2f2; padding: 0.75rem; border-radius: 0.375rem; border: 1px solid #fecaca;">
                                    {''.join([f'<div style="color: #dc2626; font-family: monospace; margin: 0.25rem 0;"><strong>{a}</strong> ↔ <strong>{b}</strong></div>' for a, b in circular_refs])}
                                </div>
                                <p style="margin: 0.75rem 0 0 0; color: #dc2626; font-size: 0.85rem; font-style: italic;">
                                    💡 Tip: Review these requirements to break the circular dependency chain.
                                </p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="content-card" style="background-color: #d1fae5; border-left: 4px solid #10b981;">
                        <div style="display: flex; align-items: center; gap: 0.75rem;">
                            <span style="font-size: 1.5rem; color: #10b981;">✅</span>
                            <div>
                                <h5 style="margin: 0; color: #065f46;">No Circular Dependencies</h5>
                                <p style="margin: 0; color: #065f46; font-size: 0.9rem;">
                                    No circular reference patterns detected. Dependency structure is clean.
                                </p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to analyze clarity: {e}")

st.divider()
with tab_tests:
    st.markdown("## 🧪 Test Ideas for Requirements")
    
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

with tab_traceability:
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
        st.dataframe(trace_df, use_container_width=True)
        csv_bytes = trace_df.to_csv(index=False).encode("utf-8")
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

    results = get_normalized_requirements()
    total_reqs = len(results)
    st.metric("Total Requirements", total_reqs)

    clarity_rows = get_clarity_results()
    issue_counts = {"TBD": 0, "Ambiguous": 0, "NonVerifiable": 0, "PassiveVoice": 0}
    for r in clarity_rows:
        types = {i.type for i in r["Issues"]}
        if "TBD" in types: issue_counts["TBD"] += 1
        if "Ambiguous" in types: issue_counts["Ambiguous"] += 1
        if "NonVerifiable" in types: issue_counts["NonVerifiable"] += 1
        if "PassiveVoice" in types: issue_counts["PassiveVoice"] += 1

    st.metric("% with TBD", f"{100 * issue_counts['TBD'] / total_reqs:.1f}%" if total_reqs else "0%")
    st.metric("% Ambiguous", f"{100 * issue_counts['Ambiguous'] / total_reqs:.1f}%" if total_reqs else "0%")
    st.metric("% Non-Verifiable", f"{100 * issue_counts['NonVerifiable'] / total_reqs:.1f}%" if total_reqs else "0%")
    st.metric("% Passive Voice", f"{100 * issue_counts['PassiveVoice'] / total_reqs:.1f}%" if total_reqs else "0%")

    requirement_rows = [
        {"Source": r["source"], "Requirement": r["text"]}
        for r in results
    ]
    trace_df = build_trace_matrix(requirement_rows)
    sys_rows = trace_df[trace_df["Type"] == "System"]
    sys_covered = sys_rows[sys_rows["CoveredBy"].apply(lambda x: any(tid.startswith("TST-") for tid in x.split(",") if tid.strip()))]
    coverage_pct = 100 * len(sys_covered) / len(sys_rows) if len(sys_rows) else 0
    st.metric("SYS Coverage by TST", f"{coverage_pct:.1f}%")

    all_categories = [cat for r in results for cat in r.get("categories", [])]
    if all_categories:
        cat_series = pd.Series(all_categories)
        st.bar_chart(cat_series.value_counts())
    else:
        st.info("Category distribution not available (no categories found).")



