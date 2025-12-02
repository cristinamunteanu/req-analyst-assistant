import os
import io
import json
import importlib
import re

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# --- Lazy imports - only import when needed ---
# Heavy ML imports moved to functions to avoid issues during Streamlit reruns
import tempfile
import shutil
from pathlib import Path


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

# Initialize session state for uploaded documents
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "use_uploaded" not in st.session_state:
    st.session_state.use_uploaded = False

def process_uploaded_files(uploaded_files):
    """Process uploaded files and return document list."""
    try:
        from unstructured.partition.auto import partition
    except ImportError as e:
        log(f"Failed to import unstructured: {e}")
        st.error("Required library 'unstructured' not available. Please check your installation.")
        return []

    if not uploaded_files:
        log("No uploaded files provided to process_uploaded_files")
        return []

    docs = []
    temp_dir = None

    try:
        temp_dir = tempfile.mkdtemp()
        log(f"Created temporary directory: {temp_dir}")

        for i, uploaded_file in enumerate(uploaded_files):
            try:
                log(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")

                # Validate file object
                if not hasattr(uploaded_file, 'name') or not hasattr(uploaded_file, 'getvalue'):
                    log(f"Invalid file object: {uploaded_file}")
                    continue

                # Save uploaded file to temporary location
                temp_file_path = Path(temp_dir) / uploaded_file.name
                file_content = uploaded_file.getvalue()

                if len(file_content) == 0:
                    log(f"File {uploaded_file.name} is empty, skipping")
                    continue

                with open(temp_file_path, "wb") as f:
                    f.write(file_content)
                log(f"Saved uploaded file to: {temp_file_path} (size: {len(file_content)} bytes)")

                # Process the file
                try:
                    elements = partition(filename=str(temp_file_path))
                    log(f"Partitioned file {uploaded_file.name}, got {len(elements)} elements")

                    text = "\n".join([getattr(el, "text", "") for el in elements if getattr(el, "text", "")])
                    if text.strip():
                        docs.append({
                            "name": uploaded_file.name,
                            "text": text,
                            "path": uploaded_file.name,
                            "source": uploaded_file.name,
                            "size": len(file_content)
                        })
                        log(f"Successfully processed uploaded file: {uploaded_file.name} (text length: {len(text)})")
                    else:
                        log(f"No text extracted from file: {uploaded_file.name}")
                        st.warning(f"No text could be extracted from {uploaded_file.name}")
                except Exception as partition_error:
                    error_msg = f"Error partitioning {uploaded_file.name}: {partition_error}"
                    log(error_msg)
                    st.warning(error_msg)
                    continue

            except Exception as file_error:
                error_msg = f"Error processing file {uploaded_file.name}: {file_error}"
                log(error_msg)
                st.warning(error_msg)
                continue

    except Exception as e:
        error_msg = f"Error setting up temporary directory: {e}"
        log(error_msg)
        st.error(error_msg)
        return []
    finally:
        # Clean up temporary directory
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
                log(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                log(f"Failed to clean up temporary directory: {cleanup_error}")

    log(f"Processed {len(docs)} documents from {len(uploaded_files)} uploaded files")
    if len(docs) == 0:
        st.warning("No documents could be processed from the uploaded files. Please check the file formats and content.")
    return docs

def load_documents_from_session():
    """Load documents from session state (uploaded files)."""
    # Debug session state
    if DEBUG:
        print(f"[DEBUG] Session state check:")
        print(f"  use_uploaded: {getattr(st.session_state, 'use_uploaded', 'NOT_SET')}")
        print(f"  uploaded_docs exists: {hasattr(st.session_state, 'uploaded_docs')}")
        print(f"  uploaded_docs count: {len(getattr(st.session_state, 'uploaded_docs', []))}")
    
    if st.session_state.use_uploaded and st.session_state.uploaded_docs:
        return st.session_state.uploaded_docs
    else:
        return []

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

# --- Utility: Get requirements from uploaded docs ---
def get_requirement_rows():
    docs = load_documents_from_session()
    requirement_rows = []

    if DEBUG:
        log(f"get_requirement_rows: Processing {len(docs)} documents")
        for i, doc in enumerate(docs):
            log(f"  Document {i+1}: {doc.get('name', 'unknown')} ({len(doc.get('text', ''))} chars)")

    if not docs:
        return []

    # Try model-based extraction first (LLM with regex fallback)
    try:
        from analysis.index import extract_requirements_with_model, convert_extracted_to_dict
        from analysis.llm_providers import get_default_provider
        import tempfile
        import os
        
        # Check if LLM is available
        llm_provider = get_default_provider()
        use_llm = llm_provider is not None
        
        if DEBUG:
            method = "LLM with regex fallback" if use_llm else "regex/heuristics only"
            log(f"Using extraction method: {method}")
        
        # Create temporary files for each document since the extractor expects file paths
        temp_files = []
        try:
            for doc in docs:
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
                temp_file.write(doc.get('text', ''))
                temp_file.close()
                temp_files.append((temp_file.name, doc.get('name', 'unknown')))
            
            # Extract requirements using model-based approach
            file_paths = [temp_path for temp_path, _ in temp_files]
            extracted_requirements = extract_requirements_with_model(file_paths, use_llm=use_llm)
            
            if DEBUG:
                log(f"Model-based extraction found {len(extracted_requirements)} requirements")
            
            # Convert to the expected format
            for req in extracted_requirements:
                # The source_hint from the extractor should contain the file path
                # Map it back to the original document name
                source_name = req.source_hint
                
                # Try to find the matching original document name
                for temp_path, orig_name in temp_files:
                    if temp_path in req.source_hint:
                        source_name = orig_name
                        break
                
                requirement_rows.append({
                    "Source": source_name,
                    "Requirement": req.text,
                })
                
        finally:
            # Clean up temporary files
            for temp_path, _ in temp_files:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                    
    except Exception as e:
        # Fallback to original regex-only method
        log(f"Model-based extraction failed: {e}, falling back to regex-only")
        
        try:
            from analysis.utils import split_into_requirements, is_requirement
        except ImportError as e:
            st.error(f"Failed to import analysis utilities: {e}")
            return []

        for doc in docs:
            doc_requirements = []
            for req in split_into_requirements(doc["text"]):
                if is_requirement(req):
                    requirement_rows.append({
                        "Source": doc.get("source") or doc.get("name", "unknown"),
                        "Requirement": req,
                    })
                    doc_requirements.append(req)

            if DEBUG:
                log(f"  Found {len(doc_requirements)} requirements in {doc.get('name', 'unknown')}")

    if DEBUG:
        log(f"Total requirement_rows: {len(requirement_rows)}")

    return requirement_rows

# --- Cache normalized requirements for reuse across tabs ---
@st.cache_data(show_spinner=True)
def get_normalized_requirements(_uploaded_docs_hash):
    requirement_rows = get_requirement_rows()
    requirement_chunks = [
        {"text": r["Requirement"], "source": r["Source"]}
        for r in requirement_rows
    ]

    if DEBUG:
        log(f"get_normalized_requirements: Processing {len(requirement_chunks)} requirement chunks")
        source_counts = {}
        for chunk in requirement_chunks:
            source = chunk["source"]
            source_counts[source] = source_counts.get(source, 0) + 1
        for source, count in source_counts.items():
            log(f"  {source}: {count} requirements")

    # Lazy import - only when needed
    try:
        from analysis.normalize_requirements import normalize_requirements
        result = normalize_requirements(requirement_chunks)

        if DEBUG:
            log(f"get_normalized_requirements: Returned {len(result)} normalized requirements")
            source_counts_result = {}
            for r in result:
                source = r["source"]
                source_counts_result[source] = source_counts_result.get(source, 0) + 1
            for source, count in source_counts_result.items():
                log(f"  Normalized {source}: {count} requirements")

        return result
    except ImportError as e:
        st.error(f"Failed to import normalize_requirements: {e}")
        return []

@st.cache_data(show_spinner=True)
def get_clarity_results(_uploaded_docs_hash):
    results = get_normalized_requirements(_uploaded_docs_hash)
    clarity_rows = []

    # Lazy import - only when needed
    try:
        from analysis.heuristics import analyze_clarity
        for r in results:
            clarity = analyze_clarity(r["text"])
            clarity_rows.append({
                "Source": r["source"],
                "Requirement": r["text"],
                "ClarityScore": clarity["clarity_score"],
                "Issues": clarity["issues"]
            })
        return clarity_rows
    except ImportError as e:
        st.error(f"Failed to import analyze_clarity: {e}")
        return []

def get_uploaded_docs_hash():
    """Create a hash of uploaded documents to use for cache invalidation."""
    if not st.session_state.uploaded_docs:
        return "no_docs"
    # Create a comprehensive hash based on document names, sizes, and content length
    doc_info = []
    for doc in st.session_state.uploaded_docs:
        doc_info.append((
            doc.get('name', ''),
            doc.get('size', 0),
            len(doc.get('text', '')),
            hash(doc.get('text', '')[:100])  # Hash of first 100 chars for uniqueness
        ))

    # Sort to ensure consistent hashing regardless of order
    doc_info_sorted = sorted(doc_info)
    result_hash = hash(str(doc_info_sorted))

    if DEBUG:
        log(f"get_uploaded_docs_hash: Hash for {len(st.session_state.uploaded_docs)} docs = {result_hash}")
        for i, info in enumerate(doc_info_sorted):
            log(f"  Doc {i+1}: {info[0]} (size: {info[1]}, text_len: {info[2]})")

    return result_hash


# Enhanced header section
st.markdown("""
    <div class="content-card header-card">
        <h1 class="header-title">Requirements Analyst Assistant</h1>
        <p class="header-subtitle">
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
    # Sidebar Header
    st.markdown("""
        <div class="sidebar-header">
            <h2 class="sidebar-header-title">Document Manager</h2>
            <p class="sidebar-header-subtitle">
                Upload and manage requirement documents
            </p>
        </div>
    """, unsafe_allow_html=True)

    # File upload section with professional styling
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload requirement documents",
        type=['pdf', 'docx', 'doc', 'txt', 'md'],
        accept_multiple_files=True,
        help="Supported formats: PDF, Word, text, and markdown files",
        label_visibility="collapsed",
        key="document_uploader"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files and len(uploaded_files) > 0:
        # Check for duplicate files
        existing_doc_names = {doc['name'] for doc in st.session_state.uploaded_docs}
        new_files = []
        duplicate_files = []

        for file in uploaded_files:
            if file.name in existing_doc_names:
                duplicate_files.append(file.name)
            else:
                new_files.append(file)

        # Show files to be processed
        if new_files:
            st.markdown("---")
            st.markdown('<div class="files-ready-section">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">Files Ready for Processing:</h3>', unsafe_allow_html=True)
            for file in new_files:
                file_size = len(file.getvalue())
                if file_size < 1024:
                    size_str = f"{file_size} B"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"

                st.markdown(f"""
                    <div class="file-item">
                        <div class="file-name">{file.name}</div>
                        <div class="file-size">{size_str}</div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Process Files", type="primary", use_container_width=True):
                try:
                    with st.spinner("Processing new files..."):
                        log(f"Starting to process {len(new_files)} new files")
                        processed_docs = process_uploaded_files(new_files)

                        if processed_docs:
                            # Append to existing documents instead of replacing
                            st.session_state.uploaded_docs.extend(processed_docs)
                            st.session_state.use_uploaded = True
                            
                            # Debug: Confirm session state is set
                            if DEBUG:
                                print(f"[DEBUG] After processing: use_uploaded = {st.session_state.use_uploaded}")
                                print(f"[DEBUG] After processing: uploaded_docs count = {len(st.session_state.uploaded_docs)}")

                            # Explicitly clear caches to ensure fresh data
                            get_normalized_requirements.clear()
                            get_clarity_results.clear()

                            log(f"Successfully processed {len(processed_docs)} documents, total now: {len(st.session_state.uploaded_docs)}")
                            log("Cleared caches for fresh data")
                            st.success(f"Successfully processed {len(processed_docs)} new documents!")
                            
                            # Show extraction method status
                            try:
                                from analysis.llm_providers import get_default_provider
                                llm_provider = get_default_provider()
                                if llm_provider:
                                    st.info("🤖 Using **LLM-based extraction** with automatic regex fallback")
                                else:
                                    st.info("📝 Using **regex-based extraction** (no LLM provider available)")
                            except Exception:
                                st.info("📝 Using **regex-based extraction**")
                            
                            if duplicate_files:
                                st.info(f"Skipped {len(duplicate_files)} duplicate file(s)")
                            st.rerun()
                        else:
                            log("No documents were processed successfully")
                            st.error("No new documents were processed successfully")

                except Exception as e:
                    error_msg = f"Error processing files: {e}"
                    log(error_msg)
                    st.error(error_msg)
                    import traceback
                    log(f"Full traceback: {traceback.format_exc()}")
        else:
            if duplicate_files:
                st.info("All selected files are already uploaded. No new files to process.")
            else:
                st.write("No files selected for processing.")

    # Document management section
    st.markdown('<div class="documents-section">', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<h3 class="section-header">Loaded Documents:</h3>', unsafe_allow_html=True)

    if st.session_state.uploaded_docs:
        st.markdown('<div class="document-list">', unsafe_allow_html=True)
        for doc in st.session_state.uploaded_docs:
            doc_text_length = len(doc.get('text', ''))
            if doc_text_length < 1000:
                size_class = "small"
                size_desc = "Small"
            elif doc_text_length < 5000:
                size_class = "medium"
                size_desc = "Medium"
            else:
                size_class = "large"
                size_desc = "Large"

            st.markdown(f"""
                <div class="document-item {size_class}">
                    <div class="doc-name">{doc['name']}</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Create a form-based clear button with blue styling
        with st.form(key="clear_docs_form"):
            clear_submitted = st.form_submit_button("Clear All Documents",
                                                   help="Remove all uploaded documents",
                                                   use_container_width=True)

        if clear_submitted:
            st.session_state.uploaded_docs = []
            st.session_state.use_uploaded = False
            # Clear all related caches
            get_normalized_requirements.clear()
            get_clarity_results.clear()
            log("Cleared all uploaded files and caches")
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="no-docs-message">No documents uploaded yet.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Search functionality - only show if documents are loaded
    if st.session_state.uploaded_docs:
        st.markdown("---")
        st.markdown('<h3 class="section-header">Filter Requirements:</h3>', unsafe_allow_html=True)

        # Search input section
        st.markdown("Enter keywords to filter requirements:")

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
                <div class="filter-stats">
                    <strong>🔵 Showing all requirements</strong>
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
    if not st.session_state.uploaded_docs:
        st.info("Please upload documents using the sidebar to start searching and analyzing requirements.")
        st.markdown("""
            **Getting Started:**
            1. Use the sidebar to upload requirement documents (PDF, Word, text files)
            2. Click "Process Uploaded Files" to extract text
            3. Once processed, you can search, analyze quality, and generate tests
        """)
    # Continue with search functionality even if no docs (will show instructive errors)

    llm_options = available_llm_providers()
    if not llm_options:
        st.warning("No LLM providers available. Please check your environment variables and dependencies.")
        # st.stop()  # COMMENTED OUT to allow other tabs to execute

    # Lazy imports for QA functionality
    try:
        from ingestion.loader import load_documents
        from analysis.index import build_index
        from analysis.qa import make_qa

        # Build index from uploaded documents
        @st.cache_resource(show_spinner=False)
        def get_index_from_uploaded():
            with st.spinner("Indexing uploaded documents…"):
                docs = load_documents_from_session()
                if DEBUG:
                    print(f"[DEBUG] get_index_from_uploaded: Loaded {len(docs)} documents")
                    for i, doc in enumerate(docs):
                        print(f"  Doc {i+1}: {doc.get('name', 'unknown')} (text_len: {len(doc.get('text', ''))})")
                
                if not docs:
                    if DEBUG:
                        print("[DEBUG] get_index_from_uploaded: No documents loaded from session")
                    return None
                
                try:
                    embed_model = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                    if DEBUG:
                        print(f"[DEBUG] get_index_from_uploaded: Using embedding model: {embed_model}")
                        print(f"[DEBUG] get_index_from_uploaded: About to call build_index with {len(docs)} docs")
                    
                    index = build_index(docs, embed_model=embed_model)
                    
                    if DEBUG:
                        print(f"[DEBUG] get_index_from_uploaded: build_index returned: {index is not None}")
                        print(f"[DEBUG] get_index_from_uploaded: Index type: {type(index)}")
                    
                    return index
                except Exception as e:
                    error_msg = f"Error building index: {e}"
                    print(f"[ERROR] get_index_from_uploaded: {error_msg}")  # Always log the actual error
                    if DEBUG:
                        import traceback
                        print(f"[DEBUG] get_index_from_uploaded: Full traceback: {traceback.format_exc()}")
                    return None

        index = get_index_from_uploaded()
        if index is None:
            if not st.session_state.uploaded_docs:
                st.info("ℹ️ Search functionality will be available once you upload and process documents.")
            else:
                st.error("Failed to build the document index. This could be due to:")
                st.markdown("""
                - Document processing issues
                - Empty or invalid document content
                - Missing embedding model dependencies
                - Memory or resource constraints
                """)
                st.info("💡 Try re-uploading your documents or check if they contain readable text content.")
            if DEBUG:
                print("Failed to build the document index from uploaded documents.")
            st.stop()

        try:
            retriever = index.as_retriever(search_kwargs={"k": 4})
            if DEBUG:
                print(f"Retriever created: {retriever}")
        except Exception as e:
            if not st.session_state.uploaded_docs:
                st.info("ℹ️ Document retriever will be ready once you upload and process documents.")
            else:
                st.error(f"Failed to create retriever: {e}")
            if DEBUG:
                print(f"Failed to create retriever: {e}")
            st.stop()

        qa = make_qa(retriever)
        if DEBUG:
            print(f"QA chain created: {qa}")
        if qa is None:
            if not st.session_state.uploaded_docs:
                st.info("ℹ️ Question-answering capability will be available once you upload and process documents.")
            else:
                st.error("QA chain was not created. Please check your retriever and LLM setup.")
            print("QA chain was not created. Please check your retriever and LLM setup.")
            st.stop()

    except ImportError as e:
        st.error(f"Required analysis modules not available: {e}")
        st.info("Some search functionality requires additional dependencies.")
        st.stop()

    # Search interface anchor
    st.markdown('<div id="search-input-section"></div>', unsafe_allow_html=True)

    # Enhanced search interface with consolidated question section
    st.markdown("""
        <div class="content-card question-card">
            <h3 class="question-title">&#128172; Ask a Question</h3>
            <p class="question-subtitle">Ask anything about your requirements documents. I can help with analysis, dependencies, quality checks, and more.</p>
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
                        <div class="content-card answer-card">
                            <h3 class="answer-title">&#129001; Answer</h3>
                            <div class="answer-content">
                                {answer_list}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="content-card answer-card">
                            <h3 class="answer-title">&#129001; Answer</h3>
                            <div class="answer-content">
                                {answer}
                            </div>
                        </div>
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
                    <div class="content-card sources-card">
                        <h3 class="sources-title">&#128196; Sources</h3>
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
    if not st.session_state.uploaded_docs:
        st.info("Please upload and process documents to view summaries and traceability information.")
        # st.stop()  # COMMENTED OUT to allow other tabs to execute

    # Create sub-tabs for Summaries and Traceability
    subtab_summaries, subtab_traceability = st.tabs(["Summaries", "Traceability"])

    with subtab_summaries:
        try:
            docs_hash = get_uploaded_docs_hash()
            results = get_normalized_requirements(docs_hash)

            if DEBUG:
                log(f"Summaries tab: Got {len(results)} normalized requirements")
                source_counts = {}
                for r in results:
                    source = r["source"]
                    source_counts[source] = source_counts.get(source, 0) + 1
                for source, count in source_counts.items():
                    log(f"  Summaries tab {source}: {count} requirements")

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
    if not st.session_state.uploaded_docs:
        st.info("Please upload and process documents to analyze requirements quality.")
        # st.stop()  # COMMENTED OUT to allow other tabs to execute

    # Add anchor for back to top functionality
    st.markdown('<div id="quality-top"></div>', unsafe_allow_html=True)

    try:
        docs_hash = get_uploaded_docs_hash()
        requirement_rows = get_clarity_results(docs_hash)
        if not requirement_rows:
            st.info("No requirements detected.")
            st.stop()

        # Filter requirements based on search query
        if search_quality:
            search_terms = search_quality.lower().split()
            filtered_requirement_rows = []

            # Get the original normalized data for more comprehensive search
            docs_hash = get_uploaded_docs_hash()
            normalized_results = get_normalized_requirements(docs_hash)

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
        try:
            from analysis.utils import analyze_dependencies
            missing_refs, circular_refs = analyze_dependencies(requirement_rows)
        except ImportError as e:
            st.error(f"Failed to import dependency analysis: {e}")
            missing_refs, circular_refs = [], []

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
                        f'<a href="#req-{abs(hash(r["Requirement"]))}" class="clarity-score-bad">{r["ClarityScore"]}</a>'
                        if r["ClarityScore"] < 100
                        else f'<span class="clarity-score-good">{r["ClarityScore"]}</span>'
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
                <div class="table-wrapper">
                    {df.to_html(escape=False, index=False)}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.divider()

            # Details & Suggested Rewrites Section (now in same tab)
            st.markdown("### Details & Suggested Rewrites")

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
                            '<span class="status-good">No issues detected</span>',
                            unsafe_allow_html=True
                        )
                    else:
                        main_issues = [i.type for i in r["Issues"]]
                        st.markdown(f"**🔎 Main issues:** <span class='main-issues'>{', '.join(main_issues)}</span>", unsafe_allow_html=True)

                        # Show details and rewrite only when button is pressed
                        if st.button("🔍 Show details", key=show_details_key):
                            st.session_state[details_state_key] = True

                        if st.session_state.get(details_state_key, False):
                            st.markdown("---")
                            st.markdown("#### 🗂️ Issue Details")
                            for i in r["Issues"]:
                                st.markdown(
                                    f'- <span class="issue-type">{i.type}</span> — {i.note}<br>&nbsp;&nbsp;&nbsp;&nbsp;_"…{i.span}…"_', unsafe_allow_html=True
                                )
                            st.markdown("---")
                            if st.button("✨ Suggest rewrite", key=rewrite_btn_key):
                                has_tbd = any(i.type == "TBD" for i in r["Issues"])
                                if has_tbd:
                                    st.markdown(
                                        '<span class="tbd-warning">🚩 TBD — requires clarification</span>',
                                        unsafe_allow_html=True
                                    )
                                    st.info("This is not resolvable by AI. You must fill in the blank.")
                                with st.spinner("Proposing rewrite…"):
                                    try:
                                        from analysis.rewrites import suggest_rewrites
                                        rewrite = suggest_rewrites(r["Requirement"], r["Issues"])
                                        st.session_state[rewrite_state_key] = rewrite
                                    except ImportError as e:
                                        st.error(f"Failed to import rewrite functionality: {e}")

                            if rewrite_state_key in st.session_state:
                                st.markdown("#### ✏️ <span class='rewrite-header'>Rewrite</span>", unsafe_allow_html=True)
                                st.info(st.session_state[rewrite_state_key])

            # Back to top button
            st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
            st.markdown('''
                <a href="#quality-top" class="back-to-top">
                    ⬆️ Back to top
                </a>
                <script>
                // Prevent unwanted scrolling when Streamlit reruns
                document.addEventListener('DOMContentLoaded', function() {
                    // Store current scroll position before Streamlit updates
                    if (window.location.hash && window.location.hash.startsWith('#req-')) {
                        // Only scroll to anchor if it was explicitly clicked, not on rerun
                        if (!sessionStorage.getItem('streamlit_rerun')) {
                            document.querySelector(window.location.hash)?.scrollIntoView({behavior: 'smooth', block: 'start'});
                        } else {
                            // Clear the flag and remove hash to prevent auto-scroll
                            sessionStorage.removeItem('streamlit_rerun');
                            history.replaceState(null, null, window.location.pathname);
                        }
                    }
                });

                // Set flag when any button is clicked to indicate this is a rerun
                document.addEventListener('click', function(e) {
                    if (e.target.tagName === 'BUTTON' || e.target.closest('button')) {
                        sessionStorage.setItem('streamlit_rerun', 'true');
                    }
                });
                </script>
            ''', unsafe_allow_html=True)

        with subtab_dependency:
            if missing_refs:
                # Build the missing refs HTML in one go
                missing_refs_items = ''.join([
                    f'<div class="ref-item"><span class="ref-badge warning">REF</span><code class="ref-code warning">{ref}</code></div>'
                    for ref in sorted(missing_refs)
                ])

                st.markdown(f'''
                    <div class="dependency-section">
                        <div class="status-card status-warning">
                            <div class="status-card-content">
                                <div class="status-icon warning"></div>
                                <div>
                                    <h3 class="section-title warning">Missing References Detected</h3>
                                    <p class="dependency-description warning">
                                        The following requirement IDs are referenced but do not exist in the current dataset:
                                    </p>
                                    <div class="dependency-refs warning">
                                        {missing_refs_items}
                                    </div>
                                    <p class="dependency-recommendation warning">
                                        <strong>Recommendation:</strong> Verify these requirement IDs exist or update the references.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                    <div class="dependency-section">
                        <div class="status-card status-success">
                            <div class="status-card-content">
                                <div class="status-icon success"></div>
                                <div>
                                    <h3 class="section-title success">All References Valid</h3>
                                    <p class="dependency-description success">
                                        No missing requirement references detected. All dependency links are valid.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

            if circular_refs:
                # Build the circular refs HTML in one go
                circular_refs_items = ''.join([
                    f'<div class="ref-item"><span class="ref-badge error">CIRCULAR</span><strong class="ref-code error">{a}</strong><span class="dependency-arrow">↔</span><strong class="ref-code error">{b}</strong></div>'
                    for a, b in circular_refs
                ])

                st.markdown(f'''
                    <div class="dependency-section">
                        <div class="status-card status-error">
                            <div class="status-card-content">
                                <div class="status-icon error"></div>
                                <div>
                                    <h3 class="section-title error">Circular Dependencies Detected</h3>
                                    <p class="dependency-description error">
                                        The following circular reference patterns were found:
                                    </p>
                                    <div class="dependency-refs error">
                                        {circular_refs_items}
                                    </div>
                                    <p class="dependency-recommendation error">
                                        <strong>Critical:</strong> Review these requirements immediately to break the circular dependency chain.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                    <div class="dependency-section">
                        <div class="status-card status-success">
                            <div class="status-card-content">
                                <div class="status-icon success"></div>
                                <div>
                                    <h3 class="section-title success">No Circular Dependencies</h3>
                                    <p class="dependency-description success">
                                        No circular reference patterns detected. Dependency structure is clean and well-organized.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

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
        try:
            from analysis.testgen import generate_test_ideas
            ideas = generate_test_ideas(requirement["Requirement"])
        except ImportError as e:
            st.error(f"Test generation not available: {e}")
            return

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
                    st.markdown(f'<div class="test-steps"><strong>Steps:</strong><br>{steps_html}</div>', unsafe_allow_html=True)
                if formatted_acceptance:
                    accept_html = "<br>".join([f"{accept}" for accept in formatted_acceptance])
                    st.markdown(f'<div class="test-acceptance"><br><strong>Accept:</strong><br>{accept_html}</div>', unsafe_allow_html=True)

                if i < len(ideas['ideas']) - 1:
                    st.divider()

        # Compact export status
        if status == 'blocked':
            st.caption("Export disabled")
        elif status == 'provisional':
            confirm_key = f"confirm_prov_{abs(hash(requirement['Requirement']))}"
            st.checkbox('Confirm for export', key=confirm_key, help="Confirm export for provisional item")
        else:
            st.caption("Ready for export")

        st.divider()

st.divider()
with tab_tests:
    if not st.session_state.uploaded_docs:
        st.info("Please upload and process documents to generate test scenarios.")
        # st.stop()  # COMMENTED OUT to allow other tabs to execute

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
            docs_hash = get_uploaded_docs_hash()
            normalized_results = get_normalized_requirements(docs_hash)

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
        docs_hash = get_uploaded_docs_hash()
        clarity_rows = get_clarity_results(docs_hash)
        clarity_map = { (c["Requirement"], c["Source"]): c for c in clarity_rows }

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
            try:
                from analysis.testgen import generate_test_ideas
            except ImportError as e:
                st.error(f"Test generation not available: {e}")
                st.stop()

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
                    st.success(f"Generated {len(all_export_rows)} test scenarios!")
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
            visible_sections.append(("ready", ready_reqs, "Ready"))
        if show_provisional and provisional_reqs:
            visible_sections.append(("provisional", provisional_reqs, "Provisional"))
        if show_blocked and blocked_reqs:
            visible_sections.append(("blocked", blocked_reqs, "Blocked"))

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
        st.markdown("## Traceability Matrix")

        try:
            # Lazy import for traceability
            from analysis.traceability import build_trace_matrix

            requirement_rows = get_requirement_rows()
            if not requirement_rows:
                st.info("No requirements detected.")
                st.stop()

            # Filter requirements based on search query
            if search_traceability:
                search_terms = search_traceability.lower().split()
                filtered_requirement_rows = []

                # Get the original normalized data for more comprehensive search
                docs_hash = get_uploaded_docs_hash()
                normalized_results = get_normalized_requirements(docs_hash)

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
    if not st.session_state.uploaded_docs:
        st.info("Please upload and process documents to view the requirements dashboard.")
        # st.stop()  # COMMENTED OUT to allow other tabs to execute

    st.markdown("## Requirements Dashboard")

    try:
        docs_hash = get_uploaded_docs_hash()
        results = get_normalized_requirements(docs_hash)
        total_reqs = len(results) if results else 0

        if total_reqs == 0:
            st.warning("No requirements data available. Please load requirements in the Search tab first.")
        else:
            # === KPI CARDS ROW ===
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(
                    label="Total Requirements",
                    value=total_reqs,
                    help="Total number of requirements processed"
                )

            # Calculate clarity metrics
            docs_hash = get_uploaded_docs_hash()
            clarity_rows = get_clarity_results(docs_hash)
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
                    label="Avg Clarity Score",
                    value=f"{avg_clarity:.1f}",
                    delta=None,  # Remove confusing delta calculation
                    help="Average clarity score (1-10 scale, higher is better)"
                )

            with col3:
                tbd_pct = 100 * issue_counts['TBD'] / total_reqs if total_reqs else 0
                st.metric(
                    label="TBD Issues",
                    value=f"{tbd_pct:.1f}%",
                    help="Percentage of requirements with TBD (To Be Determined) content"
                )

            with col4:
                ambiguous_pct = 100 * issue_counts['Ambiguous'] / total_reqs if total_reqs else 0
                st.metric(
                    label="Ambiguous",
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
                    label="Test Coverage",
                    value=f"{coverage_pct:.1f}%",
                    help="Percentage of system requirements covered by tests"
                )

            st.markdown("---")

            # === COVERAGE VISUALIZATION BAR ===
            st.markdown("### Test Coverage Overview")

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
                st.info("No system requirements found for coverage analysis")

            st.markdown("---")

            # === SIDE-BY-SIDE CHARTS ===
            chart_col1, chart_col2 = st.columns(2)

            # Left chart: Quality Issues Distribution
            with chart_col1:
                st.markdown("### Quality Issues Distribution")

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
                    st.markdown('<p class="breakdown-header">Issue Breakdown:</p>', unsafe_allow_html=True)
                    for _, row in issue_data.iterrows():
                        st.caption(f"• {row['Issue Type']}: {row['Count']} ({row['Percentage']:.1f}%)")
                else:
                    st.success("No quality issues detected!")

            # Right chart: Requirement Types Distribution
            with chart_col2:
                st.markdown("### Requirement Types Distribution")

                all_categories = [cat for r in results for cat in r.get("categories", [])]
                if all_categories:
                    cat_series = pd.Series(all_categories)
                    cat_counts = cat_series.value_counts()

                    st.bar_chart(cat_counts)

                    # Show percentages - all types without expander
                    st.markdown('<p class="breakdown-header">Type Breakdown:</p>', unsafe_allow_html=True)
                    for cat_type, count in cat_counts.items():  # Show all types
                        pct = 100 * count / total_reqs
                        st.caption(f"• {cat_type}: {count} ({pct:.1f}%)")
                else:
                    st.info("No requirement type categories available")

            # === SUMMARY INSIGHTS ===
            st.markdown("---")
            st.markdown('<h2 class="key-insights-header">Key Insights</h2>', unsafe_allow_html=True)

            insights = []

            if avg_clarity < 5:
                insights.append("**Low clarity scores** - Consider reviewing requirement definitions")
            elif avg_clarity > 7:
                insights.append("**Good clarity scores** - Requirements are well-defined")

            if tbd_pct > 20:
                insights.append("**High TBD content** - Many requirements need further definition")
            elif tbd_pct == 0:
                insights.append("**No TBD content** - All requirements are fully defined")

            if coverage_pct < 50:
                insights.append("**Low test coverage** - Consider adding more test scenarios")
            elif coverage_pct > 80:
                insights.append("**Excellent test coverage** - Most requirements are tested")

            if len(insights) == 0:
                insights.append("**Good overall status** - Requirements are in decent shape")

            for insight in insights:
                st.markdown(insight)

    except Exception as e:
        st.error(f"Dashboard error: {e}")
        if DEBUG:
            st.exception(e)



