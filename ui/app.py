import os
import streamlit as st
from dotenv import load_dotenv
from ingestion.loader import load_documents
from analysis.index import build_index
from analysis.qa import make_qa

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

st.title("🔎 RAG MVP")
st.caption("Streamlit UI • LangChain • Unstructured • OpenAI/HF")

col1, col2 = st.columns([3, 1])
with col2:
    st.write("**Settings**")
    st.selectbox("LLM Provider", ["openai", "hf"], index=0, key="provider")
    st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), key="openai_model")
    st.text_input("HF embed model", value=_get_embed_model(), key="embed_model")

index = get_index()
if index is None:
    st.error("Failed to build the document index. Please check your embedding model, input data, and logs for errors.")
    print("Failed to build the document index. Please check your embedding model, input data, and logs for errors.")
    st.stop()

try:
    retriever = index.as_retriever(search_kwargs={"k": 4})
    print("Retriever created:", retriever)
except Exception as e:
    st.error(f"Failed to create retriever: {e}")
    print(f"Failed to create retriever: {e}")
    st.stop()

qa = make_qa(retriever)
print("QA chain created:", qa)
if qa is None:
    st.error("QA chain was not created. Please check your retriever and LLM setup.")
    st.stop()

query = col1.text_input("Ask a question about the files in `data/`")
if query:
    try:
        with st.spinner("Thinking…"):
            out = qa({"query": query})
        st.subheader("Answer")
        st.write(out.get("result", "No answer returned."))
        st.subheader("Sources")
        for d in out.get("source_documents", []):
            st.write("•", d.metadata.get("source", "unknown"))
    except Exception as e:
        st.error(f"An error occurred while processing your question: {e}")
        import traceback
        st.text(traceback.format_exc())


