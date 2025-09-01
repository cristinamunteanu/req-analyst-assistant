import streamlit as st

query = st.text_input("Ask a question about the files in `data/`", key="user_query")
st.write(f"Query value: '{query}' (type: {type(query)})")
if query:
    st.write("You asked:", query)