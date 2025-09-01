"""
qa.py

This module provides utilities for building a retrieval-augmented question answering (QA) chain
using LangChain, supporting multiple LLM providers (OpenAI, HuggingFace, Anthropic, Ollama).

Main Components:
----------------
- SYSTEM: System prompt for the assistant, instructing it to cite sources by file path.
- TEMPLATE: Prompt template for formatting questions and context for the LLM.
- make_llm(): Factory function to instantiate an LLM client based on the LLM_PROVIDER environment variable.
              Supports OpenAI, HuggingFace, Anthropic, and Ollama providers.
              Handles import and initialization errors gracefully.
- make_qa(retriever): Builds a RetrievalQA chain using the selected LLM, a retriever, and the custom prompt.
                      Returns answers along with the source documents used.

Environment Variables:
----------------------
- LLM_PROVIDER: Selects the LLM backend ("openai", "huggingface", "anthropic", "ollama").
- OPENAI_MODEL: Model name for OpenAI (default: "gpt-4o-mini").
- HF_CHAT_MODEL: Model repo ID for HuggingFace (default: "mistralai/Mixtral-8x7B-Instruct-v0.1").
- ANTHROPIC_MODEL: Model name for Anthropic (default: "claude-3-haiku-20240307").
- OLLAMA_MODEL: Model name for Ollama (default: "llama3").

Usage Example:
--------------
    retriever = ...  # Your retriever instance
    qa_chain = make_qa(retriever)
    result = qa_chain({"question": "What is the project about?"})
    print(result["result"])
    print(result["source_documents"])
"""

import os
from typing import Any, Dict
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

SYSTEM = "You are a careful assistant. Cite sources by file path where relevant."

TEMPLATE = """{system}
Question: {question}

Use the context to answer concisely and cite sources.

Context:
{context}
"""

def make_llm():
    """
    Factory function to instantiate a language model (LLM) client based on the LLM_PROVIDER environment variable.

    Supported providers:
        - "openai": Uses langchain_openai.ChatOpenAI. Model name set via OPENAI_MODEL (default: "gpt-4o-mini").
        - "huggingface": Uses langchain_community.llms.HuggingFaceHub. Model repo set via HF_CHAT_MODEL (default: "mistralai/Mixtral-8x7B-Instruct-v0.1").
        - "anthropic": Uses langchain_community.llms.Anthropic. Model name set via ANTHROPIC_MODEL (default: "claude-3-haiku-20240307").
        - "ollama": Uses langchain_community.llms.Ollama. Model name set via OLLAMA_MODEL (default: "llama3").

    The function handles import and initialization errors gracefully, printing an error message and re-raising the exception.

    Returns:
        An initialized LLM client instance for the selected provider.

    Raises:
        ValueError: If the LLM_PROVIDER is unknown.
        ImportError: If the required provider package is not installed.
        Exception: For other initialization errors.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return ChatOpenAI(model=model, temperature=0)
        elif provider == "huggingface":
            from langchain_community.llms import HuggingFaceHub
            repo_id = os.getenv("HF_CHAT_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
            return HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0})
        elif provider == "anthropic":
            from langchain_community.llms import Anthropic
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
            return Anthropic(model=model, temperature=0)
        elif provider == "ollama":
            from langchain_community.llms import Ollama
            model = os.getenv("OLLAMA_MODEL", "llama3")
            return Ollama(model=model, temperature=0)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
    except ImportError as e:
        print(f"Failed to import LLM provider '{provider}': {e}")
        raise
    except Exception as e:
        print(f"Error initializing LLM for provider '{provider}': {e}")
        raise

def make_qa(retriever) -> Any:
    """
    Builds a RetrievalQA chain using the selected language model (LLM), a retriever, and a custom prompt.

    The chain uses the LLM to answer questions based on context retrieved by the retriever.
    The prompt instructs the assistant to answer concisely and cite sources by file path.

    Args:
        retriever: A retriever instance compatible with LangChain, used to fetch relevant documents/context.

    Returns:
        RetrievalQA: A LangChain RetrievalQA chain configured with the LLM, retriever, and prompt.
                     The chain returns both the answer and the source documents used.

    Raises:
        Exception: If LLM initialization or chain construction fails.

    Example:
        retriever = ...  # Your retriever instance
        qa_chain = make_qa(retriever)
        result = qa_chain({"question": "What is the project about?"})
        print(result["result"])
        print(result["source_documents"])
    """
    try:
        prompt = PromptTemplate(
            input_variables=["system", "question", "context"],
            template=TEMPLATE,
            partial_variables={"system": SYSTEM},
        )
        llm = make_llm()
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
    except Exception as e:
        print(f"Error initializing RetrievalQA chain: {e}")
        raise
