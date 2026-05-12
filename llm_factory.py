"""
llm_factory.py
--------------
Single source of truth for instantiating the LLM and embedding model.
Switching providers happens in one place: change LLM_PROVIDER in .env.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Local, free, no-API-key-required embeddings.
    First call downloads the model (~80MB) into the HF cache.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_llm():
    """
    Returns a chat model based on LLM_PROVIDER.
    - "groq":   cloud, free tier, very fast (Llama 3.3 70B)
    - "ollama": fully local, requires `ollama serve` running
    """
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "and add it to your .env file, OR switch LLM_PROVIDER=ollama."
            )
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.2,
            max_tokens=2048,
        )

    elif LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{LLM_PROVIDER}'. Use 'groq' or 'ollama'."
        )
