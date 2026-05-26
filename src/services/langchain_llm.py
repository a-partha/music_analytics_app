from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_GEMINI_ANALYSIS_MODEL = "gemini-3.1-flash-lite"
DEFAULT_GEMINI_STRATEGY_MODEL = "gemini-3.1-pro"

def resolved_analysis_model(explicit: str | None) -> str:
    load_dotenv()
    return explicit or os.getenv("GEMINI_ANALYSIS_MODEL") or os.getenv("GEMINI_SYNTHESIS_MODEL") or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_ANALYSIS_MODEL

def resolved_strategy_model(explicit: str | None) -> str:
    load_dotenv()
    return explicit or os.getenv("GEMINI_STRATEGY_MODEL") or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_STRATEGY_MODEL

def get_langchain_gemini_model(
    model_name: str | None = None,
    temperature: float = 0.2,
) -> ChatGoogleGenerativeAI:
    load_dotenv()
    resolved_model = model_name or resolved_analysis_model(None)
    api_key = os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=resolved_model,
        temperature=temperature,
        google_api_key=api_key,
    )
