from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.models import resolve_model

def resolved_analysis_model(explicit: str | None) -> str:
    return resolve_model("ANALYSIS_MODEL", explicit)

def resolved_strategy_model(explicit: str | None) -> str:
    return resolve_model("STRATEGY_MODEL", explicit)

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
