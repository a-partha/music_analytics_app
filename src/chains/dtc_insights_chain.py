from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.services.langchain_llm import get_langchain_gemini_model


def summarize_dtc_insights(
    grounded_context: str,
    model_name: str | None = None,
) -> str:
    if not grounded_context.strip():
        return "Insufficient grounded evidence for DTC insights."

    prompt = ChatPromptTemplate.from_template(
        "You are a music industry analyst focused on fan/engagement (DTC) "
        "insights.\n"
        "The evidence below may come from one or more report sections and "
        "may include headers like '=== <Section> ==='. If multiple sections "
        "are present, synthesize them without assuming a single source "
        "section.\n"
        "Output exactly 4-6 bullets. Focus on superfans, engagement "
        "patterns, platforms, behaviors, discovery channels (gaming, "
        "podcasts, short-form video, social platforms), and direct "
        "artist-to-fan monetization.\n"
        "Use only the evidence below; do not add new facts.\n"
        "When the evidence includes figures or percentages, retain them with units.\n"
        "Use concise, executive-friendly wording.\n\n"
        "Evidence:\n{grounded_context}\n"
    )
    llm = get_langchain_gemini_model(model_name=model_name, temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"grounded_context": grounded_context}).strip()
