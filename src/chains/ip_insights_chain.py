from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.services.langchain_llm import get_langchain_gemini_model


def summarize_ip_insights(
    grounded_context: str,
    model_name: str | None = None,
) -> str:
    if not grounded_context.strip():
        return "Insufficient grounded evidence for IP insights."

    prompt = ChatPromptTemplate.from_template(
        "You are a music industry analyst focused on IP, catalog, and "
        "export insights.\n"
        "The evidence below is grouped by source section, separated by "
        "headers like '=== Import / Export ===' or '### <Section>'. One or more sections may "
        "be present (typically Import / Export and/or Streaming Atlas).\n"
        "If multiple sections are present, produce a balanced summary "
        "that draws from EVERY contributing section; do not over-weight "
        "any single section.\n"
        "Output exactly 4-6 bullets. If multiple sections are present, "
        "include at least one bullet reflecting each. If only one section "
        "is present, all bullets should draw from it.\n"
        "Focus on catalog dynamics (current vs catalog age, deep catalog "
        "share, era effects), export power and rankings, local-vs-foreign "
        "streaming shares, and cross-border genre flows.\n"
        "Use only the evidence below; do not add new facts.\n"
        "When the evidence includes figures or percentages, retain them with units.\n"
        "Use concise, executive-friendly wording.\n"
        "Do not mention the section headers or the '===' separators in "
        "the output bullets.\n\n"
        "Evidence:\n{grounded_context}\n"
    )
    llm = get_langchain_gemini_model(model_name=model_name, temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"grounded_context": grounded_context}).strip()
