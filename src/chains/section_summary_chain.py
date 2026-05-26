from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.services.langchain_llm import get_langchain_gemini_model


def summarize_section_from_context(
    section_name: str,
    grounded_context: str,
    model_name: str | None = None,
) -> str:
    if not grounded_context.strip():
        return "Insufficient grounded evidence for this section."

    prompt = ChatPromptTemplate.from_template(
        "You are a music industry analyst.\n"
        "The evidence below is grouped into EARLY, MIDDLE, and LATE "
        "buckets representing the start, middle, and end of the section.\n"
        "Produce a balanced summary that draws from ALL THREE buckets so "
        "the full section is represented; do not over-weight EARLY.\n"
        "Output exactly 3-4 bullets. At minimum, include one bullet that "
        "reflects EARLY content, one that reflects MIDDLE content, and "
        "one that reflects LATE content. A 4th bullet may synthesize "
        "across buckets.\n"
        "Use only the evidence below; do not add new facts.\n"
        "When the evidence includes figures or percentages, retain them with units.\n"
        "Use concise, executive-friendly wording.\n"
        "Do not mention the bucket labels in the output bullets.\n\n"
        "Section: {section_name}\n"
        "Context:\n{grounded_context}\n"
    )
    llm = get_langchain_gemini_model(model_name=model_name, temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "section_name": section_name,
            "grounded_context": grounded_context,
        }
    ).strip()
