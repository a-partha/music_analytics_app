from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.services.langchain_llm import get_langchain_gemini_model, resolved_strategy_model


def generate_strategy_recommendations(
    analysis_bundle: str,
    model_name: str | None = None,
) -> str:
    if not analysis_bundle.strip():
        return ""

    prompt = ChatPromptTemplate.from_template(
        "You are a music industry strategy advisor for labels, publishers, "
        "and rights holders.\n"
        "The text below is the ONLY source of truth. It contains prior "
        "analysis snippets from an industry report, including DTC/IP "
        "section summaries. "
        "Do not invent facts or sections not supported by that text.\n"
        "Produce 3 to 5 strategic recommendations when the source material "
        "supports them; if the text is too thin for three grounded items, "
        "output only as many as you can fully support (never pad with guesses).\n"
        "For each recommendation, use EXACTLY this Markdown structure "
        "(repeat for each item, numbering 1..N):\n\n"
        "## Recommendation N\n"
        "**Title:** <short headline>\n"
        "**Insight:** What is happening in the market or listener behavior "
        "(one or two sentences), grounded in the source.\n"
        "**Evidence:** Name the report section (or topic) and state what it "
        "claims that supports this insight. No invented sections.\n"
        "**Action:** What a label, publisher, or rights holder should do "
        "next (one or two concrete sentences).\n\n"
        "Rules:\n"
        "- If the source material is thin, output fewer recommendations rather "
        "than inventing support.\n"
        "- Each recommendation must cover a NON-OVERLAPPING angle.\n"
        "- Evidence must reference section names and claims present in the source material.\n"
        "- Do not mention that you are following a template.\n\n"
        "Source material:\n\n"
        "{analysis_bundle}\n"
    )
    final_model = resolved_strategy_model(model_name)
    llm = get_langchain_gemini_model(model_name=final_model, temperature=0.2)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"analysis_bundle": analysis_bundle}).strip()
