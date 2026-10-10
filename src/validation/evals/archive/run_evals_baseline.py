"""Archived: section-locked baseline, superseded by run_evals.py --section-locked."""
raise SystemExit("Archived; use run_evals.py --section-locked")

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

# Must be set before deepeval is imported. Rate-limit retries below can
# push a single test case past DeepEval's 180s per-test-case default.
os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")
# Backoff waits of 5, 10, 20, 30s (~65s total) outlast a per-minute 429 window.
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "5")
os.environ.setdefault("DEEPEVAL_RETRY_INITIAL_SECONDS", "5")
os.environ.setdefault("DEEPEVAL_RETRY_CAP_SECONDS", "30")

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.evaluate import AsyncConfig, ErrorConfig
from deepeval.models import LocalModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

from src.services.file_search_retrieval import (
    RetrievalError,
    retrieve_query_chunks,
)
from src.services.file_search_store import CACHE_VERSION_TAG, SPLITTER_MODE
from src.services.section_cache import list_cached_entries

load_dotenv()

for _key in ("GEMINI_API_KEY", "NVIDIA_API_KEY"):
  if not os.getenv(_key):
    raise RuntimeError(f"{_key} is missing. Add it to .env.")

# Set to None to evaluate every pair.
_ONLY_ID = None
#"consumption_metrics_03"

# 1. Load Data
_QA_PATH = os.path.join(os.path.dirname(__file__), "static_qa.json")
with open(_QA_PATH, encoding="utf-8") as qa_file:
  qa_data = json.load(qa_file)
qa_items = [
    pair for pair in qa_data["pairs"]
    if _ONLY_ID is None or pair["id"] == _ONLY_ID
]
if not qa_items:
  raise RuntimeError(f"No QA pair with id {_ONLY_ID!r} in {_QA_PATH}.")


def _match_subsection(section, manifest):
  """Return the one manifest row for a QA section name, or None."""
  wanted = section.strip().upper()
  exact = [
      row for row in manifest
      if row.get("display_title", "").strip().upper() == wanted
  ]
  if len(exact) == 1:
    return exact[0]
  contains = [
      row for row in manifest
      if wanted in row.get("display_title", "").upper()
  ]
  return contains[0] if len(contains) == 1 else None


def _resolve_store(sections):
  """Pick the single cached store whose manifest covers every QA section.

  Reads section_cache.json from disk only, so a missing or ambiguous store
  stops the run before any Gemini call.
  """
  suffix = f":{SPLITTER_MODE}:{CACHE_VERSION_TAG}"
  candidates = {
      key: entry for key, entry in list_cached_entries().items()
      if key.endswith(suffix)
  }
  matches = [
      (key, entry) for key, entry in candidates.items()
      if all(
          _match_subsection(section, entry.get("manifest") or [])
          for section in sections
      )
  ]
  if len(matches) != 1:
    raise RuntimeError(
        f"Expected exactly one '{suffix}' cache row covering sections "
        f"{sections}, found {len(matches)}. Considered keys: "
        f"{sorted(candidates) or 'none'}. Index the year-end PDF in the app "
        "first; this script does not upload."
    )
  return matches[0][1]


_store_entry = _resolve_store(qa_data["sections_found"])
_STORE_NAME = _store_entry["store_name"]
_MANIFEST = _store_entry["manifest"]

# 2. Configure Models
EVAL_RETRIEVAL_MODEL = "gemini-3.1-flash-lite"
generator_model = EVAL_RETRIEVAL_MODEL
# Stays under the Gemini free tier's 15 requests per minute.
generator_llm = ChatGoogleGenerativeAI(
    model=generator_model,
    temperature=0.0,
    rate_limiter=InMemoryRateLimiter(requests_per_second=12 / 60),
)
# Limits every judge call, not just test-case starts, so a slow case can't
# cause a burst when the queued cases behind it run back to back.
_JUDGE_RATE_LIMITER = InMemoryRateLimiter(requests_per_second=30 / 60)


class RateLimitedLocalModel(LocalModel):

  def generate(self, *args, **kwargs):
    _JUDGE_RATE_LIMITER.acquire()
    return super().generate(*args, **kwargs)

  async def a_generate(self, *args, **kwargs):
    await _JUDGE_RATE_LIMITER.aacquire()
    return await super().a_generate(*args, **kwargs)


judge_model = RateLimitedLocalModel(
    model="nvidia/nemotron-3-super-120b-a12b",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
    temperature=0,
    # DeepEval timeouts are disabled above; without this a hung call waits
    # the openai client's 600s default. Timeouts are retried like 429s.
    timeout=60,
    generation_kwargs={
        "response_format": {"type": "json_object"},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
)

metrics = [
    FaithfulnessMetric(model=judge_model, threshold=0.7),
    AnswerRelevancyMetric(model=judge_model, threshold=0.7),
    ContextualRecallMetric(model=judge_model, threshold=0.7),
]

# 3. Generate Answers & Build DeepEval Test Cases
_ANSWERS_PATH = os.path.join(
    os.path.dirname(__file__), "generated_answers.json"
)
answer_cache = {}
if os.path.exists(_ANSWERS_PATH):
  with open(_ANSWERS_PATH, encoding="utf-8") as answers_file:
    answer_cache = json.load(answers_file)

test_cases = []
items_by_question = {}

for item in qa_items:
  query = item["question"]

  manifest_row = _match_subsection(item["section"], _MANIFEST)
  if manifest_row is None:
    raise RuntimeError(
        f"[{item['id']}] Section {item['section']!r} does not map to exactly "
        "one manifest title. Candidates: "
        f"{[row.get('display_title') for row in _MANIFEST]}"
    )
  subsection_key = manifest_row["subsection_key"]

  try:
    retrieved_chunks = retrieve_query_chunks(
        file_search_store_name=_STORE_NAME,
        query=query,
        subsection_key=subsection_key,
        model_name=generator_model,
    )
  except RetrievalError as exc:
    raise RuntimeError(
        f"[{item['id']}] Retrieval failed for subsection_key "
        f"{subsection_key!r}: {exc}"
    ) from exc
  if not retrieved_chunks:
    raise RuntimeError(
        f"[{item['id']}] Retrieval failed for subsection_key "
        f"{subsection_key!r}: grounding_chunks empty."
    )
  retrieved_chunk_texts = [chunk.text for chunk in retrieved_chunks]
  context = "\n\n".join(retrieved_chunk_texts)
  context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()

  prompt = (
      "Answer the question using only the provided context. If the context "
      "does not contain enough information, state that clearly. Answer in "
      "complete, self-contained sentences.\n"
      f"Question: {query}\n"
      f"Context:\n{context}"
  )

  cached = answer_cache.get(item["id"], {})
  if (
      cached.get("model") == generator_model
      and cached.get("prompt") == prompt
      and cached.get("context_hash") == context_hash
  ):
    generated_answer = cached["answer"]
  else:
    response = generator_llm.invoke(prompt)

    if isinstance(response.content, list):
      answer_text = "".join(
          c["text"] for c in response.content if "text" in c
      )
    else:
      answer_text = response.content

    generated_answer = answer_text.strip()
    # Empty answers (e.g. safety blocks) aren't cached so the next run retries.
    if not generated_answer:
      raise RuntimeError(
          f"[{item['id']}] Generator {generator_model} returned an empty "
          "answer (possible safety block)."
      )
    answer_cache[item["id"]] = {
        "model": generator_model,
        "prompt": prompt,
        "context_hash": context_hash,
        "answer": generated_answer,
    }
    # Saved per answer so a crash mid-loop keeps what was already generated;
    # the temp-file swap keeps an interrupted write from corrupting the cache.
    _tmp_path = f"{_ANSWERS_PATH}.tmp"
    with open(_tmp_path, "w", encoding="utf-8") as answers_file:
      json.dump(answer_cache, answers_file, indent=2)
    os.replace(_tmp_path, _ANSWERS_PATH)

  test_case = LLMTestCase(
      input=query,
      actual_output=generated_answer,
      expected_output=item["ground_truth"],
      retrieval_context=retrieved_chunk_texts,
  )
  test_cases.append(test_case)
  items_by_question[query] = item

# 4. Run Evaluation
# Pacing comes from _JUDGE_RATE_LIMITER (30 RPM, under NVIDIA's 40 RPM).
results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(max_concurrent=1),
    # A single smoke item should stop on the first judge error, not finish
    # the run with a silently missing metric.
    error_config=ErrorConfig(ignore_errors=_ONLY_ID is None),
)

# 5. Format and Export Review Results
review_rows = []
for test_result in results.test_results:
  item = items_by_question.get(test_result.input, {})
  metric_rows = {}
  for metric_data in test_result.metrics_data or []:
    metric_rows[metric_data.name] = {
        "score": metric_data.score,
        "threshold": metric_data.threshold,
        "passed": metric_data.success,
        "reason": metric_data.reason,
        "error": metric_data.error,
    }
  review_rows.append({
      "id": item.get("id"),
      "question": test_result.input,
      "passed": test_result.success,
      "metrics": metric_rows,
      "ground_truth": test_result.expected_output,
      "retrieved_context": test_result.retrieval_context,
      "answer": test_result.actual_output,
  })

_RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "smoke_test_results.json"
)
run = {
    "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "results": review_rows,
}
with open(_RESULTS_PATH, "w", encoding="utf-8") as results_file:
  json.dump(run, results_file, indent=2)
