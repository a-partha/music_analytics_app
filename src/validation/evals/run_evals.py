import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# This file must stay at src/validation/evals/run_evals.py; parents[3] is the
# repo root that holds the src package.
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

def _force_utf8_stdio():
  """Let DeepEval's checkmark and sparkle print on a cp1252 Windows console."""
  for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
      continue
    try:
      reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
      pass


_force_utf8_stdio()

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
from google import genai
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.models import resolve_model
from src.services.file_search_retrieval import (
    RetrievalError,
    retrieve_subsection_evidence_with_usage,
)
from src.services.file_search_store import CACHE_VERSION_TAG, SPLITTER_MODE
from src.services.section_cache import list_cached_entries

load_dotenv()

# Set to a tuple of ids to evaluate only those pairs.
_ONLY_IDS = None
_ARGS = sys.argv[1:]
# The default is a normal run which regenerates every answer, 
# so generation cost is measured.
# Pass --use-cache to reuse a saved answer when the retrieved text matches.
_NO_CACHE = "--use-cache" not in _ARGS
# Section lock is required. Every retrieval filters on source_filename and
# subsection_key. Passing --section-locked changes nothing.
_SECTION_LOCKED = True
# Rebuild the summary of an existing results file. No API calls.
_SUMMARIZE_ONLY = "--summarize-only" in _ARGS

_EVAL_DIR = Path(__file__).resolve().parent
_QA_PATH = _EVAL_DIR / "static_qa.json"
_ANSWERS_PATH = _EVAL_DIR / "generated_answers.json"
_MANUAL_REVIEW_PATH = _EVAL_DIR / "manual_review.json"


def _results_path():
  # Section lock is always on. This name keeps the cold and smoke reports.
  return _EVAL_DIR / "testing_report.json"


def _cache_key(item_id):
  if _SECTION_LOCKED:
    return f"{item_id}:section_locked"
  return item_id


_RESULTS_PATH = _results_path()

# Each eval model falls back to FALLBACK_MODEL, the app's retrieval model.
EVAL_RETRIEVAL_MODEL = resolve_model("EVAL_RETRIEVAL_MODEL")
GENERATOR_MODEL = resolve_model("EVAL_GENERATOR_MODEL")
# FALLBACK_MODEL is a Gemini model, so it cannot back up the NVIDIA judge.
JUDGE_MODEL = (os.getenv("EVAL_JUDGE_MODEL") or "").strip()
if not JUDGE_MODEL:
  raise RuntimeError("EVAL_JUDGE_MODEL is missing. Add it to .env.")
JUDGE_BASE_URL = "https://integrate.api.nvidia.com/v1"
METRIC_THRESHOLD = 0.7

# USD per token. Split by retrieval model (gemini-3.1-flash-lite),
# generator model (gemini-3.5-flash-lite), and judge (nemotron-3-super-120b).
RETRIEVAL_INPUT_RATE = 0.25 / 1_000_000
RETRIEVAL_OUTPUT_RATE = 1.50 / 1_000_000
GENERATOR_INPUT_RATE = 0.30 / 1_000_000
GENERATOR_OUTPUT_RATE = 2.50 / 1_000_000
JUDGE_INPUT_RATE = 0.08 / 1_000_000
JUDGE_OUTPUT_RATE = 0.45 / 1_000_000

_INPUT_TOKEN_KEYS = ("input_tokens", "prompt_token_count", "prompt_tokens")
_OUTPUT_TOKEN_KEYS = (
    "output_tokens",
    "candidates_token_count",
    "completion_tokens",
)

# Stays under the Gemini free tier's 15 requests per minute. Acquired before
# the generation timer starts so rate-limit waits are not reported as latency.
_GENERATOR_RATE_LIMITER = InMemoryRateLimiter(requests_per_second=12 / 60)
# Limits every judge call, not just test-case starts, so a slow case can't
# cause a burst when the queued cases behind it run back to back.
_JUDGE_RATE_LIMITER = InMemoryRateLimiter(requests_per_second=30 / 60)


def _load_qa_items():
  with open(_QA_PATH, encoding="utf-8") as qa_file:
    qa_data = json.load(qa_file)
  qa_items = [
      pair for pair in qa_data["pairs"]
      if _ONLY_IDS is None or pair["id"] in _ONLY_IDS
  ]
  if not qa_items:
    raise RuntimeError(f"No QA pair with id in {_ONLY_IDS!r} in {_QA_PATH}.")
  return qa_data["sections_found"], qa_items


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


def _resolve_source_filename(store_entry):
  """source_filename metadata on the store's uploaded sections (read-only)."""
  doc_names = list((store_entry.get("section_doc_names") or {}).values())
  if not doc_names:
    raise RuntimeError("Cached store row has no section documents.")
  client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
  document = client.file_search_stores.documents.get(name=doc_names[0])
  for meta in getattr(document, "custom_metadata", None) or []:
    if meta.key == "source_filename" and meta.string_value:
      return meta.string_value
  raise RuntimeError(
      f"Document {doc_names[0]} has no source_filename metadata."
  )


def _usage_field(usage, key):
  if isinstance(usage, dict):
    return usage.get(key)
  return getattr(usage, key, None)


def _first_count(usage, keys):
  for key in keys:
    value = _usage_field(usage, key)
    if value is not None:
      return int(value)
  return None


def usage_to_counts(usage, label):
  """(input, output) from a usage dict or object; raises if unreadable."""
  if usage is None:
    raise RuntimeError(f"{label}: usage_metadata is missing.")
  input_tokens = _first_count(usage, _INPUT_TOKEN_KEYS)
  output_tokens = _first_count(usage, _OUTPUT_TOKEN_KEYS)
  if input_tokens is None or output_tokens is None:
    raise RuntimeError(
        f"{label}: usage_metadata has no readable token counts. Expected "
        f"one of {_INPUT_TOKEN_KEYS} and one of {_OUTPUT_TOKEN_KEYS}; "
        f"got {usage!r}"
    )
  return input_tokens, output_tokens


def extract_token_counts(response, label):
  usage = getattr(response, "usage_metadata", None)
  if usage is None:
    usage = (getattr(response, "response_metadata", None) or {}).get(
        "usage_metadata"
    )
  return usage_to_counts(usage, label)


def _retrieval_token_counts(usage, label):
  input_tokens, output_tokens = usage_to_counts(usage, label)
  # File Search bills retrieved chunks as tool-use input and any thinking as
  # output; neither is inside prompt_token_count / candidates_token_count.
  input_tokens += _usage_field(usage, "tool_use_prompt_token_count") or 0
  output_tokens += _usage_field(usage, "thoughts_token_count") or 0
  return input_tokens, output_tokens


def _retrieval_cost(input_tokens, output_tokens):
  return (
      (input_tokens or 0) * RETRIEVAL_INPUT_RATE
      + (output_tokens or 0) * RETRIEVAL_OUTPUT_RATE
  )


def _generator_cost(input_tokens, output_tokens):
  return (
      (input_tokens or 0) * GENERATOR_INPUT_RATE
      + (output_tokens or 0) * GENERATOR_OUTPUT_RATE
  )


def _ms(start, end):
  return round((end - start) * 1000, 2)


def _strict_json_schema(node):
  """OpenAI strict mode rejects objects that omit these two fields."""
  if isinstance(node, dict):
    properties = node.get("properties")
    if node.get("type") == "object" or isinstance(properties, dict):
      node["additionalProperties"] = False
      if isinstance(properties, dict):
        node["required"] = list(properties)
    for value in node.values():
      _strict_json_schema(value)
  elif isinstance(node, list):
    for value in node:
      _strict_json_schema(value)
  return node


class RateLimitedLocalModel(LocalModel):
  """Rate-limited judge that records each API call under the active item.

  LocalModel builds a new OpenAI client on every load_model call, so each
  client is wrapped exactly once and wrappers never stack.

  _use_schema swaps generation_kwargs for the length of one call, so the
  evaluation must run with max_concurrent=1.
  """

  def __init__(self, *args, **kwargs):
    self.active_id = None
    self.calls = defaultdict(list)
    super().__init__(*args, **kwargs)

  def generate(self, *args, **kwargs):
    _JUDGE_RATE_LIMITER.acquire()
    schema = kwargs.get("schema")
    self._use_schema(schema)
    try:
      return super().generate(*args, **kwargs)
    finally:
      self._restore_response_format(schema)

  async def a_generate(self, *args, **kwargs):
    await _JUDGE_RATE_LIMITER.aacquire()
    schema = kwargs.get("schema")
    self._use_schema(schema)
    try:
      return await super().a_generate(*args, **kwargs)
    finally:
      self._restore_response_format(schema)

  def _use_schema(self, schema):
    if schema is None:
      return
    self._saved_response_format = self.generation_kwargs.get("response_format")
    self.generation_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": _strict_json_schema(schema.model_json_schema()),
            "strict": True,
        },
    }

  def _restore_response_format(self, schema):
    if schema is None:
      return
    saved = getattr(self, "_saved_response_format", None)
    if saved is None:
      self.generation_kwargs.pop("response_format", None)
    else:
      self.generation_kwargs["response_format"] = saved

  def load_model(self, async_mode=False):
    client = super().load_model(async_mode=async_mode)
    completions = client.chat.completions
    create = completions.create

    if async_mode:
      async def timed_create(*args, **kwargs):
        start = time.perf_counter()
        response = None
        try:
          response = await create(*args, **kwargs)
          return response
        finally:
          self._record(start, response)
    else:
      def timed_create(*args, **kwargs):
        start = time.perf_counter()
        response = None
        try:
          response = create(*args, **kwargs)
          return response
        finally:
          self._record(start, response)

    completions.create = timed_create
    return client

  def _record(self, start, response):
    usage = getattr(response, "usage", None)
    self.calls[self.active_id].append({
        "ms": (time.perf_counter() - start) * 1000,
        "input": getattr(usage, "prompt_tokens", None) if usage else None,
        "output": getattr(usage, "completion_tokens", None) if usage else None,
    })


def _build_judge():
  return RateLimitedLocalModel(
      model=JUDGE_MODEL,
      base_url=JUDGE_BASE_URL,
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


def _load_answer_cache():
  if not _ANSWERS_PATH.exists():
    return {}
  with open(_ANSWERS_PATH, encoding="utf-8") as answers_file:
    return json.load(answers_file)


def _save_answer_cache(answer_cache):
  # The temp-file swap keeps an interrupted write from corrupting the cache.
  tmp_path = _ANSWERS_PATH.with_suffix(".json.tmp")
  with open(tmp_path, "w", encoding="utf-8") as answers_file:
    json.dump(answer_cache, answers_file, indent=2)
  os.replace(tmp_path, _ANSWERS_PATH)


def _answer_text(response):
  if isinstance(response.content, list):
    return "".join(c["text"] for c in response.content if "text" in c)
  return response.content


def _build_record(item, store_name, source_filename, generator_llm,
                  answer_cache, subsection_key, display_title):
  query = item["question"]
  if (
      not source_filename
      or not str(subsection_key or "").strip()
      or not str(display_title or "").strip()
  ):
    raise RuntimeError(
        f"[{item['id']}] Section-locked retrieval requires source_filename, "
        "subsection_key, and display_title."
    )
  try:
    evidence, search_usage, retrieval_ms = (
        retrieve_subsection_evidence_with_usage(
            file_search_store_name=store_name,
            subsection_key=subsection_key,
            display_title=display_title,
            source_filename=source_filename,
            model_name=EVAL_RETRIEVAL_MODEL,
        )
    )
  except RetrievalError as exc:
    raise RuntimeError(
        f"[{item['id']}] Retrieval failed in {source_filename!r} "
        f"section {display_title!r}: {exc}"
    ) from exc
  retrieval_in, retrieval_out = _retrieval_token_counts(
      search_usage, f"[{item['id']}] File Search"
  )

  context = evidence
  context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()

  prompt = (
      "Answer the question using only the provided context. If the context "
      "does not contain enough information, state that clearly. Answer in "
      "complete, self-contained sentences.\n"
      f"Question: {query}\n"
      f"Context:\n{context}"
  )

  cached = {} if _NO_CACHE else answer_cache.get(_cache_key(item["id"]), {})
  is_cached = (
      cached.get("model") == GENERATOR_MODEL
      and cached.get("prompt") == prompt
      and cached.get("context_hash") == context_hash
  )
  if is_cached:
    generated_answer = cached["answer"]
    generation_ms = 0.0
    generation_in = cached.get("input_tokens")
    generation_out = cached.get("output_tokens")
  else:
    _GENERATOR_RATE_LIMITER.acquire()
    t_gen_start = time.perf_counter()
    response = generator_llm.invoke(prompt)
    generation_ms = _ms(t_gen_start, time.perf_counter())

    generated_answer = _answer_text(response).strip()
    # Empty answers (e.g. safety blocks) aren't cached so the next run retries.
    if not generated_answer:
      raise RuntimeError(
          f"[{item['id']}] Generator {GENERATOR_MODEL} returned an empty "
          "answer (possible safety block)."
      )
    generation_in, generation_out = extract_token_counts(
        response, f"[{item['id']}] Generator {GENERATOR_MODEL}"
    )
    answer_cache[_cache_key(item["id"])] = {
        "model": GENERATOR_MODEL,
        "prompt": prompt,
        "context_hash": context_hash,
        "answer": generated_answer,
        "input_tokens": generation_in,
        "output_tokens": generation_out,
    }
    # Saved per answer so a crash mid-loop keeps what was already generated.
    _save_answer_cache(answer_cache)

  generation_usage_missing = generation_in is None or generation_out is None
  return {
      "item": item,
      "test_case": LLMTestCase(
          input=query,
          actual_output=generated_answer,
          expected_output=item["ground_truth"],
          retrieval_context=[evidence],
      ),
      "cached": is_cached,
      "retrieval_ms": retrieval_ms,
      "generation_ms": generation_ms,
      "retrieval_in": retrieval_in,
      "retrieval_out": retrieval_out,
      "retrieval_cost": _retrieval_cost(retrieval_in, retrieval_out),
      "generation_in": generation_in,
      "generation_out": generation_out,
      "generation_cost": (
          0.0 if generation_usage_missing
          else _generator_cost(generation_in, generation_out)
      ),
      "generation_usage_missing": generation_usage_missing,
  }


def _judge_record(record, judge_model, metrics):
  item_id = record["item"]["id"]
  judge_model.active_id = item_id
  results = evaluate(
      test_cases=[record["test_case"]],
      metrics=metrics,
      async_config=AsyncConfig(max_concurrent=1),
      # A garbled judge JSON is stored on that metric. The other questions
      # still run, and the error is written into the results file.
      error_config=ErrorConfig(ignore_errors=True),
  )
  record["test_result"] = results.test_results[0]

  judge_calls = judge_model.calls.get(item_id, [])
  judge_in = sum(call["input"] or 0 for call in judge_calls)
  judge_out = sum(call["output"] or 0 for call in judge_calls)
  record["judge_ms"] = round(sum(call["ms"] for call in judge_calls), 2)
  record["judge_in"] = judge_in
  record["judge_out"] = judge_out
  record["judge_calls"] = len(judge_calls)
  record["judge_calls_without_usage"] = sum(
      1 for call in judge_calls
      if call["input"] is None or call["output"] is None
  )
  record["judge_cost"] = (
      judge_in * JUDGE_INPUT_RATE + judge_out * JUDGE_OUTPUT_RATE
  )


def _review_row(record):
  test_result = record["test_result"]
  metric_rows = {}
  for metric_data in test_result.metrics_data or []:
    metric_rows[metric_data.name] = {
        "score": metric_data.score,
        "threshold": metric_data.threshold,
        "passed": metric_data.success,
        "reason": metric_data.reason,
        "error": metric_data.error,
    }
  return {
      "id": record["item"]["id"],
      "question": test_result.input,
      "passed": test_result.success,
      "execution_stats": {
          "cached": record["cached"],
          "latency": {
              "retrieval_ms": record["retrieval_ms"],
              "generation_ms": record["generation_ms"],
              "pipeline_ms": round(
                  record["retrieval_ms"] + record["generation_ms"], 2
              ),
              "judge_api_ms": record["judge_ms"],
          },
          "production_usage": {
              "retrieval_tokens": {
                  "input": record["retrieval_in"],
                  "output": record["retrieval_out"],
                  "cost_usd": round(record["retrieval_cost"], 6),
              },
              "generation_tokens": {
                  "input": record["generation_in"],
                  "output": record["generation_out"],
                  "cost_usd": round(record["generation_cost"], 6),
                  "billed_this_run": not record["cached"],
              },
              "total_pipeline_cost_usd": round(
                  record["retrieval_cost"] + record["generation_cost"], 6
              ),
          },
          "judge_usage": {
              "input_tokens": record["judge_in"],
              "output_tokens": record["judge_out"],
              "total_tokens": record["judge_in"] + record["judge_out"],
              "cost_usd": round(record["judge_cost"], 6),
              "api_calls": record["judge_calls"],
              "calls_without_usage": record["judge_calls_without_usage"],
          },
      },
      "metrics": metric_rows,
      "ground_truth": test_result.expected_output,
      "retrieved_context": test_result.retrieval_context,
      "answer": test_result.actual_output,
  }


def _answer_hash(answer):
  return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def _load_manual_review():
  if not _MANUAL_REVIEW_PATH.exists():
    return {}
  with open(_MANUAL_REVIEW_PATH, encoding="utf-8") as review_file:
    return json.load(review_file)


def _distribution(values):
  """mean, median, p95, max. p95 matches the index used by earlier runs."""
  if not values:
    return None
  ordered = sorted(values)
  count = len(ordered)
  if count % 2:
    median = ordered[count // 2]
  else:
    median = (ordered[count // 2 - 1] + ordered[count // 2]) / 2
  return {
      "mean_ms": round(sum(ordered) / count, 2),
      "median_ms": round(median, 2),
      "p95_ms": round(ordered[math.ceil(0.95 * count) - 1], 2),
      "max_ms": round(ordered[-1], 2),
  }


def _mean_tokens(rows, reader):
  counts = [reader(row) for row in rows]
  counts = [count for count in counts if count is not None]
  if not counts:
    return None
  return round(sum(counts) / len(counts), 2)


def _summary(review_rows):
  """Stats from saved result rows, so a rerun and --summarize-only match."""
  total_queries = len(review_rows)
  cached_queries = sum(
      1 for row in review_rows
      if row["execution_stats"]["cached"]
  )
  passed_queries = sum(1 for row in review_rows if row["passed"])
  live_rows = [
      row for row in review_rows
      if not row["execution_stats"]["cached"]
  ]

  def latency(row, key):
    return row["execution_stats"]["latency"][key]

  def usage(row):
    return row["execution_stats"]["production_usage"]

  metric_passes = {}
  for row in review_rows:
    for name, metric in (row.get("metrics") or {}).items():
      bucket = metric_passes.setdefault(name, {"passed": 0, "scored": 0})
      if metric.get("score") is None:
        continue
      bucket["scored"] += 1
      if metric.get("passed"):
        bucket["passed"] += 1

  manual = _load_manual_review()
  reviewed = 0
  unreviewed = 0
  manual_correct = 0
  judge_agreement = 0
  for row in review_rows:
    label = manual.get(row["id"])
    if (
        not label
        or label.get("answer_sha256") != _answer_hash(row.get("answer") or "")
    ):
      unreviewed += 1
      continue
    reviewed += 1
    if label["correct"]:
      manual_correct += 1
    if bool(row["passed"]) == bool(label["correct"]):
      judge_agreement += 1

  retrieval_total = sum(
      usage(row)["retrieval_tokens"]["cost_usd"] for row in review_rows
  )
  generation_total = sum(
      usage(row)["generation_tokens"]["cost_usd"] for row in review_rows
  )
  generation_billed = sum(
      usage(row)["generation_tokens"]["cost_usd"]
      for row in review_rows
      if usage(row)["generation_tokens"].get("billed_this_run")
  )
  judge_total = sum(
      row["execution_stats"]["judge_usage"]["cost_usd"] for row in review_rows
  )
  generation_usage_missing = sum(
      1 for row in review_rows
      if usage(row)["generation_tokens"]["input"] is None
  )

  return {
      "total_queries": total_queries,
      "cached_queries": cached_queries,
      "live_queries": total_queries - cached_queries,
      "passed_queries": passed_queries,
      "pass_rate": (
          round(passed_queries / total_queries, 4) if total_queries else 0.0
      ),
      "metric_passes": metric_passes,
      "manual_review": {
          "reviewed": reviewed,
          "unreviewed": unreviewed,
          "correct": manual_correct,
          "judge_agreement": judge_agreement,
      },
      "generation_usage_missing": generation_usage_missing,
      "latency": {
          "retrieval": _distribution(
              [latency(row, "retrieval_ms") for row in live_rows]
          ),
          "generation": _distribution(
              [latency(row, "generation_ms") for row in live_rows]
          ),
          "pipeline": _distribution(
              [latency(row, "pipeline_ms") for row in live_rows]
          ),
          "judge": _distribution(
              [latency(row, "judge_api_ms") for row in live_rows]
          ),
      },
      "mean_tokens": {
          "retrieval_input": _mean_tokens(
              review_rows, lambda row: usage(row)["retrieval_tokens"]["input"]
          ),
          "retrieval_output": _mean_tokens(
              review_rows, lambda row: usage(row)["retrieval_tokens"]["output"]
          ),
          "generation_input": _mean_tokens(
              review_rows, lambda row: usage(row)["generation_tokens"]["input"]
          ),
          "generation_output": _mean_tokens(
              review_rows, lambda row: usage(row)["generation_tokens"]["output"]
          ),
          "judge_input": _mean_tokens(
              review_rows,
              lambda row: row["execution_stats"]["judge_usage"]["input_tokens"],
          ),
          "judge_output": _mean_tokens(
              review_rows,
              lambda row: row["execution_stats"]["judge_usage"]["output_tokens"],
          ),
      },
      "financials": {
          "actual_run_spend_usd": round(
              retrieval_total + generation_billed + judge_total, 6
          ),
          "cold_run_estimate_usd": round(
              retrieval_total + generation_total + judge_total, 6
          ),
          "judge_overhead_usd": round(judge_total, 6),
          "pricing": "list-price estimate, not an invoice",
          "pricing_models": {
              "retrieval": EVAL_RETRIEVAL_MODEL,
              "generator": GENERATOR_MODEL,
              "judge": JUDGE_MODEL,
          },
      },
  }


def _locked_sections(qa_items, manifest):
  """Map each QA id to that section's subsection_key and display_title."""
  locked = {}
  for item in qa_items:
    row = _match_subsection(item["section"], manifest)
    if row is None:
      raise RuntimeError(
          f"[{item['id']}] Section {item['section']!r} does not map to "
          "exactly one manifest title."
      )
    subsection_key = str(row.get("subsection_key") or "").strip()
    display_title = str(row.get("display_title") or "").strip()
    if not subsection_key or not display_title:
      raise RuntimeError(
          f"[{item['id']}] Manifest row for {item['section']!r} is missing "
          "subsection_key or display_title."
      )
    locked[item["id"]] = {
        "subsection_key": subsection_key,
        "display_title": display_title,
    }
  return locked


def _write_run(run):
  with open(_RESULTS_PATH, "w", encoding="utf-8") as results_file:
    json.dump(run, results_file, indent=2)


def _summarize_existing():
  if not _RESULTS_PATH.exists():
    raise RuntimeError(f"No results file at {_RESULTS_PATH}.")
  with open(_RESULTS_PATH, encoding="utf-8") as results_file:
    run = json.load(results_file)
  run["mode"] = run.get("mode") or "report_wide"
  run["summary"] = _summary(run["results"])
  _write_run(run)
  summary = run["summary"]
  pipeline = summary["latency"]["pipeline"]
  print(
      f"Rewrote summary in {_RESULTS_PATH.name}: "
      f"mean {pipeline['mean_ms']} ms, p95 {pipeline['p95_ms']} ms, "
      f"median {pipeline['median_ms']} ms, "
      f"spend ${summary['financials']['actual_run_spend_usd']}, "
      f"judge agreement "
      f"{summary['manual_review']['judge_agreement']}/"
      f"{summary['manual_review']['reviewed']} "
      f"({summary['manual_review']['unreviewed']} unreviewed)."
  )


def main():
  if _SUMMARIZE_ONLY:
    _summarize_existing()
    return

  for key in ("GEMINI_API_KEY", "NVIDIA_API_KEY"):
    if not os.getenv(key):
      raise RuntimeError(f"{key} is missing. Add it to .env.")

  sections, qa_items = _load_qa_items()
  store_entry = _resolve_store(sections)
  store_name = store_entry["store_name"]
  source_filename = _resolve_source_filename(store_entry)
  locked_sections = _locked_sections(
      qa_items, store_entry.get("manifest") or []
  )

  generator_llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, temperature=0.0)
  judge_model = _build_judge()
  metrics = [
      FaithfulnessMetric(model=judge_model, threshold=METRIC_THRESHOLD),
      AnswerRelevancyMetric(model=judge_model, threshold=METRIC_THRESHOLD),
      ContextualRecallMetric(model=judge_model, threshold=METRIC_THRESHOLD),
  ]

  answer_cache = _load_answer_cache()
  records = [
      _build_record(
          item, store_name, source_filename, generator_llm, answer_cache,
          subsection_key=locked_sections[item["id"]]["subsection_key"],
          display_title=locked_sections[item["id"]]["display_title"],
      )
      for item in qa_items
  ]

  # Pacing comes from _JUDGE_RATE_LIMITER (30 RPM, under NVIDIA's 40 RPM).
  for record in records:
    _judge_record(record, judge_model, metrics)

  review_rows = [_review_row(record) for record in records]
  run = {
      "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
      "mode": "section_locked" if _SECTION_LOCKED else "report_wide",
      "models": {
          "retrieval": EVAL_RETRIEVAL_MODEL,
          "generator": GENERATOR_MODEL,
          "judge": JUDGE_MODEL,
      },
      "summary": _summary(review_rows),
      "results": review_rows,
  }
  _write_run(run)


if __name__ == "__main__":
  main()
