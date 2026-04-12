"""
model_service.py
----------------
Summarization pipeline with:
  - Two model tiers: Fast (distilbart) and Accurate (bart-large-cnn)
  - LRU cache for repeated inputs
  - Chunked two-pass summarization for long documents
"""

from __future__ import annotations

import hashlib
import logging
import time
from functools import lru_cache

logger = logging.getLogger(__name__)

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "⚡ Fast (DistilBART)":       "sshleifer/distilbart-cnn-12-6",
    "🎯 Accurate (BART-large)":  "facebook/bart-large-cnn",
}
DEFAULT_MODEL = "⚡ Fast (DistilBART)"

MIN_SUMMARY_LEN = 40
MAX_SUMMARY_LEN = 400
MAX_CHUNK_CHARS = 3000

# ── Singleton pool (one pipeline per model ID) ─────────────────────────────────
_pipeline_cache: dict = {}


def load_model(model_label: str = DEFAULT_MODEL):
    """Return (pipeline, model_label), loading lazily and caching."""
    from transformers import pipeline  # noqa: PLC0415

    model_id = MODELS.get(model_label, MODELS[DEFAULT_MODEL])

    if model_id not in _pipeline_cache:
        logger.info("Loading summarization model: %s", model_id)
        _pipeline_cache[model_id] = pipeline(
            "summarization",
            model=model_id,
            tokenizer=model_id,
            device=-1,
            framework="pt",
        )
        logger.info("Model loaded: %s", model_id)

    return _pipeline_cache[model_id], model_label


def warmup_summarizer():
    """Pre-load the default summarization model."""
    try:
        load_model(DEFAULT_MODEL)
        logger.info("Summarization model warm.")
    except Exception:
        logger.warning("Summarization warmup failed.", exc_info=True)


# ── Input cache (keyed on text hash + params) ──────────────────────────────────
_result_cache: dict[str, tuple[str, float]] = {}

def _cache_key(text: str, model_label: str, min_len: int, max_len: int) -> str:
    raw = f"{model_label}|{min_len}|{max_len}|{text}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Core inference ─────────────────────────────────────────────────────────────
def summarize_chunks(
    chunks: list[str],
    model_label: str = DEFAULT_MODEL,
    min_len: int     = MIN_SUMMARY_LEN,
    max_len: int     = MAX_SUMMARY_LEN,
) -> tuple[str, float, bool]:
    """
    Summarize a list of text chunks.

    Returns:
        (summary_text, elapsed_seconds, from_cache)
    """
    pipe, _ = load_model(model_label)

    # ── Single-chunk cache check ───────────────────────────────────────────────
    if len(chunks) == 1:
        key = _cache_key(chunks[0], model_label, min_len, max_len)
        if key in _result_cache:
            logger.info("Cache hit.")
            cached_text, cached_time = _result_cache[key]
            return cached_text, cached_time, True

    start           = time.perf_counter()
    partial_results = []

    for chunk in chunks:
        if len(chunk.split()) < 30:
            partial_results.append(chunk)
            continue
        result = pipe(
            chunk,
            min_length=min_len,
            max_length=max_len,
            do_sample=False,
            truncation=True,
        )
        partial_results.append(result[0]["summary_text"])

    merged = " ".join(partial_results)

    # Second-pass compression for multi-chunk inputs
    if len(chunks) > 1 and len(merged.split()) > max_len:
        second = pipe(
            merged[:MAX_CHUNK_CHARS],
            min_length=min_len,
            max_length=max_len,
            do_sample=False,
            truncation=True,
        )
        merged = second[0]["summary_text"]

    elapsed = round(time.perf_counter() - start, 2)
    merged  = merged.strip()

    # Store in cache for single-chunk inputs
    if len(chunks) == 1:
        _result_cache[key] = (merged, elapsed)

    return merged, elapsed, False
