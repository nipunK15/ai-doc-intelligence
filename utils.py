"""
utils.py
--------
Shared utilities:
  - File reading  (.pdf, .txt)
  - Text cleaning and chunking
  - ROUGE evaluation
  - Analytics / stats formatting
  - Export helpers (summary download, chat export)
"""

from __future__ import annotations

import io
import json
import logging
import re
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_CHARS      = 50_000
MIN_CHARS      = 100
CHUNK_CHARS    = 3_000
CHUNK_OVERLAP  = 0          # summarisation chunks don't need overlap
SUPPORTED_EXTS = {".pdf", ".txt"}


# ── Exceptions ─────────────────────────────────────────────────────────────────
class InputError(ValueError):
    """User-facing input validation error."""


# ── File reading ───────────────────────────────────────────────────────────────
def extract_text_from_file(filepath: str) -> str:
    path = Path(filepath)
    ext  = path.suffix.lower()

    if ext not in SUPPORTED_EXTS:
        raise InputError(f"Unsupported file type '{ext}'. Please upload .pdf or .txt.")

    try:
        if ext == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")
        return _read_pdf(path)
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"Could not read file '{path.name}': {exc}") from exc


def _read_pdf(path: Path) -> str:
    try:
        import pypdf  # noqa: PLC0415
    except ImportError as exc:
        raise InputError("pypdf not installed. Run: pip install pypdf") from exc

    reader = pypdf.PdfReader(str(path))
    if not reader.pages:
        raise InputError("The PDF appears to be empty.")

    pages = [p.extract_text() or "" for p in reader.pages]
    text  = "\n".join(pages).strip()
    if not text:
        raise InputError(
            "No text could be extracted. The PDF may be a scanned image. "
            "Please use a text-based PDF."
        )
    return text


def read_multiple_files(filepaths: list[str]) -> list[tuple[str, str]]:
    """
    Read several uploaded files.
    Returns list of (filename, text) tuples, skipping unreadable files with a warning.
    """
    results = []
    for fp in filepaths:
        name = Path(fp).name
        try:
            text = extract_text_from_file(fp)
            results.append((name, text))
            logger.info("Read %d chars from '%s'.", len(text), name)
        except InputError as e:
            logger.warning("Skipping '%s': %s", name, e)
    return results


# ── Validation ─────────────────────────────────────────────────────────────────
def validate_text(text: str) -> str:
    text = (text or "").strip()
    if len(text) < MIN_CHARS:
        raise InputError(
            f"Input too short ({len(text)} chars). Need at least {MIN_CHARS} characters."
        )
    if len(text) > MAX_CHARS:
        logger.warning("Truncating input from %d → %d chars.", len(text), MAX_CHARS)
        text = text[:MAX_CHARS]
    return text


# ── Chunking (for summarisation) ───────────────────────────────────────────────
def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Sentence-boundary-aware chunking for summarisation."""
    text      = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        slen = len(sent) + 1
        if current_len + slen > max_chars and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += slen

    if current:
        chunks.append(" ".join(current))

    logger.info("Chunked into %d piece(s) for summarisation.", len(chunks))
    return chunks


# ── ROUGE evaluation ───────────────────────────────────────────────────────────
def compute_rouge(reference: str, hypothesis: str) -> dict:
    try:
        from rouge_score import rouge_scorer  # noqa: PLC0415
    except ImportError:
        return {}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "ROUGE-1": round(scores["rouge1"].fmeasure, 4),
        "ROUGE-2": round(scores["rouge2"].fmeasure, 4),
        "ROUGE-L": round(scores["rougeL"].fmeasure, 4),
    }


# ── Stats ──────────────────────────────────────────────────────────────────────
def compression_ratio(original: str, summary: str) -> float:
    orig = max(len(original.split()), 1)
    summ = len(summary.split())
    return round(summ / orig, 3)


def build_stats_md(
    original: str,
    summary: str,
    elapsed: float,
    rouge: dict,
    from_cache: bool = False,
    model_label: str = "",
) -> str:
    orig_w = len(original.split())
    summ_w = len(summary.split())
    ratio  = compression_ratio(original, summary)

    cache_badge = " *(cached)*" if from_cache else ""
    model_row   = f"| 🤖 Model | `{model_label}` |\n" if model_label else ""

    rows = (
        f"### 📊 Analytics\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"{model_row}"
        f"| ⏱ Inference time | `{elapsed}s`{cache_badge} |\n"
        f"| 📝 Input words | `{orig_w:,}` |\n"
        f"| ✂️ Summary words | `{summ_w:,}` |\n"
        f"| 📉 Compression | `{ratio:.1%}` |\n"
    )

    if rouge:
        rows += (
            f"| 🔴 ROUGE-1 F1 | `{rouge['ROUGE-1']}` |\n"
            f"| 🟠 ROUGE-2 F1 | `{rouge['ROUGE-2']}` |\n"
            f"| 🟡 ROUGE-L F1 | `{rouge['ROUGE-L']}` |\n"
        )

    return rows


# ── Export helpers ─────────────────────────────────────────────────────────────
def export_summary_as_txt(summary: str, stats_md: str) -> str:
    """Write summary + stats to a temp .txt file; return its path."""
    content = f"=== AI SUMMARY ===\n\n{summary}\n\n=== STATS ===\n\n{stats_md}"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        return f.name


def export_chat_as_txt(history: list[dict]) -> str:
    """Write chat history to a temp .txt file; return its path."""
    lines = []
    for msg in history:
        role    = "You" if msg["role"] == "user" else "AI"
        content = msg["content"]
        lines.append(f"[{role}]\n{content}\n")
    content = "\n".join(lines)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        return f.name
