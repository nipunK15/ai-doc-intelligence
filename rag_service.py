"""
rag_service.py
--------------
RAG (Retrieval-Augmented Generation) pipeline.

Responsibilities:
  - Embed document chunks using sentence-transformers
  - Store / retrieve via FAISS in-memory index
  - Answer questions by retrieving top-k context + feeding to QA model
  - Support multiple documents in one session
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_MODEL_ID   = "sentence-transformers/all-MiniLM-L6-v2"   # 80 MB, fast on CPU
QA_MODEL_ID      = "deepset/roberta-base-squad2"               # lightweight extractive QA
TOP_K            = 4          # number of chunks to retrieve per query
CHUNK_SIZE_CHARS = 600        # smaller than summarization chunks for precise retrieval
CHUNK_OVERLAP    = 80         # character overlap between chunks

# ── Lazy singletons ────────────────────────────────────────────────────────────
_embedder = None
_qa_pipe  = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        logger.info("Loading embedding model: %s", EMBED_MODEL_ID)
        _embedder = SentenceTransformer(EMBED_MODEL_ID)
        logger.info("Embedding model loaded.")
    return _embedder


def _get_qa_pipe():
    global _qa_pipe
    if _qa_pipe is None:
        from transformers import pipeline  # noqa: PLC0415
        logger.info("Loading QA model: %s", QA_MODEL_ID)
        _qa_pipe = pipeline(
            "question-answering",
            model=QA_MODEL_ID,
            tokenizer=QA_MODEL_ID,
            device=-1,
        )
        logger.info("QA model loaded.")
    return _qa_pipe


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class DocumentStore:
    """
    In-memory vector store for one session.

    Holds:
      - raw text chunks
      - L2-normalised FAISS index over their embeddings
      - source file labels for provenance
    """
    chunks:      list[str]       = field(default_factory=list)
    sources:     list[str]       = field(default_factory=list)   # filename per chunk
    index:       object          = None   # faiss.IndexFlatIP
    is_built:    bool            = False

    def reset(self):
        self.chunks  = []
        self.sources = []
        self.index   = None
        self.is_built = False

    @property
    def doc_count(self) -> int:
        return len(set(self.sources))

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


# ── Module-level session store (one per Gradio session via State) ──────────────
# We keep a single global store; Gradio's gr.State will carry the object per tab.


# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk_for_rag(text: str, source: str = "document") -> tuple[list[str], list[str]]:
    """
    Split text into overlapping character-level chunks.
    Returns (chunks, sources) parallel lists.
    """
    import re
    text   = re.sub(r"\s+", " ", text).strip()
    chunks, sources = [], []

    start = 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE_CHARS, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:          # skip tiny trailing fragments
            chunks.append(chunk)
            sources.append(source)
        start += CHUNK_SIZE_CHARS - CHUNK_OVERLAP

    return chunks, sources


# ── Indexing ───────────────────────────────────────────────────────────────────
def add_documents(
    store: DocumentStore,
    texts: list[str],
    filenames: list[str],
) -> str:
    """
    Chunk, embed, and add documents to the store.
    Existing documents are retained (append behaviour).

    Returns a status message.
    """
    import faiss  # noqa: PLC0415

    embedder = _get_embedder()
    new_chunks, new_sources = [], []

    for text, fname in zip(texts, filenames):
        c, s = chunk_for_rag(text, source=fname)
        new_chunks.extend(c)
        new_sources.extend(s)

    if not new_chunks:
        return "⚠️ No usable text found in the uploaded documents."

    logger.info("Embedding %d chunks from %d file(s)…", len(new_chunks), len(filenames))
    embeddings = embedder.encode(new_chunks, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    # ── Build / extend FAISS index ─────────────────────────────────────────────
    dim = embeddings.shape[1]

    if store.index is None:
        store.index = faiss.IndexFlatIP(dim)   # inner product on normalised vecs = cosine sim

    store.index.add(embeddings)
    store.chunks.extend(new_chunks)
    store.sources.extend(new_sources)
    store.is_built = True

    return (
        f"✅ Indexed **{len(new_chunks)} chunks** from **{len(filenames)} file(s)**. "
        f"Total in store: {store.chunk_count} chunks from {store.doc_count} document(s)."
    )


# ── Retrieval ──────────────────────────────────────────────────────────────────
def retrieve(store: DocumentStore, query: str, top_k: int = TOP_K) -> list[tuple[str, str, float]]:
    """
    Retrieve top-k chunks most relevant to query.
    Returns list of (chunk_text, source_filename, similarity_score).
    """
    if not store.is_built:
        return []

    embedder = _get_embedder()
    q_emb    = embedder.encode([query], normalize_embeddings=True)
    q_emb    = np.array(q_emb, dtype="float32")

    k        = min(top_k, store.chunk_count)
    scores, indices = store.index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append((store.chunks[idx], store.sources[idx], float(score)))

    return results


# ── QA ─────────────────────────────────────────────────────────────────────────
def answer_question(
    store: DocumentStore,
    question: str,
    chat_history: list[dict],
) -> tuple[str, list[dict], float]:
    """
    Answer a question using retrieved context.
    Maintains chat_history as list of {"role": "user"|"assistant", "content": str}.

    Returns:
        (answer_text, updated_history, elapsed_seconds)
    """
    if not question.strip():
        return "⚠️ Please type a question.", chat_history, 0.0

    if not store.is_built:
        answer = (
            "⚠️ No documents indexed yet. "
            "Please upload documents in the **Ask Questions** tab first."
        )
        chat_history = chat_history + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
        return answer, chat_history, 0.0

    start    = time.perf_counter()
    qa_pipe  = _get_qa_pipe()
    results  = retrieve(store, question)

    if not results:
        answer = "I couldn't find relevant information in the indexed documents."
    else:
        # Concatenate top-k chunks as context
        context = "\n\n".join(chunk for chunk, _, _ in results)

        # Prepend last 2 assistant turns for conversational continuity
        prior_context = ""
        assistant_turns = [m["content"] for m in chat_history if m["role"] == "assistant"]
        if assistant_turns:
            prior_context = "Previous answers:\n" + "\n".join(assistant_turns[-2:]) + "\n\n"

        full_context = prior_context + context

        try:
            qa_result = qa_pipe(
                question=question,
                context=full_context[:4000],   # RoBERTa max ~512 tokens; stay safe
                max_answer_len=150,
                handle_impossible_answer=True,
            )
            raw_answer = qa_result["answer"]
            score      = qa_result["score"]

            if not raw_answer or score < 0.05:
                answer = (
                    "I found relevant passages but couldn't extract a confident answer. "
                    "Try rephrasing your question or asking about something more specific.\n\n"
                    f"**Relevant excerpt:**\n> {results[0][0][:300]}…"
                )
            else:
                source_list = ", ".join(set(src for _, src, _ in results))
                answer      = (
                    f"{raw_answer}\n\n"
                    f"<sub>📄 Sources: {source_list} · Confidence: {score:.0%}</sub>"
                )
        except Exception as exc:
            logger.exception("QA model error.")
            answer = f"❌ Model error: {exc}"

    elapsed = round(time.perf_counter() - start, 2)

    updated_history = chat_history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]
    return answer, updated_history, elapsed


# ── Warm-up ────────────────────────────────────────────────────────────────────
def warmup_rag():
    """Pre-load both RAG models at startup to reduce first-request latency."""
    try:
        _get_embedder()
        _get_qa_pipe()
        logger.info("RAG models warm.")
    except Exception:
        logger.warning("RAG warmup failed (non-fatal).", exc_info=True)
