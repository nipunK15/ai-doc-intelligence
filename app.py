"""
app.py
------
AI Document Intelligence System
─────────────────────────────────────────────────────────────────────────────
Tabs:
  1. 📄 Summarize   — paste text or upload files, choose model + length
  2. 💬 Ask Questions — RAG-powered chat with memory over uploaded docs
  3. 📊 Analytics    — run stats dashboard with ROUGE comparison

Powered by:
  - facebook/bart-large-cnn / sshleifer/distilbart-cnn-12-6  (summarisation)
  - sentence-transformers/all-MiniLM-L6-v2                   (embeddings)
  - deepset/roberta-base-squad2                              (extractive QA)
  - FAISS                                                    (vector store)
─────────────────────────────────────────────────────────────────────────────
"""

import logging

import gradio as gr

from model_service import (
    MODELS,
    DEFAULT_MODEL,
    warmup_summarizer,
    summarize_chunks,
)
from rag_service import (
    DocumentStore,
    add_documents,
    answer_question,
    warmup_rag,
)
from utils import (
    InputError,
    validate_text,
    extract_text_from_file,
    read_multiple_files,
    chunk_text,
    compute_rouge,
    build_stats_md,
    export_summary_as_txt,
    export_chat_as_txt,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Pre-warm models ────────────────────────────────────────────────────────────
warmup_summarizer()
warmup_rag()

# ── Length presets ─────────────────────────────────────────────────────────────
LENGTH_PRESETS = {
    "Short  (1–2 sentences)": (40,  80),
    "Medium (3–5 sentences)": (80,  200),
    "Long   (detailed)":      (150, 400),
}

EXAMPLE_TEXT = (
    "The Amazon rainforest covers over 5.5 million square kilometres and is home to an "
    "estimated 10% of all species on Earth. It plays a crucial role in regulating the global "
    "climate by absorbing vast quantities of carbon dioxide. However, deforestation driven by "
    "agricultural expansion and illegal logging has destroyed nearly 20% of the original forest "
    "in five decades. Scientists warn that if deforestation continues, the Amazon could reach a "
    "tipping point within 15 to 30 years after which large sections would convert to dry savannah, "
    "releasing stored carbon and accelerating global warming. International conservation efforts, "
    "indigenous land rights, and sustainable economic alternatives are considered critical to "
    "preventing this outcome."
)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SUMMARIZE
# ══════════════════════════════════════════════════════════════════════════════

def do_summarize(
    raw_text: str,
    uploaded_files,          # list[str] or None
    length_preset: str,
    model_label: str,
    reference_text: str,
):
    """Main summarisation callback."""
    # ── Resolve source ─────────────────────────────────────────────────────────
    try:
        if uploaded_files:
            # Take first file for summarisation (multi-file RAG is in Tab 2)
            source_text = extract_text_from_file(
                uploaded_files[0] if isinstance(uploaded_files, list) else uploaded_files
            )
        elif raw_text and raw_text.strip():
            source_text = raw_text
        else:
            return "⚠️ Please paste text or upload a file.", "", "", gr.update(visible=False)
    except InputError as e:
        return f"❌ **Input Error:** {e}", "", "", gr.update(visible=False)
    except Exception as e:
        logger.exception("File reading error.")
        return f"❌ **Unexpected error:** {e}", "", "", gr.update(visible=False)

    # ── Validate ───────────────────────────────────────────────────────────────
    try:
        source_text = validate_text(source_text)
    except InputError as e:
        return f"⚠️ {e}", "", "", gr.update(visible=False)

    # ── Chunk + summarise ──────────────────────────────────────────────────────
    min_len, max_len = LENGTH_PRESETS.get(length_preset, (80, 200))
    chunks = chunk_text(source_text)

    try:
        summary, elapsed, from_cache = summarize_chunks(
            chunks, model_label=model_label, min_len=min_len, max_len=max_len
        )
    except Exception as e:
        logger.exception("Summarisation error.")
        return f"❌ **Model error:** {e}", "", "", gr.update(visible=False)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    rouge = {}
    if reference_text and reference_text.strip():
        try:
            rouge = compute_rouge(reference_text.strip(), summary)
        except Exception:
            logger.warning("ROUGE failed.", exc_info=True)

    stats_md = build_stats_md(source_text, summary, elapsed, rouge, from_cache, model_label)
    word_note = f"*{len(summary.split())} words*"
    return summary, word_note, stats_md, gr.update(visible=True)


def do_export_summary(summary: str, stats_md: str):
    if not summary.strip():
        return gr.update(visible=False), "⚠️ No summary to export yet."
    path = export_summary_as_txt(summary, stats_md)
    return gr.update(value=path, visible=True), ""


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ASK QUESTIONS (RAG chat)
# ══════════════════════════════════════════════════════════════════════════════

def do_index_documents(
    uploaded_files,
    store: DocumentStore,
):
    """Index uploaded files into the RAG store."""
    if not uploaded_files:
        return store, "⚠️ No files uploaded.", []

    file_list = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    pairs     = read_multiple_files(file_list)

    if not pairs:
        return store, "❌ Could not read any files.", []

    texts     = [t for _, t in pairs]
    filenames = [n for n, _ in pairs]

    status = add_documents(store, texts, filenames)
    return store, status, []    # reset chat on new index


def do_chat(
    question: str,
    history: list[dict],
    store: DocumentStore,
):
    """Single-turn RAG QA, maintaining history."""
    if not question.strip():
        return history, store, ""

    _, updated_history, elapsed = answer_question(store, question, history)
    return updated_history, store, ""


def do_clear_chat(store: DocumentStore):
    return [], store, ""


def do_export_chat(history: list[dict]):
    if not history:
        return gr.update(visible=False), "⚠️ No conversation to export yet."
    path = export_chat_as_txt(history)
    return gr.update(value=path, visible=True), ""


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def do_analytics(
    summary: str,
    source_text_state: str,
    elapsed_state: float,
    rouge: dict,
    model_label: str,
):
    """Recompute analytics from stored state."""
    if not summary or not source_text_state:
        return "⚠️ Run a summarisation first (Tab 1) to see analytics here."

    stats = build_stats_md(
        source_text_state, summary, elapsed_state, rouge, False, model_label
    )
    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
/* ── Global ───────────────────────────────────────── */
body, .gradio-container { font-family: 'Inter', sans-serif !important; }
footer { display: none !important; }

/* ── Header ───────────────────────────────────────── */
#app-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 16px;
    padding: 28px 32px 20px;
    margin-bottom: 8px;
    color: white !important;
}
#app-header h1 { color: white !important; margin: 0 0 6px; font-size: 1.9rem; }
#app-header p  { color: rgba(255,255,255,0.85) !important; margin: 0; font-size: 0.95rem; }

/* ── Cards ─────────────────────────────────────────── */
.card {
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 20px;
}

/* ── Status chip ───────────────────────────────────── */
.status-ok  { color: #16a34a; font-weight: 600; }
.status-err { color: #dc2626; font-weight: 600; }

/* ── Chat bubbles ───────────────────────────────────── */
.message.user     { background: #ede9fe !important; }
.message.assistant{ background: #f0fdf4 !important; }

/* ── Primary button ─────────────────────────────────── */
#primary-btn {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
"""

with gr.Blocks(
    title="AI Document Intelligence System",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    css=CSS,
) as demo:

    # ── Session state ──────────────────────────────────────────────────────────
    rag_store          = gr.State(DocumentStore())          # RAG vector store
    source_text_state  = gr.State("")                     # raw input for analytics
    elapsed_state      = gr.State(0.0)                    # last inference time
    rouge_state        = gr.State({})                     # last ROUGE scores

    # ── Header ────────────────────────────────────────────────────────────────
    with gr.Column(elem_id="app-header"):
        gr.HTML(
            "<h1>🧠 AI Document Intelligence System</h1>"
            "<p>Summarise documents · Ask questions with RAG · Analyse performance</p>"
        )

    # ══════════════════════════════════════════════════════════════════════════
    with gr.Tabs():

        # ────────────────────────────────────────────────────────────────────
        # TAB 1 — SUMMARIZE
        # ────────────────────────────────────────────────────────────────────
        with gr.TabItem("📄 Summarize"):
            with gr.Row(equal_height=False):

                # Left: inputs
                with gr.Column(scale=1):
                    gr.Markdown("#### Input")
                    sum_text_input = gr.Textbox(
                        label="Paste text",
                        placeholder="Paste any article, report, legal doc, meeting notes…",
                        lines=10,
                        max_lines=18,
                    )
                    sum_file_input = gr.File(
                        label="Or upload file(s) — .pdf or .txt",
                        file_types=[".pdf", ".txt"],
                        file_count="single",
                        type="filepath",
                    )

                    with gr.Row():
                        sum_length = gr.Radio(
                            choices=list(LENGTH_PRESETS.keys()),
                            value="Medium (3–5 sentences)",
                            label="Summary length",
                        )
                        sum_model = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value=DEFAULT_MODEL,
                            label="Model",
                        )

                    with gr.Accordion("🔬 ROUGE Evaluation (optional)", open=False):
                        sum_reference = gr.Textbox(
                            label="Reference summary",
                            placeholder="Paste a human-written reference to compute ROUGE scores…",
                            lines=4,
                        )

                    with gr.Row():
                        sum_btn   = gr.Button("✨ Summarize", variant="primary", elem_id="primary-btn")
                        sum_clear = gr.Button("🗑 Clear", variant="secondary")

                    

                # Right: outputs
                with gr.Column(scale=1):
                    gr.Markdown("#### Summary")
                    sum_output = gr.Textbox(
                        label="Generated summary",
                        lines=10,
                        interactive=False,
                    )
                    sum_word_count = gr.Markdown("")
                    sum_stats      = gr.Markdown("")

                    with gr.Row():
                        sum_export_btn  = gr.Button("⬇️ Export Summary", visible=False)
                        sum_export_file = gr.File(label="Download", visible=False)

            # Wiring
            sum_btn.click(
                fn=do_summarize,
                inputs=[sum_text_input, sum_file_input, sum_length, sum_model, sum_reference],
                outputs=[sum_output, sum_word_count, sum_stats, sum_export_btn],
            )

            sum_export_btn.click(
                fn=do_export_summary,
                inputs=[sum_output, sum_stats],
                outputs=[sum_export_file, sum_word_count],
            )

            def _clear_sum():
                return "", None, "Medium (3–5 sentences)", DEFAULT_MODEL, "", "", "", "", gr.update(visible=False), gr.update(visible=False)

            sum_clear.click(
                fn=_clear_sum,
                outputs=[sum_text_input, sum_file_input, sum_length, sum_model, sum_reference,
                         sum_output, sum_word_count, sum_stats, sum_export_btn, sum_export_file],
            )

        # ────────────────────────────────────────────────────────────────────
        # TAB 2 — ASK QUESTIONS
        # ────────────────────────────────────────────────────────────────────
        with gr.TabItem("💬 Ask Questions"):

            gr.Markdown(
                "> Upload one or more documents, click **Index Documents**, "
                "then chat with your data. The AI retrieves relevant passages and "
                "answers based only on your documents."
            )

            with gr.Row(equal_height=False):

                # Left: upload + index
                with gr.Column(scale=1):
                    gr.Markdown("#### 1. Upload Documents")
                    rag_files = gr.File(
                        label="Upload .pdf or .txt files",
                        file_types=[".pdf", ".txt"],
                        file_count="multiple",
                        type="filepath",
                    )
                    rag_index_btn = gr.Button("📚 Index Documents", variant="primary")
                    rag_status    = gr.Markdown("")

                    gr.Markdown("---")
                    gr.Markdown("#### 2. Ask a Question")
                    rag_question = gr.Textbox(
                        label="Your question",
                        placeholder="What is the main argument? What are the key findings?…",
                        lines=3,
                    )
                    with gr.Row():
                        rag_ask_btn   = gr.Button("🔍 Ask", variant="primary")
                        rag_clear_btn = gr.Button("🗑 Clear Chat", variant="secondary")

                    rag_export_btn  = gr.Button("⬇️ Export Chat", visible=False)
                    rag_export_file = gr.File(label="Download", visible=False)

                # Right: chat
                with gr.Column(scale=1):
                    gr.Markdown("#### Conversation")
                    rag_chatbot = gr.Chatbot(
                        label="",
                        height=480,
                    )

            # Wiring
            rag_index_btn.click(
                fn=do_index_documents,
                inputs=[rag_files, rag_store],
                outputs=[rag_store, rag_status, rag_chatbot],
            )

            def _ask_and_show_export(question, history, store):
                updated_history, updated_store, cleared_q = do_chat(question, history, store)
                export_visible = gr.update(visible=bool(updated_history))
                return updated_history, updated_store, cleared_q, export_visible

            rag_ask_btn.click(
                fn=_ask_and_show_export,
                inputs=[rag_question, rag_chatbot, rag_store],
                outputs=[rag_chatbot, rag_store, rag_question, rag_export_btn],
            )
            rag_question.submit(
                fn=_ask_and_show_export,
                inputs=[rag_question, rag_chatbot, rag_store],
                outputs=[rag_chatbot, rag_store, rag_question, rag_export_btn],
            )

            rag_clear_btn.click(
                fn=do_clear_chat,
                inputs=[rag_store],
                outputs=[rag_chatbot, rag_store, rag_question],
            )

            rag_export_btn.click(
                fn=do_export_chat,
                inputs=[rag_chatbot],
                outputs=[rag_export_file, rag_status],
            )

        # ────────────────────────────────────────────────────────────────────
        # TAB 3 — ANALYTICS
        # ────────────────────────────────────────────────────────────────────
        with gr.TabItem("📊 Analytics"):
            gr.Markdown(
                "> This tab reflects the **most recent summarisation run** from Tab 1. "
                "Run a summary first, then come back here."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    analytics_model = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        value=DEFAULT_MODEL,
                        label="Model used",
                        interactive=False,
                    )
                    analytics_refresh = gr.Button("🔄 Refresh Analytics", variant="primary")

                with gr.Column(scale=2):
                    analytics_output = gr.Markdown("*Run a summarisation in Tab 1 first.*")

            analytics_refresh.click(
                fn=do_analytics,
                inputs=[sum_output, source_text_state, elapsed_state, rouge_state, analytics_model],
                outputs=[analytics_output],
            )

            # Keep analytics model in sync with summarise tab selection
            sum_model.change(fn=lambda m: gr.update(value=m), inputs=[sum_model], outputs=[analytics_model])

    # ── Global footer ──────────────────────────────────────────────────────────
    gr.HTML(
        "<center style='margin-top:16px; color: #9ca3af; font-size:0.8rem;'>"
        "AI Document Intelligence System · "
        "BART-large-CNN · DistilBART · MiniLM · RoBERTa · FAISS · "
        "<a href='https://github.com/sanikamal/genai-huggingface' "
        "   style='color:#6366f1;' target='_blank'>GitHub</a>"
        "</center>"
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share=True,    
        show_error=True
    )
