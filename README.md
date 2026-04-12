# 🧠 Smart Document Summarizer

> **Production-grade text summarization app** powered by `facebook/bart-large-cnn`, built with
> 🤗 Transformers and Gradio. Deployable to Hugging Face Spaces in minutes.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📄 **Multi-source input** | Paste text directly OR upload `.pdf` / `.txt` files |
| 📚 **Long-document support** | Automatic sentence-boundary chunking + two-pass summarization |
| 🎚 **Length control** | Short / Medium / Long presets |
| 📊 **ROUGE evaluation** | Optional ROUGE-1/2/L scoring against a reference summary |
| ⏱ **Latency tracking** | Inference time displayed after every run |
| 📉 **Compression stats** | Word count + compression ratio shown per summary |
| 🛡 **Robust error handling** | Empty input, oversized input, invalid file, model errors |
| 🚀 **Hugging Face Spaces ready** | Single `gradio` command, all deps in `requirements.txt` |

---

## 🗂 Project Structure

```
smart-summarizer/
├── app.py              # Gradio UI + event wiring
├── model_service.py    # Model singleton + inference
├── utils.py            # Chunking, PDF parsing, ROUGE, stats
├── requirements.txt    # Pinned dependencies
└── README.md
```

---

## 🏃 Run Locally

```bash
# 1. Clone / copy files into a directory
cd smart-summarizer

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
python app.py
# → Open http://localhost:7860
```

> **Note:** First launch downloads the BART model (~1.6 GB). Subsequent starts are instant.

---

## 🚀 Deploy to Hugging Face Spaces

```bash
# 1. Create a new Space at https://huggingface.co/new-space
#    → SDK: Gradio  |  Hardware: CPU Basic (free)

# 2. Clone the Space repo
git clone https://huggingface.co/spaces/<YOUR_USERNAME>/<SPACE_NAME>
cd <SPACE_NAME>

# 3. Copy project files into it
cp /path/to/smart-summarizer/* .

# 4. Commit and push
git add .
git commit -m "Initial production deployment"
git push

# → Your app goes live at https://huggingface.co/spaces/<YOUR_USERNAME>/<SPACE_NAME>
```

> The Space will automatically install `requirements.txt` and launch `app.py`.

---

## 🤖 Model Details

| Property | Value |
|---|---|
| Model | `facebook/bart-large-cnn` |
| Architecture | BART (Bidirectional + Auto-Regressive Transformer) |
| Trained on | CNN / DailyMail dataset |
| Max input | 1024 tokens per chunk |
| Task | Abstractive summarization |

---

## 📈 How Long-Document Chunking Works

```
Input text (any length)
       │
       ▼
 [Sentence splitter]
       │
       ▼
 Chunk 1 │ Chunk 2 │ Chunk 3  (≤3000 chars each)
       │
       ▼
 BART summarizes each chunk independently
       │
       ▼
 Merge partial summaries
       │
       ▼
 (if merged > max_len) → second-pass BART compression
       │
       ▼
 Final Summary ✅
```

---

## 📋 Resume-Ready Project Description

```
Smart Document Summarizer | Python · Hugging Face Transformers · Gradio
• Engineered a production-ready NLP web application that summarizes text and PDF documents
  using facebook/bart-large-cnn, achieving abstractive summaries with configurable length.
• Implemented sentence-boundary-aware chunking with a two-pass summarization pipeline to
  handle documents up to 50,000 characters without quality degradation.
• Integrated ROUGE-1/2/L evaluation, inference latency tracking, and compression ratio
  reporting for quantifiable output quality assessment.
• Designed a modular architecture (app.py / model_service.py / utils.py) with full error
  handling for empty input, oversized documents, and corrupt PDF files.
• Deployed on Hugging Face Spaces with a clean Gradio UI; model warm-starts at app launch
  to eliminate first-request latency.
```

---

## 🛠 Tech Stack

- **Model:** `facebook/bart-large-cnn` (Hugging Face Transformers)
- **UI:** Gradio 4.x (Blocks API, Soft theme)
- **PDF parsing:** pypdf
- **Evaluation:** rouge-score
- **Deployment:** Hugging Face Spaces (CPU Basic — free tier)
