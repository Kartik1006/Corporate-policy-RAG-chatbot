# Corporate Policy RAG Chatbot

A production-ready Retrieval-Augmented Generation chatbot that ingests corporate policy documents, stores them in an embedded ChromaDB vector store, and answers questions using **Groq (Llama 3.3 70B)** or **Google Gemini (2.5 Flash)** via LangChain.

API keys are loaded securely from a `.env` file — no manual entry in the UI required.

---

## Features

| Feature | Details |
|---|---|
| **LLM Providers** | Groq (Llama 3.3 70B Versatile) or Google Gemini 2.5 Flash — switchable from the UI |
| **Secure API Keys** | Loaded from `.env` file automatically — never exposed in the interface |
| **Temperature Control** | Adjustable temperature slider (0.0–1.0) for fine-tuning response creativity |
| **Token Usage Tracking** | Daily token usage monitoring with per-model limits and visual progress bars |
| **Document Upload** | Upload your own PDF/TXT policy documents directly through the UI |
| **PDF Text Extraction** | PyMuPDF (fitz) for high-quality text extraction from PDF files |
| **Embedded Models** | HuggingFace `all-MiniLM-L6-v2` — no paid embedding API required |
| **Vector Store** | ChromaDB with persistent storage in `chroma_db/` |
| **Metadata Tagging** | Automatic company & department extraction from filenames |
| **Smart Chunking** | `RecursiveCharacterTextSplitter` — 750 tokens / 100 overlap |
| **Rate Limiting** | Tenacity exponential backoff + inter-call delays for free-tier APIs |
| **Company Filter** | Dropdown to narrow retrieval to a single company |
| **Chat History** | Full conversational context passed to the LLM |
| **Professional UI** | Clean, theme-adaptive design with dark/light mode support |

---

## Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- A free API key from one of:
  - **Groq**: https://console.groq.com
  - **Google AI Studio**: https://aistudio.google.com/apikey

---

## Step-by-Step Setup

### 1. Clone / navigate to the project

```bash
cd c:\2026\cbot
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (cmd)
.\.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will also download the `all-MiniLM-L6-v2` model (~80 MB).

### 4. Configure API keys

Create a `.env` file in the project root with your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

You can provide one or both keys. The app auto-detects which providers are available.

### 5. Add your documents

Place `.txt` or `.pdf` files in the `docs/` folder. The app extracts metadata from filenames:

| Filename | Extracted Metadata |
|---|---|
| `Google.txt` | company: Google, department: General |
| `Tesla_HR_Policy.pdf` | company: Tesla, department: HR |
| `SpaceX.txt` | company: SpaceX, department: General |
| `random-paper.pdf` | company: Unknown, department: General |

You can also **upload documents directly** through the sidebar in the app.

### 6. Run the app

```bash
streamlit run app.py
```

### 7. Start using

1. Select your **LLM Provider** (Groq or Gemini) in the sidebar.
2. Adjust the **Temperature** slider for response style.
3. Optionally filter by a **specific company**.
4. Upload new policy documents via the **Upload Documents** section.
5. Start chatting!

---

## Project Structure

```
cbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── .env                # API keys (GROQ_API_KEY, GOOGLE_API_KEY)
├── .token_usage.json   # Auto-generated daily token tracking
├── docs/               # Your policy documents (txt, pdf)
│   ├── Google.txt
│   ├── Tesla.txt
│   └── ...
└── chroma_db/          # Auto-generated vector store (persisted)
```

---

## Token Usage & Limits

The app tracks token usage per model on a **daily** basis. Limits reset at midnight:

| Model | Daily Token Limit |
|---|---|
| Groq (Llama 3.3 70B) | 100,000 tokens |
| Google Gemini (2.5 Flash) | 150,000 tokens |

Token usage is displayed as progress bars in the sidebar. When a limit is reached, the app prompts you to switch providers or wait until the next day.

---

## Document Upload

You can train the chatbot on your own policy documents:

1. Click the **Upload Documents** section in the sidebar.
2. Drag & drop or browse for `.pdf` or `.txt` files.
3. Click **Process & Add to KB** to add them to the knowledge base.
4. The documents are saved to `docs/`, chunked, embedded, and added to ChromaDB automatically.

**PDF extraction** uses PyMuPDF (fitz) for high-quality text extraction that handles complex layouts.

---

## Rate Limiting Details

Free-tier LLM APIs (Groq, Gemini) impose strict RPM limits (typically 15–30 req/min). This app handles this at two levels:

1. **Ingestion phase**: Documents are batched (500 chunks/batch) using embedded models — no API rate limits.
2. **Query phase**: Each LLM call includes a 2.5-second delay, plus tenacity's exponential backoff (retries up to 5× on 429 errors).

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `No API keys found in .env` | Create a `.env` file with `GROQ_API_KEY` and/or `GOOGLE_API_KEY`. |
| `429 Too Many Requests` | Wait 30–60 seconds and retry. The backoff logic handles most cases automatically. |
| Daily token limit reached | Switch to another provider, or wait until tomorrow for the limit to reset. |
| Embedding model download fails | Check your internet connection. The model is cached after the first download. |
| ChromaDB stale after adding new docs | Click **Rebuild KB** in the sidebar, or delete `chroma_db/` and restart. |
| `ModuleNotFoundError` | Ensure you activated your virtual environment and ran `pip install -r requirements.txt`. |
| Uploaded PDF shows no text | The PDF may be image-based (scanned). PyMuPDF extracts text-layer only. |
