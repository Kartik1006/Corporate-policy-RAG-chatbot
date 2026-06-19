"""
RAG-Based Corporate Policy Chatbot
===================================
A production-ready Retrieval-Augmented Generation chatbot that ingests
corporate policy documents, stores them in ChromaDB with metadata, and
answers queries using Groq (Llama 3.3) or Google Gemini via LangChain.

API keys are loaded from the .env file — no manual entry needed.
Users can upload their own policy documents (PDF / TXT) to train the model.

Usage:
    streamlit run app.py
"""

import os
import re
import time
import hashlib
import json
import streamlit as st
from pathlib import Path
from typing import Optional
from datetime import datetime

# --- Environment ---
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# --- LangChain core ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Embeddings & Vector Store ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Document Loaders ---
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# --- PDF text extraction ---
import fitz  # PyMuPDF

# --- Rate-limit / retry ---
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 100

BATCH_SIZE = 500
LLM_CALL_DELAY = 2.5

TOKEN_LIMITS = {
    "Groq (Llama 3.3 70B)": 100_000,
    "Google Gemini (2.5 Flash)": 150_000,
}

USAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".token_usage.json")

# Inline SVG icons (no emoji, no external deps)
SVG_POLICY = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm-1 7V3.5L18.5 9H13zM8 13h8v2H8v-2zm0 4h8v2H8v-2zm0-8h4v2H8V9z"/></svg>'
SVG_CHAT = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/></svg>'
SVG_FOLDER = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>'


# ─────────────────────────────────────────────
# TOKEN USAGE TRACKING
# ─────────────────────────────────────────────
def _load_token_usage() -> dict:
    """Load token usage data from disk."""
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE, "r") as f:
                data = json.load(f)
            today = datetime.now().strftime("%Y-%m-%d")
            if data.get("date") != today:
                return {"date": today, "groq": 0, "gemini": 0}
            return data
        except (json.JSONDecodeError, KeyError):
            pass
    return {"date": datetime.now().strftime("%Y-%m-%d"), "groq": 0, "gemini": 0}


def _save_token_usage(usage: dict):
    """Save token usage data to disk."""
    with open(USAGE_FILE, "w") as f:
        json.dump(usage, f)


def track_tokens(provider: str, tokens_used: int):
    """Track token usage for a provider."""
    usage = _load_token_usage()
    key = "groq" if "Groq" in provider else "gemini"
    usage[key] = usage.get(key, 0) + tokens_used
    _save_token_usage(usage)
    return usage


def get_token_usage() -> dict:
    """Get current token usage."""
    return _load_token_usage()


def check_token_limit(provider: str) -> tuple[bool, int, int]:
    """Check if the token limit has been reached. Returns (allowed, used, limit)."""
    usage = _load_token_usage()
    key = "groq" if "Groq" in provider else "gemini"
    used = usage.get(key, 0)
    limit = TOKEN_LIMITS.get(provider, 100_000)
    return used < limit, used, limit


# ─────────────────────────────────────────────
# HELPERS — Metadata extraction from filenames
# ─────────────────────────────────────────────
def extract_metadata_from_filename(filepath: str) -> dict:
    """
    Derive metadata tags from a document's filename.

    Naming convention supported:
      • "Google_HR_Policy.pdf"  → company=Google,  department=HR
      • "Tesla.txt"             → company=Tesla,    department=General
      • "attention-is-all-you-need.pdf" → company=Unknown, department=General
    """
    stem = Path(filepath).stem
    parts = re.split(r"[_\-\s]+", stem)

    company = "Unknown"
    department = "General"

    if parts:
        candidate = parts[0]
        if candidate[0].isupper() and candidate.isalpha():
            company = candidate
        if len(parts) >= 2 and parts[1].isalpha():
            department = parts[1]

    return {
        "company": company,
        "department": department,
        "source_file": Path(filepath).name,
    }


# ─────────────────────────────────────────────
# PDF TEXT EXTRACTION (PyMuPDF)
# ─────────────────────────────────────────────
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF (fitz)."""
    text_parts = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_parts.append(page.get_text())
        doc.close()
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")
    return "\n\n".join(text_parts)


# ─────────────────────────────────────────────
# DOCUMENT LOADING + CHUNKING
# ─────────────────────────────────────────────
def load_documents(docs_dir: str) -> list[Document]:
    """Load .txt and .pdf files from *docs_dir*, attaching metadata."""
    documents: list[Document] = []
    supported_ext = {".txt", ".pdf"}

    for fname in os.listdir(docs_dir):
        fpath = os.path.join(docs_dir, fname)
        ext = Path(fname).suffix.lower()
        if ext not in supported_ext:
            continue

        meta = extract_metadata_from_filename(fpath)

        try:
            if ext == ".txt":
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                doc = Document(page_content=content, metadata=meta)
                documents.append(doc)
            elif ext == ".pdf":
                text = extract_text_from_pdf(fpath)
                if text.strip():
                    doc = Document(page_content=text, metadata=meta)
                    documents.append(doc)
        except Exception as exc:
            st.warning(f"Skipped `{fname}`: {exc}")

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ─────────────────────────────────────────────
# EMBEDDING MODEL (cached across reruns)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model …")
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─────────────────────────────────────────────
# VECTOR STORE — Build or Load
# ─────────────────────────────────────────────
def _compute_docs_hash(docs_dir: str) -> str:
    """Quick hash of filenames + sizes to detect changes."""
    entries = sorted(
        (f, os.path.getsize(os.path.join(docs_dir, f)))
        for f in os.listdir(docs_dir)
        if os.path.isfile(os.path.join(docs_dir, f))
    )
    return hashlib.md5(str(entries).encode()).hexdigest()


@st.cache_resource(show_spinner="Building / loading vector store …")
def get_vector_store(_embeddings: HuggingFaceEmbeddings, docs_hash: str) -> Chroma:
    """
    Return a ChromaDB vector store. Re-ingests when *docs_hash* changes.
    """
    hash_file = os.path.join(CHROMA_DIR, ".docs_hash")
    if os.path.exists(CHROMA_DIR) and os.path.isfile(hash_file):
        with open(hash_file, "r") as fh:
            stored_hash = fh.read().strip()
        if stored_hash == docs_hash:
            return Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=_embeddings,
                collection_name="corporate_policies",
            )

    st.info("Ingesting documents — this may take a minute on first run.")
    raw_docs = load_documents(DOCS_DIR)
    if not raw_docs:
        st.error("No documents found in `docs/` folder.")
        st.stop()

    chunks = chunk_documents(raw_docs)
    st.info(f"Loaded **{len(raw_docs)}** raw pages → **{len(chunks)}** chunks.")

    vector_store = None
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    progress_bar = st.progress(0, text="Embedding chunks …")

    for batch_idx, i in enumerate(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i : i + BATCH_SIZE]
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=_embeddings,
                persist_directory=CHROMA_DIR,
                collection_name="corporate_policies",
            )
        else:
            vector_store.add_documents(batch)
        progress_bar.progress(
            (batch_idx + 1) / total_batches,
            text=f"Embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks",
        )

    progress_bar.empty()

    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(hash_file, "w") as fh:
        fh.write(docs_hash)

    return vector_store


# ─────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────
def get_llm(provider: str, temperature: float):
    """Return a LangChain chat-model wrapper for the chosen provider."""
    if provider == "Groq (Llama 3.3 70B)":
        from langchain_groq import ChatGroq

        api_key = os.environ.get("GROQ_API_KEY", "")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=temperature,
            max_tokens=1024,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=1024,
        )


# ─────────────────────────────────────────────
# RAG CHAIN
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert corporate policy assistant. Answer the user's question
based **only** on the retrieved context below. If the context does not
contain enough information to answer, say so clearly — do not fabricate.

When referencing policies, mention the source document and company.

### Retrieved Context
{context}
"""


def build_rag_chain(llm, retriever):
    """Assemble the RAG chain: retriever → prompt → LLM → parser."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs):
        formatted = []
        for i, d in enumerate(docs, 1):
            meta = d.metadata
            header = (
                f"[Doc {i} | {meta.get('source_file', '?')} | "
                f"Company: {meta.get('company', '?')} | "
                f"Dept: {meta.get('department', '?')}]"
            )
            formatted.append(f"{header}\n{d.page_content}")
        return "\n\n---\n\n".join(formatted)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def invoke_with_backoff(chain_input):
        return (prompt | llm | StrOutputParser()).invoke(chain_input)

    def rag_invoke(inputs: dict) -> str:
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        docs = retriever.invoke(question)
        context = format_docs(docs)

        time.sleep(LLM_CALL_DELAY)

        answer = invoke_with_backoff(
            {"context": context, "question": question, "chat_history": chat_history}
        )

        input_tokens = len(context + question) // 4
        output_tokens = len(answer) // 4
        total_tokens = input_tokens + output_tokens

        return answer, total_tokens

    return rag_invoke


# ─────────────────────────────────────────────
# UPLOAD HANDLER
# ─────────────────────────────────────────────
def handle_file_upload(uploaded_files, embeddings):
    """Process uploaded files: save to docs/, extract text, add to vector store."""
    if not uploaded_files:
        return

    os.makedirs(DOCS_DIR, exist_ok=True)
    new_documents = []
    saved_files = []

    progress = st.progress(0, text="Processing uploads …")

    for idx, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name
        file_path = os.path.join(DOCS_DIR, file_name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_name)

        meta = extract_metadata_from_filename(file_path)
        ext = Path(file_name).suffix.lower()

        try:
            if ext == ".pdf":
                text = extract_text_from_pdf(file_path)
                if text.strip():
                    doc = Document(page_content=text, metadata=meta)
                    new_documents.append(doc)
            elif ext == ".txt":
                content = uploaded_file.getvalue().decode("utf-8")
                doc = Document(page_content=content, metadata=meta)
                new_documents.append(doc)
        except Exception as e:
            st.error(f"Failed to process `{file_name}`: {e}")

        progress.progress(
            (idx + 1) / len(uploaded_files),
            text=f"Processed {idx + 1}/{len(uploaded_files)} files",
        )

    if new_documents:
        chunks = chunk_documents(new_documents)
        progress.progress(0.0, text="Adding to knowledge base …")

        vector_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="corporate_policies",
        )
        vector_store.add_documents(chunks)

        new_hash = _compute_docs_hash(DOCS_DIR)
        hash_file = os.path.join(CHROMA_DIR, ".docs_hash")
        os.makedirs(CHROMA_DIR, exist_ok=True)
        with open(hash_file, "w") as fh:
            fh.write(new_hash)

        progress.empty()
        st.success(
            f"Successfully added **{len(saved_files)}** file(s) — "
            f"**{len(chunks)}** new chunks indexed."
        )

        get_vector_store.clear()
        time.sleep(1)
        st.rerun()
    else:
        progress.empty()
        st.warning("No valid text could be extracted from the uploaded files.")


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Font — set on body, inherit naturally. Don't touch spans (icons). */
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Protect ALL Streamlit icon spans — they use Material Symbols Rounded */
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarExpandButton"] span,
[data-testid="stBaseButton-header"] span,
[data-testid="stBaseButton-headerNoPadding"] span,
[data-testid="baseButton-header"] span,
[data-testid="stChatInputSubmitButton"] span,
[data-testid="stFileUploaderDropzone"] button span,
[data-testid="stChatMessage"] [data-testid*="chatAvatarIcon"] span,
[data-testid="stStatusWidget"] span,
span[class*="material"],
span.e1nzilvr5,
button[kind="header"] span {
    font-family: 'Material Symbols Rounded' !important;
}



/* ── Light theme tokens ── */
:root, [data-testid="stAppViewContainer"] {
    --bg-page: #f7f8fa;
    --bg-card: #ffffff;
    --bg-card-hover: #f3f4f6;
    --bg-inset: #f1f3f5;
    --border: #e3e5e8;
    --border-light: #eef0f2;
    --text-1: #111827;
    --text-2: #4b5563;
    --text-3: #9ca3af;
    --accent: #4f46e5;
    --accent-soft: #eef2ff;
    --ok-bg: #f0fdf4;  --ok-fg: #15803d;  --ok-bdr: #bbf7d0;  --ok-dot: #22c55e;
    --err-bg: #fef2f2; --err-fg: #b91c1c; --err-bdr: #fecaca;
    --bar-track: #e5e7eb;
    --bar-yellow: #eab308;
    --bar-red: #ef4444;
}

/* ── Dark theme ── */
@media (prefers-color-scheme: dark) {
  :root, [data-testid="stAppViewContainer"] {
    --bg-page: #0e1117;
    --bg-card: #1a1c23;
    --bg-card-hover: #262833;
    --bg-inset: #14161d;
    --border: #2a2d38;
    --border-light: #22252e;
    --text-1: #e5e7eb;
    --text-2: #9ca3af;
    --text-3: #6b7280;
    --accent: #818cf8;
    --accent-soft: rgba(129,140,248,0.12);
    --ok-bg: rgba(34,197,94,0.1);  --ok-fg: #4ade80;  --ok-bdr: rgba(34,197,94,0.25);  --ok-dot: #4ade80;
    --err-bg: rgba(239,68,68,0.1); --err-fg: #f87171; --err-bdr: rgba(239,68,68,0.25);
    --bar-track: #2a2d38;
  }
}

.stApp { background-color: var(--bg-page); }

/* ── Header ── */
.app-header {
    background: var(--bg-card);
    padding: 1rem 1.35rem;
    border-radius: 10px;
    margin-bottom: 0.9rem;
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.app-header-icon {
    width: 36px; height: 36px;
    background: var(--accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.app-header-icon svg { width: 18px; height: 18px; fill: #fff; }
.app-header-text h1 {
    color: var(--text-1); margin: 0;
    font-size: 1rem; font-weight: 700; letter-spacing: -0.3px;
}
.app-header-text p {
    color: var(--text-3); margin: 0.05rem 0 0 0;
    font-size: 0.76rem; font-weight: 400;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {
    color: var(--text-2) !important;
}

/* ── Metrics ── */
.metric-row { display: flex; gap: 0.5rem; margin-bottom: 0.9rem; }
.metric-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.7rem 0.85rem;
    transition: border-color 0.12s ease;
}
.metric-card:hover { border-color: var(--text-3); }
.metric-value { font-size: 1.3rem; font-weight: 700; color: var(--text-1); line-height: 1.2; }
.metric-label {
    font-size: 0.62rem; color: var(--text-3);
    text-transform: uppercase; letter-spacing: 0.5px;
    font-weight: 600; margin-top: 0.12rem;
}
.status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--ok-dot); }
.status-dot.offline { background: var(--bar-red); }

/* ── Status badges ── */
.status-badge {
    display: inline-flex; align-items: center; gap: 0.25rem;
    padding: 0.18rem 0.5rem; border-radius: 5px;
    font-size: 0.68rem; font-weight: 500; margin-bottom: 0.25rem;
}
.badge-ready { background: var(--ok-bg); color: var(--ok-fg); border: 1px solid var(--ok-bdr); }
.badge-error { background: var(--err-bg); color: var(--err-fg); border: 1px solid var(--err-bdr); }

/* ── Chat ── */
[data-testid="stChatMessage"] {
    border-radius: 8px; padding: 0.8rem 0.95rem;
    margin-bottom: 0.35rem; border: 1px solid var(--border);
    background: var(--bg-card);
}

/* ── Token bars ── */
.token-bar-container {
    background: var(--bg-inset); border-radius: 6px;
    padding: 0.38rem 0.55rem; margin: 0.25rem 0;
    border: 1px solid var(--border);
}
.token-bar-bg { background: var(--bar-track); border-radius: 3px; height: 4px; overflow: hidden; margin-top: 0.2rem; }
.token-bar-fill { height: 100%; border-radius: 3px; transition: width 0.4s ease; }
.token-bar-fill.green { background: var(--ok-dot); }
.token-bar-fill.yellow { background: var(--bar-yellow); }
.token-bar-fill.red { background: var(--bar-red); }
.token-label { display: flex; justify-content: space-between; font-size: 0.64rem; color: var(--text-3); font-weight: 500; }

/* ── Doc list ── */
.doc-item {
    display: flex; align-items: center; gap: 0.4rem;
    padding: 0.35rem 0.5rem; background: var(--bg-inset);
    border-radius: 5px; margin-bottom: 0.2rem;
    border: 1px solid var(--border);
    font-size: 0.72rem; color: var(--text-2);
    transition: background 0.1s ease;
}
.doc-item:hover { background: var(--bg-card-hover); }
.doc-icon { font-size: 0.8rem; line-height: 1; }
.doc-name { flex: 1; font-weight: 500; }
.doc-size { font-size: 0.62rem; color: var(--text-3); }

/* ── Welcome ── */
.welcome-container { text-align: center; padding: 2rem 2rem 1rem 2rem; }
.welcome-icon {
    width: 44px; height: 44px; margin: 0 auto 0.6rem auto;
    background: var(--accent-soft); border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
}
.welcome-icon svg { width: 22px; height: 22px; fill: var(--accent); }
.welcome-title { font-size: 1rem; font-weight: 600; color: var(--text-1); margin-bottom: 0.25rem; }
.welcome-subtitle { font-size: 0.8rem; color: var(--text-3); margin-bottom: 1.1rem; line-height: 1.5; }

/* ── Sidebar sections ── */
.sidebar-section {
    font-size: 0.62rem; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--text-3);
    font-weight: 700; margin: 0.7rem 0 0.3rem 0;
    padding-bottom: 0.15rem; border-bottom: 1px solid var(--border-light);
}

/* ── Footer ── */
.footer-bar {
    text-align: center; padding: 0.6rem; margin-top: 1.2rem;
    font-size: 0.66rem; color: var(--text-3);
    border-top: 1px solid var(--border);
}
.footer-bar a { color: var(--accent); text-decoration: none; font-weight: 500; }
.footer-bar a:hover { text-decoration: underline; }

/* ── Token pill ── */
.token-pill {
    display: inline-flex; align-items: center; gap: 0.15rem;
    padding: 0.08rem 0.38rem; border-radius: 4px;
    font-size: 0.62rem; font-weight: 500;
    background: var(--bg-inset); color: var(--text-3);
    border: 1px solid var(--border); margin-top: 0.15rem;
}

/* ── Sidebar brand ── */
.sidebar-brand { text-align: center; padding: 0.25rem 0 0.45rem 0; }
.sidebar-brand-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 32px; height: 32px; background: var(--accent);
    border-radius: 8px; margin-bottom: 0.25rem;
}
.sidebar-brand-icon svg { width: 16px; height: 16px; fill: #fff; }
.sidebar-brand-name { font-size: 0.85rem; font-weight: 700; color: var(--text-1); letter-spacing: -0.2px; }
.sidebar-brand-sub { font-size: 0.56rem; color: var(--text-3); letter-spacing: 0.3px; text-transform: uppercase; font-weight: 600; }

/* ── Loading dots ── */
.loading-indicator { display: flex; align-items: center; gap: 0.45rem; padding: 0.5rem 0; color: var(--text-3); font-size: 0.8rem; font-weight: 500; }
.loading-dots { display: flex; gap: 3px; }
.loading-dots span {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--accent); opacity: 0.3;
    animation: dotPulse 1.4s ease-in-out infinite;
}
.loading-dots span:nth-child(2) { animation-delay: 0.2s; }
.loading-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes dotPulse {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1); }
}
</style>
"""


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="PolicyBot — Corporate Policy Assistant",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Header ──
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-header-icon">{SVG_POLICY}</div>
            <div class="app-header-text">
                <h1>Corporate Policy Assistant</h1>
                <p>Search and query your organization's policy documents using AI</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load API keys from .env ──
    groq_key = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GOOGLE_API_KEY", "")

    has_groq = bool(groq_key and len(groq_key) > 5)
    has_gemini = bool(gemini_key and len(gemini_key) > 5)

    providers = []
    if has_groq:
        providers.append("Groq (Llama 3.3 70B)")
    if has_gemini:
        providers.append("Google Gemini (2.5 Flash)")

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="sidebar-brand-icon">{SVG_POLICY}</div>
                <div class="sidebar-brand-name">PolicyBot</div>
                <div class="sidebar-brand-sub">Policy Assistant</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Model
        st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)

        if not providers:
            st.error("No API keys found in `.env` file. Add `GROQ_API_KEY` and/or `GOOGLE_API_KEY`.")
            st.stop()

        provider = st.selectbox(
            "Provider",
            providers,
            help="Switch between available LLM providers. Keys loaded from .env",
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Lower = precise and factual. Higher = creative and varied.",
        )

        # API Status
        st.markdown('<div class="sidebar-section">API Status</div>', unsafe_allow_html=True)

        if has_groq:
            st.markdown('<span class="status-badge badge-ready">● Groq Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-error">✕ Groq Not Configured</span>', unsafe_allow_html=True)

        if has_gemini:
            st.markdown('<span class="status-badge badge-ready">● Gemini Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-error">✕ Gemini Not Configured</span>', unsafe_allow_html=True)

        # Token Usage
        st.markdown('<div class="sidebar-section">Token Usage (Daily)</div>', unsafe_allow_html=True)

        usage = get_token_usage()
        for prov_name, key in [("Groq (Llama 3.3 70B)", "groq"), ("Google Gemini (2.5 Flash)", "gemini")]:
            used = usage.get(key, 0)
            limit = TOKEN_LIMITS.get(prov_name, 100_000)
            pct = min(used / limit * 100, 100)
            color_class = "green" if pct < 60 else ("yellow" if pct < 85 else "red")
            label = "Groq" if key == "groq" else "Gemini"
            st.markdown(
                f'<div class="token-bar-container">'
                f'<div class="token-label"><span>{label}</span><span>{used:,} / {limit:,}</span></div>'
                f'<div class="token-bar-bg"><div class="token-bar-fill {color_class}" style="width:{pct}%"></div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Filter
        st.markdown('<div class="sidebar-section">Filter</div>', unsafe_allow_html=True)

        companies = ["All Companies"]
        if os.path.isdir(DOCS_DIR):
            for fname in sorted(os.listdir(DOCS_DIR)):
                meta = extract_metadata_from_filename(os.path.join(DOCS_DIR, fname))
                c = meta["company"]
                if c != "Unknown" and c not in companies:
                    companies.append(c)

        selected_company = st.selectbox(
            "Company",
            companies,
            help="Narrow retrieval to a single company's documents.",
        )

        st.markdown("---")

        # Upload
        st.markdown('<div class="sidebar-section">Upload Documents</div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload policy documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload PDF or TXT files to add to the knowledge base.",
            label_visibility="collapsed",
        )

        if uploaded_files:
            if st.button("Process & Add to KB", use_container_width=True, type="primary"):
                embeddings = get_embedding_model()
                handle_file_upload(uploaded_files, embeddings)

        st.markdown("---")

        # Knowledge Base
        st.markdown('<div class="sidebar-section">Knowledge Base</div>', unsafe_allow_html=True)

        docs_count = 0
        doc_files = []
        if os.path.isdir(DOCS_DIR):
            for f in sorted(os.listdir(DOCS_DIR)):
                fp = os.path.join(DOCS_DIR, f)
                if os.path.isfile(fp):
                    docs_count += 1
                    size = os.path.getsize(fp)
                    ext = Path(f).suffix.lower()
                    icon = "TXT" if ext == ".txt" else "PDF" if ext == ".pdf" else "DOC"
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f} MB"
                    doc_files.append((icon, f, size_str))

        if doc_files:
            docs_html = "".join(
                f'<div class="doc-item">'
                f'<span class="doc-icon" style="font-size:0.58rem;font-weight:700;color:var(--accent);background:var(--accent-soft);padding:0.12rem 0.3rem;border-radius:3px;">{icon}</span>'
                f'<span class="doc-name">{name}</span>'
                f'<span class="doc-size">{size}</span>'
                f'</div>'
                for icon, name, size in doc_files
            )
            st.markdown(docs_html, unsafe_allow_html=True)
        else:
            st.caption("No documents loaded yet.")

        st.markdown("---")

        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Rebuild KB", use_container_width=True):
                get_vector_store.clear()
                hash_file = os.path.join(CHROMA_DIR, ".docs_hash")
                if os.path.exists(hash_file):
                    os.remove(hash_file)
                st.rerun()

    # ── Guard: need docs ──
    if not os.path.isdir(DOCS_DIR) or docs_count == 0:
        st.markdown(
            f"""
            <div class="welcome-container">
                <div class="welcome-icon">{SVG_FOLDER}</div>
                <div class="welcome-title">No Documents Found</div>
                <div class="welcome-subtitle">
                    Upload PDF or TXT policy documents using the sidebar,<br>
                    or place files in the <code>docs/</code> folder to get started.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Metrics ──
    embeddings = get_embedding_model()
    docs_hash = _compute_docs_hash(DOCS_DIR)
    vector_store = get_vector_store(embeddings, docs_hash)

    try:
        collection = vector_store._collection
        chunk_count = collection.count()
    except Exception:
        chunk_count = "—"

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{docs_count}</div>
                <div class="metric-label">Documents</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{chunk_count}</div>
                <div class="metric-label">Chunks</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(companies) - 1}</div>
                <div class="metric-label">Companies</div>
            </div>
            <div class="metric-card">
                <div class="metric-value"><span class="status-dot{"" if providers else " offline"}"></span></div>
                <div class="metric-label">LLM Status</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Build retriever
    search_kwargs: dict = {"k": 15, "fetch_k": 30}
    if selected_company != "All Companies":
        search_kwargs["filter"] = {"company": selected_company}

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )
    llm = get_llm(provider, temperature)
    rag_chain = build_rag_chain(llm, retriever)

    # ── Chat State ──
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Welcome ──
    if not st.session_state.messages:
        st.markdown(
            f"""
            <div class="welcome-container">
                <div class="welcome-icon">{SVG_CHAT}</div>
                <div class="welcome-title">How can I help you?</div>
                <div class="welcome-subtitle">
                    Ask questions about your corporate policies. Responses are grounded<br>
                    in your uploaded documents with source citations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        suggestions = [
            "How many paid sick days do employees at Microsoft get?",
            "How does Tesla handle labor disputes?",
            "How do I report a code of conduct violation?",
            "What are the data privacy rules for vendors?",
        ]
        cols = st.columns(len(suggestions))
        for i, (col, suggestion) in enumerate(zip(cols, suggestions)):
            with col:
                if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    st.rerun()

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "tokens" in msg:
                st.markdown(
                    f'<span class="token-pill">~{msg["tokens"]:,} tokens</span>',
                    unsafe_allow_html=True,
                )

    # ── Chat Input ──
    prompt = st.chat_input("Ask a question about corporate policies …")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generate response if the last message is from the user
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_input = st.session_state.messages[-1]["content"]

        allowed, used, limit = check_token_limit(provider)
        if not allowed:
            st.error(
                f"Daily token limit reached for **{provider}** ({used:,}/{limit:,}). "
                f"Switch providers or wait until tomorrow."
            )
            st.stop()

        chat_history = []
        for m in st.session_state.messages[:-1]:
            if m["role"] == "user":
                chat_history.append(HumanMessage(content=m["content"]))
            else:
                chat_history.append(AIMessage(content=m["content"]))

        # Show loading indicator then generate
        with st.chat_message("assistant"):
            loading_placeholder = st.empty()
            loading_placeholder.markdown(
                '<div class="loading-indicator">'
                '<div class="loading-dots"><span></span><span></span><span></span></div>'
                'Searching policies and generating response…'
                '</div>',
                unsafe_allow_html=True,
            )

            try:
                answer, tokens_used = rag_chain(
                    {"question": user_input, "chat_history": chat_history}
                )

                loading_placeholder.empty()
                st.markdown(answer)
                st.markdown(
                    f'<span class="token-pill">~{tokens_used:,} tokens</span>',
                    unsafe_allow_html=True,
                )

                track_tokens(provider, tokens_used)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "tokens": tokens_used}
                )
            except Exception as exc:
                loading_placeholder.empty()
                error_msg = str(exc)
                if "429" in error_msg or "rate" in error_msg.lower():
                    st.error(
                        "Rate limit hit. Please wait a moment and try again. "
                        "Free-tier APIs allow ~15-30 requests/minute."
                    )
                else:
                    st.error(f"Error: {error_msg}")

    # ── Footer ──
    st.markdown(
        """
        <div class="footer-bar">
            Powered by <a href="https://www.langchain.com">LangChain</a> ·
            <a href="https://www.trychroma.com">ChromaDB</a> ·
            <a href="https://streamlit.io">Streamlit</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
