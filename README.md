# 📜 Smart Contract Assistant

## Overview

A **Retrieval-Augmented Generation (RAG)** web application for analysing legal contracts and documents. Users upload PDF, DOCX, or TXT files through a Gradio interface, which automatically indexes them in a FAISS vector store. The system then answers questions about the uploaded documents with **source citations**, **structured analysis**, and **LLM-as-a-Judge quality evaluation**.

Built with **LangChain Expression Language (LCEL)**, **FAISS** vector search, and **Hugging Face** free-tier models.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Smart Contract Assistant                         │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌────────────┐ │
│  │ Ingestion│──▶│  FAISS Index  │──▶│ Retriever│──▶│  LLM Gen   │ │
│  │ Pipeline │   │ (Embeddings) │   │  (Top-8) │   │ (QA Chain) │ │
│  └──────────┘   └──────────────┘   └──────────┘   └────────────┘ │
│       ▲                                                  │        │
│       │              ┌──────────────┐                    ▼        │
│  PDF/DOCX/TXT        │   Gradio UI  │◀──── Answer + Sources      │
│  (Auto-Ingest)       └──────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Description |
|-----------|-----------|-------------|
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | Local, free, 384-dimensional embeddings |
| **LLM** | `meta-llama/Llama-3.1-8B-Instruct` | Cloud (HuggingFace Inference API, free tier) |
| **Vector Store** | FAISS (Facebook AI Similarity Search) | Local, persistent index with incremental merge |
| **Ingestion** | PyPDFLoader, Docx2txtLoader, RecursiveCharacterTextSplitter | Multi-format parsing with 800-char chunks, 200 overlap |
| **RAG Chain** | LangChain LCEL with Long Context Reorder | Dynamic retriever with document re-ranking |
| **Knowledge Base** | Pydantic BaseModel | Running-state tracking (parties, dates, clauses, financials) |
| **Evaluation** | LLM-as-a-Judge | Automated scoring: relevance, groundedness, completeness |
| **Frontend** | Gradio Blocks | Multi-tab UI with auto-ingestion and emerald theme |

## Project Structure

```text
Smart_Contract_Assistant/
│
├── data/                       # Uploaded documents (cleared on each run)
├── vector_store/               # FAISS index (cleared on each run)
├── notebooks/
│   └── Smart_Contract_Assistant_Complete_final_1.ipynb   # Complete notebook
├── .env                        # API Keys (HUGGINGFACEHUB_API_TOKEN)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Notebook Sections

The application is contained in a single notebook (`Smart_Contract_Assistant_Complete_final_1.ipynb`) divided into 4 sections:

### Section 1: Configuration & API Setup
- **1.1** — Package verification (14 dependencies)
- **1.2** — Rich console with emerald-themed output
- **1.3** — LLM configuration (`Llama-3.1-8B-Instruct`, `max_new_tokens=1100`, `temp=0.2`, `repetition_penalty=1.4`)
- **1.4** — Embedding sanity check with cosine similarity
- **1.5** — Simple LCEL chain validation
- **1.6** — Utility runnables (`RPrint()`, `docs2str()`)

### Section 2: Document Ingestion Pipeline (ETL)
- **2.1** — Fresh start: clears `data/` and `vector_store/` on every run
- **2.2** — Document loader (PDF, DOCX, TXT)
- **2.3** — Text splitter (800-char chunks, 200 overlap)
- **2.4** — Cumulative ingestion with FAISS merge
- **2.5** — Batch ingestion with progress tracking and file sizes
- **2.5b** — Vector store reset utility
- **2.6** — FAISS index inspection and similarity search test

### Section 3: RAG Chain Logic
- **3.1** — Imports
- **3.2** — FAISS loading with dummy document fallback + top-8 retriever
- **3.3** — RAG prompt (demands detailed, structured answers with source citations; bilingual Arabic/English)
- **3.5** — Advanced LCEL chain with **dynamic retriever** (`RunnableLambda`) and **grouped context** by source file
- **3.6** — `ask_question()` helper with unique source de-duplication
- **3.7** — Pydantic `ContractKnowledge` model (parties, dates, clauses, financials, summary)
- **3.8** — Conversational `ask_with_history()` with history tracking
- **3.9** — RAG evaluation chain (LLM-as-a-Judge with JSON output)

### Section 4: Interactive UI & Assessment
- **4.1** — UI imports and session stats initialization
- **4.2** — `process_upload()` — auto-ingests files, reloads vectorstore, rebuilds retriever + entire chain
- **4.3** — `chat_fn()` with `<think>` tag removal and markdown cleanup
- **4.4** — UI wrappers (`streaming_chat`, `auto_ingest_wrapper`)
- **4.5** — Gradio Blocks UI with 2 tabs:
  - **💬 Chat & Upload** — Document upload (multi-file) with auto-ingestion + ChatInterface with example prompts
  - **📊 Analysis & Export** — Session stats, quality evaluation, knowledge export (JSON)
- **4.6** — Server launch with `gr.close_all()`, emerald Soft theme, and custom CSS

## Key Features

- ✅ **Fresh Start** — `data/` and `vector_store/` are cleared on each run to avoid stale data
- ✅ **Multi-File Upload** — Upload multiple PDF/DOCX/TXT files simultaneously
- ✅ **Auto-Ingestion** — Files are automatically indexed when uploaded (no manual step)
- ✅ **Dynamic Retriever** — Chain always uses the latest retriever after new uploads
- ✅ **Grouped Context** — Retrieved chunks are grouped by source file to prevent the LLM from treating chunks as separate documents
- ✅ **Bilingual Support** — Responds in Arabic or English depending on the question
- ✅ **Source Citations** — Every answer includes cited source documents and page numbers
- ✅ **LLM-as-a-Judge** — Automated evaluation scoring (relevance, groundedness, completeness)
- ✅ **Knowledge Export** — Export session data, knowledge base, and conversation history as JSON
- ✅ **Anti-Repetition** — `repetition_penalty=1.4` to prevent repeated output

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (Python 3.11 recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 2. Configure API Key

Create or edit `.env` in the project root:

```
HUGGINGFACEHUB_API_TOKEN=hf_your_actual_token_here
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 3. Run the Notebook

Open in Jupyter or VS Code and run all cells:

```bash
jupyter lab notebooks/Smart_Contract_Assistant_Complete_final_1.ipynb
```

### 4. Upload & Chat

1. The Gradio UI will launch automatically at `http://localhost:7860` (or the next available port).
2. Go to the **💬 Chat & Upload** tab.
3. Upload one or more contract files (PDF/DOCX/TXT).
4. Wait for the **Ingestion Status** to confirm indexing.
5. Ask questions in the chat (e.g., "Summarize all uploaded contracts", "ما هي شروط الدفع؟").
6. View evaluation metrics in the **📊 Analysis & Export** tab.

## LLM Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_new_tokens` | 1100 | Longer, more detailed answers |
| `temperature` | 0.2 | Factual, grounded output |
| `top_p` | 0.95 | Controlled sampling |
| `repetition_penalty` | 1.4 | Prevents repeated phrasing |
| `do_sample` | True | Natural language generation |

## Requirements

```
langchain, langchain-community, langchain-huggingface, langchain-text-splitters
faiss-cpu, gradio, pypdf, python-docx, python-dotenv
huggingface_hub, sentence-transformers, ipykernel
docx2txt, rich, pydantic, matplotlib, numpy, scikit-learn
```

## License

This project is for educational and research purposes.
