# 📜 Smart Contract Summary & Q&A Assistant

## Overview

This project is a **Retrieval-Augmented Generation (RAG)** web application designed to process long-form documents such as legal contracts and insurance policies. The system allows users to upload PDF, DOCX, or TXT files, which are then analysed to answer user queries with **strict source citations**, **optional summarisation**, and **quality evaluation**.

Built using **LangChain Expression Language (LCEL)**, **FAISS** vector search, and **Hugging Face** free-tier models, this project follows the patterns and best practices from the NVIDIA DLI course notebooks.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Smart Contract Assistant                         │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌────────────┐ │
│  │ Ingestion│──▶│  FAISS Index  │──▶│ Retriever│──▶│  LLM Gen   │ │
│  │ Pipeline │   │ (Embeddings) │   │  (Top-K) │   │ (QA Chain) │ │
│  └──────────┘   └──────────────┘   └──────────┘   └────────────┘ │
│       ▲                                                  │        │
│       │              ┌──────────────┐                    ▼        │
│  PDF/DOCX/TXT        │   Gradio UI  │◀──── Answer + Sources      │
│                      └──────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Description |
|-----------|-----------|-------------|
| **Ingestion Engine** | PyPDFLoader, Docx2txtLoader, RecursiveCharacterTextSplitter | Parses documents, splits into overlapping chunks |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, 384-dimensional embeddings |
| **Vector Store** | FAISS (Facebook AI Similarity Search) | Local, persistent vector index with incremental updates |
| **LLM** | `deepseek-ai/DeepSeek-R1-0528` (HuggingFace Inference API) | Free cloud LLM for answer generation |
| **RAG Chain** | LangChain LCEL with Long Context Reorder | Advanced retrieval with document re-ranking |
| **Knowledge Base** | Pydantic BaseModel | Running-state tracking of contract details |
| **Evaluation** | LLM-as-a-Judge | Automated scoring of relevance, groundedness, completeness |
| **Frontend** | Gradio Blocks | Multi-tab UI (Upload, Chat, Analysis Dashboard) |

## Project Structure

```text
Smart_Contract_Assistant/
│
├── data/                   # Store uploaded PDF/DOCX/TXT files
├── vector_store/           # FAISS index (persistent)
├── notebooks/              # Implementation steps (run sequentially)
│   ├── 01_Setup_and_Config.ipynb     # Environment, models, LCEL basics
│   ├── 02_Ingestion_Pipeline.ipynb   # ETL: Load → Split → Embed → Index
│   ├── 03_RAG_Chain_Logic.ipynb      # RAG chain, evaluation, knowledge base
│   └── 04_App_UI.ipynb              # Gradio UI (Upload + Chat + Analysis)
├── .env                    # API Keys (HUGGINGFACEHUB_API_TOKEN)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Features

### Notebook 1: Configuration & APIs

- ✅ Environment verification and dependency checking
- ✅ Rich console output for beautiful formatting
- ✅ Embedding model exploration with cosine similarity
- ✅ Basic LCEL chain test
- ✅ Utility runnables (`RPrint`, `docs2str`)

### Notebook 2: Ingestion Pipeline (ETL)

- ✅ Multi-format document loading (PDF, DOCX, TXT)
- ✅ Configurable `RecursiveCharacterTextSplitter` (800 chars, 200 overlap)
- ✅ FAISS vector indexing with **incremental merge** support
- ✅ Batch ingestion for multiple files
- ✅ PCA visualisation of chunk embeddings
- ✅ Running-state document summarisation

### Notebook 3: RAG Chain Logic

- ✅ Simple RAG chain (high-level `create_retrieval_chain`)
- ✅ Advanced LCEL RAG chain with **Long Context Reorder**
- ✅ Source citation in answers (document name + page number)
- ✅ Pydantic **ContractKnowledge** running state tracking
- ✅ Conversation history management
- ✅ **LLM-as-a-Judge** evaluation (relevance, groundedness, completeness)

### Notebook 4: Gradio UI

- ✅ Tab 1: Document upload with ingestion status
- ✅ Tab 2: Chat interface with sample questions
- ✅ Tab 3: Analysis dashboard (stats, evaluation, export)
- ✅ Custom theming with emerald accent
- ✅ Session tracking and knowledge export

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 2. Configure API Key

Edit `.env` and add your Hugging Face token:

```
HUGGINGFACEHUB_API_TOKEN=hf_your_actual_token_here
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 3. Run the Notebooks

Open in Jupyter/VS Code and run notebooks **in order** (1 → 2 → 3 → 4):

```bash
jupyter lab notebooks/
```

### 4. Upload & Chat

1. In Notebook 4, the Gradio UI will launch at `http://localhost:7860`.
2. Upload a contract (PDF/DOCX/TXT) via the Upload tab.
3. Ask questions in the Chat tab.
4. Review quality metrics in the Analysis tab.

## Reference Course Notebooks

This project is built on patterns from:

- **Notebook 03:** LangChain Expression Language (LCEL) and chains
- **Notebook 04:** Running State Chains and Knowledge Bases
- **Notebook 05:** Document loading, chunking, and summarisation
- **Notebook 06:** Embedding models and similarity
- **Notebook 07:** FAISS vector stores and RAG workflows
- **Notebook 08:** LLM-as-a-Judge evaluation
- **Notebook 09:** LangServe and API deployment
