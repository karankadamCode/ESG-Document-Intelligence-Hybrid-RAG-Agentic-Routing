# ESG Document Intelligence – Hybrid RAG Case Study

A production-oriented **document intelligence system** for asking natural-language questions over large ESG and sustainability reports and receiving **grounded, explainable answers with citations**.

This project demonstrates how long, complex ESG documents (governance, climate risk, emissions, human capital, compliance) can be made searchable and trustworthy using **Hybrid Retrieval-Augmented Generation (RAG)** with evaluation and monitoring built in.

**Author:** Karan Kadam

---

## Problem Statement

### Domain

**Environmental, Social, and Governance (ESG) & Financial Reporting**

Organizations publish detailed ESG and sustainability reports disclosing information related to climate strategy, emissions, governance, risk management, human capital, and regulatory commitments. These documents are publicly available but are typically:

* 100+ pages long
* Written in dense, compliance-driven language
* Published as static PDFs
* Difficult to query precisely using traditional keyword search

---

### Use Case

**Explainable Question-Answering over ESG Reports**

Stakeholders such as compliance teams, ESG analysts, investors, sustainability officers, and auditors need to ask natural-language questions like:

* “What is the company’s climate risk management approach?”
* “What emissions reduction targets are disclosed?”
* “Which sections describe ESG governance and oversight?”

and receive answers that are:

* Accurate and concise
* Strictly grounded in the source document
* Accompanied by explicit citations (source + page)
* Free from hallucinations or external assumptions

---

### Problem Being Solved

Traditional approaches suffer from key limitations:

* Keyword search returns incomplete or irrelevant sections
* Manual review is slow and not scalable
* Naive LLM usage risks hallucinated, unverifiable answers
* Lack of explainability and traceability makes outputs unsuitable for audits

There is a need for a system that understands query intent, retrieves the right evidence, generates grounded answers only, and supports evaluation and monitoring.

---

## Goal

Provide accurate, explainable Q&A over ESG / sustainability reports by:

* Ingesting large PDF reports (e.g., ESG disclosures)
* Chunking and embedding content for semantic search
* Combining **vector search + lexical (BM25) search**
* Using an **LLM-based router** to decide how each query is handled
* Generating answers **strictly grounded in retrieved content**
* Returning **explicit citations (source + page)**
* Exposing the system via **Streamlit UI** and **FastAPI**
* Measuring answer quality using **DeepEval** and tracing with **Langfuse**

---

## Architecture (High Level)

```text
┌──────────────────────────┐
│      ESG PDF Reports     │
│   (Sustainability / ESG) │
└─────────────┬────────────┘
              │  (offline ingestion)
              ▼
┌──────────────────────────┐
│   PDF Loader             │
│   Page-wise extraction   │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│   Chunking               │
│   Recursive splitter     │
│   (overlap enabled)      │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│   Embeddings             │
│   OpenAI embeddings      │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│   Vector Store           │
│   FAISS (local, disk)    │
└─────────────┬────────────┘
              │
              │  (online query time)
              ▼
┌──────────────────────────┐
│   Agentic Router (LLM)   │
│   - rag / summarize      │
│   - extract_kpi          │
│   - vector / lexical /  │
│     hybrid retrieval     │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│   Retrieval Layer        │
│   - Vector search        │
│   - BM25 lexical search  │
│   - Hybrid fusion        │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│   Optional Reranking     │
│   LLM JSON-based rerank  │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│   Grounded Answering     │
│   Strict RAG prompt      │
│   Citations enforced    │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│   Interfaces             │
│   - Streamlit UI         │
│   - FastAPI (/qa)        │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ Evaluation & Monitoring  │
│ DeepEval + Langfuse      │
└──────────────────────────┘
```

---

## Key Technical Choices (Brief)

* **LLM:** `gpt-4o-mini`
  Balanced choice for quality, latency, and cost. Used for routing, reranking, and answer generation.

* **Vector Store:** **FAISS**
  Lightweight, fast, and locally persisted for deterministic retrieval.

* **Lexical Search:** **BM25**
  Used for exact KPIs, numeric values, and compliance-driven queries.

* **Chunking Strategy:** **Recursive chunking with overlap**
  Preserves semantic boundaries and prevents loss of context.

* **Routing:** **LLM-based agentic router**
  Dynamically selects query intent and optimal retrieval strategy.

* **Evaluation & Tracing:** **DeepEval + Langfuse**
  Enables faithfulness evaluation, traceability, and monitoring.

---

## Features

* Hybrid retrieval (Vector + BM25)
* Agentic query routing
* Optional LLM reranking
* Strictly grounded answers with citations
* Multi-threaded Streamlit chat UI
* Streaming token-level responses
* Retrieval transparency (sources, pages, chunk previews)
* FastAPI endpoint for programmatic access
* Unit + smoke tests
* Faithfulness evaluation using retrieved context

---

## Repository Structure

```bash
├── app.py                 # Streamlit UI
├── api.py                 # FastAPI wrapper
├── main.py                # Core Hybrid RAG + routing pipeline
├── faiss_ingest.py        # PDF ingestion & FAISS index creation
├── prompt_manager.py      # Prompt loading/versioning
├── evals.py               # DeepEval + Langfuse integration
├── settings.py            # Central config
├── prompts/
│   ├── rag/               # Grounded RAG prompts
│   ├── rerank/            # JSON reranking prompt
│   └── router/            # Agentic router prompt
├── tests/
│   ├── smoke tests
│   ├── retrieval contract tests
│   └── LLM eval tests
├── vectorstore/           # FAISS index (generated)
├── data/                  # ESG PDFs
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Evaluation & Testing

* **DeepEval metrics**

  * Faithfulness
  * Relevance
  * Context precision
* **Tests**

  * Smoke tests (non-LLM)
  * Retrieval contract tests
  * Optional LLM eval tests (flag-controlled)

---

## Production Considerations Identified

* Typed API contracts (FastAPI + Pydantic)
* Non-blocking evaluation and tracing
* Deterministic prompts with strict output rules
* Separation of ingestion vs query-time logic
* Secure containerization (non-root Docker user)
* Extensible design for multi-document and agent workflows

