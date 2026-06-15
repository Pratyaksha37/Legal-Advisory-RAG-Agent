# Indian Legal Advisory RAG Platform

Retrieval-Augmented Generation (RAG) system designed for Indian Law. Unlike generic conversational chatbots, this platform operates as an intelligent legal research assistant. 
It delivers structured, verifiable, and fully grounded legal reasoning based strictly on official documents (Constitution of India, BNS, BNSS, etc.), complete with exact citations and confidence scoring.

## 🌟 Project Vision & Core Principles

- **Trust & Grounding**: The system relies strictly on retrieved official legal texts. The Large Language Model (LLM) acts purely as a synthesizer and simplifier—not a source of legal knowledge.
- **Source Citation**: Every response is linked to specific legal provisions (e.g., Article 21 of the Constitution, Section 103 of BNS). If the necessary grounding is absent, the system refuses to answer.
- **Explainability & Observability**: Every step—from normalization to result fusion and guardrail enforcement—is logged in detail with full telemetry.
- **Extensible Legal Knowledge**: Designed to seamlessly scale from the initial corpus (Fundamental Rights) to the entire Constitution of India, BNS, BNSS, legacy IPC, rules/regulations, and Supreme Court / High Court judgments.


## 🏗️ System Architecture

### Query & Inference Flow
```
User Query
   │
   ▼
Input Validation (Injection protection, scope check)
   │
   ▼
Query Normalization (Legal terms expansion, abbreviation mapping)
   │
   ▼
Hybrid Retrieval ──────────────┐
   ├── Vector Search (Semantic) │
   └── BM25 Search (Lexical)    │
   │                            │
   ▼                            ▼
Result Fusion (e.g., Reciprocal Rank Fusion)
   │
   ▼
Re-ranking Layer (Future Expansion)
   │
   ▼
Context Selection (Token-budget optimization, relevance filtering)
   │
   ▼
LLM Generation (Groq API, strict context-only constraints)
   │
   ▼
Guardrail Validation (Hallucination check, citation alignment)
   │
   ▼
Confidence Calculation (RAGUS-based similarity & retrieval scoring)
   │
   ▼
Telemetry Logging (Full pipeline trace, latency, evaluation metrics)
   │
   ▼
Final Response (Answer + Citations + Confidence score)
```

## 📥 Ingestion Pipeline
To preserve legal meaning, documents are processed through a structured pipeline rather than being directly chunked by arbitrary token lengths:
```
Official Documents (PDF, DOCX, TXT, JSON)
   │
   ▼
Text Extraction (Avoiding scanned OCR where possible)
   │
   ▼
Cleaning & Normalization
   │
   ▼
Metadata Extraction
   │
   ▼
Structured JSON Corpus (Source of Truth)
   ├──► BM25 Index Generator ──► BM25 Index
   └──► Embedding Generator   ──► FAISS Vector Index
```

### Chunking Strategy
Chunks are bound to **legal structures** (e.g., one complete Article, one complete Section, one Rule, or one Clause Group). This ensures the structural hierarchy and textual integrity of legal code are preserved.

## 🗂️ Knowledge Base Schema Design
All retrieval indexes are compiled from a single, unified source-of-truth JSON corpus. Each provision or logical division fits the following flexible schema:
```json
{
  "id": "const-india-art-21",
  "law_name": "Constitution of India",
  "law_abbreviation": "COI",
  "document_type": "Constitution",
  "chapter": "Part III",
  "part": "Fundamental Rights",
  "section_type": "Article",
  "section_number": "21",
  "title": "Protection of life and personal liberty",
  "content": "No person shall be deprived of his life or personal liberty except according to procedure established by law.",
  "summary": "Protects the fundamental right to life and personal liberty.",
  "keywords": ["life", "liberty", "procedure", "deprived"],
  "category": "Fundamental Rights",
  "sub_category": "Right to Freedom",
  "jurisdiction": "India",
  "language": "English",
  "effective_date": "1950-01-26",
  "last_updated": "2026-06-16",
  "related_provisions": ["const-india-art-20", "const-india-art-22"],
  "cross_references": [],
  "citations": ["AIR 1978 SC 597"],
  "source_document": "constitution_of_india.pdf",
  "source_url": "https://www.india.gov.in/my-government/constitution-india",
  "source_type": "Official PDF",
  "version": "1.0",
  "metadata": {}
}
```

## 🔍 Hybrid Retrieval Engine

Hybrid retrieval merges lexical precision with semantic understanding:
1. **Vector Search (FAISS)**: Uses HuggingFace embeddings (`BAAI/bge-large-en-v1.5`) to capture semantic context.
2. **BM25 Search (rank-bm25)**: Handles precise identifiers (e.g., "Section 103", "Article 21") and domain-specific terminology.
3. **Fusion & Deduping**: Combines scores and eliminates duplicate references before context construction.

## 🛡️ Guardrails & Safety Layers
- **Input Validation**: Rejects query injection attempts and out-of-scope prompts.
- **Context-Only Generation**: Strict system prompts instruct the Groq LLM (e.g., `llama-3.3-70b-versatile`) to generate answers based solely on the provided context.
- **Citation Enforcement**: Requires explicit references for statements made.
- **Hallucination Prevention**: Rejects answers where claims cannot be backed by the retrieved text.
- **Refusal Default**: Defaults to a standardized fallback message if context is insufficient: 
  > *"I could not find sufficient information in the available legal corpus."*

## 📈 Telemetry, Observability & Evaluation

### Telemetry Logging
Every inference call logs detailed performance metrics:
- Query details, retrieval scores (vector & BM25), selected context, LLM responses.
- Active guardrail triggers, latency breakdowns, and system confidence scores.
- Telemetry outputs are designed to integrate with **LangSmith**, **OpenTelemetry**, and **Weights & Biases**.
### Confidence Scoring (RAGUS)
Computes a confidence score ($[0.0, 1.0]$) in the [confidence service](file:///Users/pratyaksha_003/Desktop/Legal%20Rag/src/confidence) based on:
- Retrieval matching quality and similarity scores.
- BM25 term overlap.
- Number of supporting documents.
### Automated Evaluation
Includes test configurations to run metrics using **RAGAS**:
- Faithfulness
- Answer Relevancy
- Context Precision & Recall
- Citation Correctness

## 📁 Project Structure
The codebase is organized into highly isolated, domain-specific modules:
```
legal-rag-platform/
├── data/                             # Raw text, processed corpus, and compiled indexes
│   ├── raw/
│   ├── processed/
│   └── indexes/
├── src/
│   ├── api/                          # FastAPI route handlers
│   ├── chunking/                     # Structurally aware legal text chunkers
│   ├── confidence/                   # RAGUS confidence scoring system
│   ├── config/                       # Application settings and environment parser
│   ├── core/                         # Standard logging and exceptions
│   ├── embeddings/                   # Embedding generation pipelines
│   ├── evaluation/                   # RAGAS metrics & validation logic
│   ├── generation/                   # LLM synthesis interface (Groq wrapper)
│   ├── guardrails/                   # Pre-processing and post-processing safety layers
│   ├── ingestion/                    # Raw file extractors (PDF, DOCX, etc.)
│   ├── pipeline/                     # Modular orchestrator linking retrieval and generation
│   ├── preprocessing/                # Text cleaning and normalizing utilities
│   ├── retrieval/                    # Search modules (vector, bm25, hybrid)
│   │   ├── bm25/
│   │   ├── hybrid/
│   │   └── vector/
│   ├── schema/                       # Pydantic schemas for request/response & corpus mapping
│   ├── telemetry/                    # Latency, tracing, and metrics exporter
│   ├── utils/                        # Generic helper tools
│   └── main.py                       # FastAPI application entrypoint
├── tests/                            # Unit and integration tests
├── configs/                          # Static configuration files
├── logs/                             # Local runtime log storage
├── scripts/                          # Administration, evaluation, and build scripts
├── README.md                         # Project documentation
├── pyproject.toml                    # Poetry/Setuptools configuration and dependency list
├── Dockerfile                        # Backend image specification
├── docker-compose.yml                # Multi-service setup (App, external indices, etc.)
└── .env.example                      # Environment variable templates
```

## ⚡ API Interface

### **POST** `/ask`
**Request Body**
```json
{
  "question": "What is the fundamental right guaranteed under Article 21?"
}
```
**Response Body**
```json
{
  "answer": "Article 21 of the Constitution of India guarantees the protection of life and personal liberty. It mandates that no person shall be deprived of their life or personal liberty except according to the procedure established by law.",
  "sources": [
    {
      "law_name": "Constitution of India",
      "law_abbreviation": "COI",
      "section_type": "Article",
      "section_number": "21",
      "title": "Protection of life and personal liberty"
    }
  ],
  "confidence": 0.96,
  "metadata": {
    "latency_ms": 142.5,
    "retrieval_method": "hybrid"
  }
}
```

## ⚙️ Setup & Installation

1. **Clone the Repository & Navigate**:
   ```bash
   git clone <repository_url>
   cd legal-rag-platform
   ```
2. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   # Add your Groq API Key and adjust configuration variables as needed
   ```
3. **Install Dependencies**:
   If using `pip`:
   ```bash
   pip install -r requirements.txt
   ```
   Or using standard pyproject.toml environments (e.g. `pip install .` or with `poetry`/`pipenv`):
   ```bash
   pip install -e .
   ```
4. **Run the Application**:
   ```bash
   python -m uvicorn src.main:app --reload
   ```
