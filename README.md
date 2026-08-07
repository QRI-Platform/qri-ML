# QRI — Multi-Tenant Stateful RAG API

> **Author:** VashuTheGreat &nbsp;|&nbsp; **Stack:** FastAPI · LangGraph · Pinecone · Groq · PostgreSQL (Neon) · Docling

An industrial-grade, **fully async**, stateful Multi-Tenant RAG (Retrieval-Augmented Generation) backend. Each user session is isolated by `thread_id`. Documents are indexed into a personal Pinecone namespace; the LLM answers questions grounded in those documents while maintaining persistent long-term memory across sessions.

---

## Table of Contents

1. [Tech Stack](#tech-stack)
2. [Features](#features)
3. [Architecture Overview](#architecture-overview)
4. [LangGraph Workflow](#langgraph-workflow)
5. [End-to-End Request Flow](#end-to-end-request-flow)
6. [Directory Layout](#directory-layout)
7. [System Constants](#system-constants)
8. [API Reference](#api-reference)
9. [Performance & Latency Optimizations](#performance--latency-optimizations)
10. [Developer Guide](#developer-guide)
11. [Running Locally](#running-locally)
12. [Environment Variables](#environment-variables)

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Web Framework** | FastAPI + Uvicorn (async) |
| **AI Orchestration** | LangGraph (StateGraph) |
| **LLM** | Groq — `llama-3.3-70b-versatile` |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (384-dim) |
| **Vector Store** | Pinecone Serverless (AWS us-east-1, cosine) |
| **Checkpointer** | PostgreSQL via Neon (`AsyncPostgresSaver`) |
| **Long-Term Memory** | PostgreSQL via Neon (`AsyncPostgresStore`) |
| **Document Parsing** | Docling (PDF, DOCX, tables, OCR) |
| **Monitoring** | LangSmith tracing |
| **Math Tool** | numexpr (`solver` tool) |
| **Package Manager** | uv (Python 3.14+) |

---

## Features

- ✅ **Stateful Multi-Tenant RAG** — each `thread_id` has its own Pinecone namespace and conversation history
- ✅ **`@filename` Scoped Retrieval** — user types `@report.pdf` in chat to restrict retrieval to that file only
- ✅ **Multimodal Document Parsing** — Docling extracts text, tables (→ markdown), and OCR from images
- ✅ **Parallel Pinecone Search** — multiple queries fired simultaneously via `asyncio.gather()`
- ✅ **Non-Blocking Ingestion** — Docling runs in thread-pool executor, event loop stays free
- ✅ **Long-Term Memory** — LLM auto-extracts user facts and persists them across all sessions
- ✅ **Conversation Summarization** — old messages compressed into a summary to stay within context window; uses `add_messages` reducer to cleanly prune state history
- ✅ **Single-Call `PydanticOutputParser`** — dynamic schema injection with `{format_instructions}`, avoiding prompt hardcoding and dual-LLM overhead
- ✅ **Math Tool Calling** — LLM can invoke `solver(expression)` for calculations
- ✅ **Structured SSE Event Streaming** — typed JSON SSE stream (`tool_start`, `tool_end`, `token`, `final`, `error`) parsed via modern Python `match-case`
- ✅ **Full Swagger Docs** — `/docs` with examples, response schemas, and descriptions
- ✅ **LangSmith Monitoring** — every node and LLM call traced

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 7 — API & Presentation  (api/)                                       │
│  graph_routes.py · user_routes.py · middlewares · schemas                   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────────┐
│  LAYER 6 — Pipeline  (src/pipelines/)                                       │
│  GraphRunnerPipeline → astream_events() → SSE chunks                        │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────────┐
│  LAYER 5 — LangGraph Workflow  (src/graphs/ · src/nodes/)                   │
│  builder.py compiles StateGraph → 8 nodes · 5 conditional edges             │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────────┐
│  LAYER 4 — Domain  (src/domain/)                                            │
│  State · OrchastratorOutput · QueryGenerationOutput · ChatOutput            │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
┌─────────────────┬─────────────────▼──────────────────┬──────────────────────┐
│  LAYER 3a       │  LAYER 3b                          │  LAYER 3c            │
│  Services       │  Retrievers                        │  LLM & Embeddings    │
│  data_ingest    │  pinecone_retriever                │  llm_loader          │
│  conv_service   │  pinecone_client                   │  embedding_loader    │
└────────┬────────┴──────────────────┬─────────────────┴──────────────────────┘
         │                           │
┌────────▼───────────────────────────▼────────────────────────────────────────┐
│  LAYER 2 — Core DI Hub  (src/core/)   ★ Singleton Registry ★               │
│  config · constants · logger · exceptions · memory · dependencies           │
└────────┬───────────────────────────────────────────────────────────────────-┘
         │
┌────────▼────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Infrastructure                                                   │
│  PostgreSQL / Neon DB  (checkpointer + store)                               │
│  Pinecone Vector DB    (per-thread namespaces)                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## LangGraph Workflow

### Node Map

```mermaid
graph TD
    START --> thread_manager_node

    thread_manager_node -->|file_paths present| ingestion_node
    thread_manager_node -->|no file| orchastrator_node

    ingestion_node --> END_ingest["END"]

    orchastrator_node -->|require_db_search=True| query_generation_node
    orchastrator_node -->|messages > threshold| summary_node_A["summary_node"]
    orchastrator_node -->|else| chat_node

    query_generation_node --> retreiver_node

    retreiver_node -->|messages > threshold| summary_node_B["summary_node"]
    retreiver_node -->|else| chat_node

    summary_node_A --> chat_node
    summary_node_B --> chat_node

    chat_node -->|tool_calls present| tool_node
    chat_node -->|no tool| END_chat["END"]

    tool_node --> chat_node
```

### Node Descriptions

| Node | File | What it does |
|---|---|---|
| `thread_manager_node` | `advance_nodes.py` | Entry gate — pass-through (thread eviction logic stub) |
| `ingestion_node` | `main_nodes.py` | Loads files via Docling, chunks, upserts into Pinecone namespace |
| `orchastrator_node` | `main_nodes.py` | Checks Pinecone vector count; LLM decides if retrieval needed |
| `query_generation_node` | `main_nodes.py` | LLM expands user query into multiple search queries |
| `retreiver_node` | `main_nodes.py` | Parallel Pinecone similarity search; applies `@filename` filter |
| `summary_node` | `advance_nodes.py` | Summarizes old messages → injects SystemMessage summary |
| `chat_node` | `main_nodes.py` | Groq LLM generates answer; extracts long-term memory; may call tools |
| `tool_node` | `builder.py` (prebuilt) | Executes `solver(expression)` math tool |

### Conditional Routing

```
route_entry(state):
    file_paths → ingestion_node
    else       → orchastrator_node

route_after_orchastrator(state):
    require_db_search=True    → query_generation_node
    messages > 10             → summary_node
    else                      → chat_node

route_summary_node(state):
    messages > 10             → summary_node
    else                      → chat_node

tools_condition (prebuilt):
    tool_calls present        → tool_node
    else                      → END
```

---

## End-to-End Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant AuthMiddleware
    participant FastAPI
    participant Pipeline
    participant LangGraph
    participant Docling
    participant Pinecone
    participant GroqLLM
    participant NeonDB

    Client->>AuthMiddleware: POST /api/v1/graph/chat (user_id, thread_id)
    AuthMiddleware->>FastAPI: request.state.user_id, thread_id set
    FastAPI->>Pipeline: pipeline.initiate(user_id, thread_id, message)
    Pipeline->>LangGraph: astream_events(State, config)

    LangGraph->>NeonDB: Load checkpoint (conversation history)

    alt File Upload (POST /ingest)
        LangGraph->>Docling: loader.load() in run_in_executor
        Docling-->>LangGraph: documents (text + tables + OCR)
        LangGraph->>Pinecone: upsert chunks with filename metadata
        LangGraph-->>Client: {"success": true}
    else Chat Request (POST /chat)
        LangGraph->>GroqLLM: Orchestrator — require_db_search?
        alt Retrieval needed
            LangGraph->>GroqLLM: generate N search queries
            LangGraph->>Pinecone: asyncio.gather(*N similarity searches)
            Pinecone-->>LangGraph: top-K relevant chunks
        end
        LangGraph->>GroqLLM: Chat with context + memories
        GroqLLM-->>LangGraph: streaming tokens + optional tool_calls
        alt Tool call
            LangGraph->>LangGraph: solver(expression) → result
            LangGraph->>GroqLLM: tool result → final answer
        end
        LangGraph->>NeonDB: Save checkpoint + long-term memory
        LangGraph-->>Client: SSE token stream
    end
```

---

## Directory Layout

```
.
├── main.py                              Uvicorn entry point
├── Dockerfile                           Docker config
├── pyproject.toml                       uv dependencies (Python 3.14+)
├── graph_visualization.png              Auto-generated LangGraph PNG
│
├── api/
│   ├── main.py                          FastAPI app + lifespan warmup + router mounts
│   ├── routes/
│   │   ├── graph_routes.py              POST /ingest · POST /chat
│   │   └── user_routes.py               GET|DELETE /conversation · /long_term_memory · /pine_cone
│   ├── middlewares/
│   │   ├── authentication_middleware.py  Validates user_id + thread_id (params or headers)
│   │   └── multi_middleware.py           Saves List[UploadFile] to uploads/{thread_id}/
│   └── schemas/
│       └── chat_schema.py               ChatRequest Pydantic model
│
├── src/
│   ├── core/
│   │   ├── config.py                    AppConfig (Pydantic BaseSettings → .env)
│   │   ├── constants.py                 All system constants (see table below)
│   │   ├── logger.py                    RotatingFileHandler + StreamHandler
│   │   ├── exceptions.py               MyException — captures file + line number
│   │   ├── memory.py                    AsyncPostgresSaver + AsyncPostgresStore (Neon)
│   │   └── dependencies.py             ★ Singleton DI Hub + warmup_dependencies()
│   │
│   ├── domain/
│   │   ├── state.py                     State TypedDict · OrchastratorOutput · ChatOutput
│   │   ├── config_entities.py           RetrieverConfig · DataIngestionConfig (dataclasses)
│   │   ├── artifacts.py                 DataIngestionArtifact
│   │   └── enums.py                     Pipeline ABC (abstract base for pipelines)
│   │
│   ├── nodes/
│   │   ├── main_nodes.py                ingestion · orchestrator · query_gen · retriever · chat
│   │   ├── advance_nodes.py             summarizer · thread_manager · _cleanup_evicted_thread
│   │   └── conditional_nodes.py         route_entry · route_after_orchastrator · route_summary_node
│   │
│   ├── graphs/
│   │   └── builder.py                   StateGraph compiler → get_graph() @lru_cache
│   │
│   ├── pipelines/
│   │   └── graph_runner_pipeline.py     GraphRunnerPipeline.initiate() → astream_events
│   │
│   ├── services/
│   │   ├── data_ingestion_service.py    DataIngestion: load → inject metadata → chunk → upsert
│   │   └── conversation_service.py      load_conversation · delete · long_term_memory CRUD
│   │
│   ├── retrievers/
│   │   ├── pinecone_client.py           Pinecone client singleton
│   │   └── pinecone_retriever.py        create_retriever · add_documents · get_similar_documents · delete_namespace
│   │
│   ├── llm/
│   │   └── llm_loader.py               get_llm() → ChatGroq singleton
│   │
│   ├── embeddings/
│   │   └── embedding_loader.py         get_embeddings() → HuggingFaceEmbeddings singleton
│   │
│   ├── prompts/
│   │   └── templates.py                ORCHESTRATOR_PROMPT · QUERY_GENERATION_PROMPT · CHAT_PROMPT · SUMMARIZER_PROMPT
│   │
│   ├── tools/
│   │   └── solver_tool.py              @tool solver(expression) using numexpr
│   │
│   └── db/
│       └── thread_manager.py           ThreadManager (SQLite stub — deprecated)
│
├── data/
│   └── app.db                           SQLite (legacy — Neon Postgres is primary)
├── logs/                                Rotating log files
└── uploads/                             Temp file storage (auto-cleaned per request)
```

---

## System Constants

| Constant | Value | Purpose |
|---|---|---|
| `DEFAULT_INDEX_NAME` | `notebooklm-index` | Pinecone index name |
| `EMBEDDING_DIM` | `384` | MiniLM vector dimension |
| `METRIC` | `cosine` | Pinecone similarity metric |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `RETRIEVER_TOP_K` | `10` | Top-K docs per similarity search |
| `LLM_MODEL_NAME` | `llama-3.3-70b-versatile` | Groq model |
| `CHUNK_SIZE` | `800` | Characters per text chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `NO_OF_LAST_MESSAGES_TO_KEEP` | `10` | Messages before summarization triggers |
| `LENGTH_OF_SUMMARY_GENERATED` | `150` | Words in generated summary |
| `MINIMUM_LENGTH_OF_LONG_TERM_MEMORY` | `10` | Min chars to save a memory item |
| `MAXIMUM_CONNECTION_POOL_SIZE` | `20` | Neon DB async pool size |
| `CONTENT_TTL_MINUTES` | `60` | Thread content TTL |
| `PUBLIC_TEMP_DIR` | `uploads/` | Temp file upload directory |

---

## API Reference

> **Auth:** Every endpoint requires `user_id` + `thread_id` as query params `?user_id=...&thread_id=...` or headers `x-user-id` / `x-thread-id`.

### RAG Pipeline (`/api/v1/graph`)

#### `POST /ingest` — Upload Documents

```
Content-Type: multipart/form-data
Field name: files (repeat for multiple files)
```

**Response:**
```json
{ "success": true, "message": "Data ingested successfully", "data": null }
```

**What happens:**
1. Files saved to `uploads/{thread_id}/`
2. Docling parses each file (text + tables + OCR) in thread-pool executor
3. Each chunk gets `metadata["filename"] = "{thread_id}_{filename}"`
4. Chunks embedded and upserted into Pinecone namespace `thread_id`
5. Temp files auto-deleted

---

#### `POST /chat` — Chat (SSE Stream)

```json
{ "message": "Explain the revenue table from @report.pdf" }
```

**Response:** `Content-Type: text/event-stream` — Industry-standard structured SSE JSON stream:

```text
data: {"type": "tool_start", "tool": "solver", "input": {"expression": "25 * 4"}}
data: {"type": "tool_end", "tool": "solver", "output": "100.0"}
data: {"type": "token", "content": " The result is 100."}
data: {"type": "final", "ai_response": "The result is 100.", "state": "completed"}
```

**`@filename` filter:** Mentioning `@report.pdf` restricts Pinecone retrieval to only that file's chunks.

---

### User & Conversation (`/api/v1/user`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/conversation` | Full conversation history (all LangChain message fields) |
| `DELETE` | `/conversation` | Delete LangGraph checkpoint for this thread |
| `DELETE` | `/pine_cone` | Delete Pinecone namespace for this thread (returns 200 on success, 404 if namespace not found) |
| `GET` | `/long_term_memory/` | All persisted user memory key-value pairs |
| `POST` | `/long_term_memory` | Manually add/update a memory key |
| `DELETE` | `/long_term_memory/{key}` | Delete a single memory key |
| `DELETE` | `/long_term_memory` | Delete ALL memory for this user |

**Conversation response shape:**
```json
{
  "success": true,
  "message": "retrieved data successfully",
  "data": [
    {
      "type": "human",
      "data": { "content": "Explain the revenue table", "id": "abc" }
    },
    {
      "type": "ai",
      "data": {
        "content": "The revenue table shows...",
        "tool_calls": [],
        "additional_kwargs": { "reasoning_content": "..." },
        "usage_metadata": { "input_tokens": 120, "output_tokens": 45 },
        "response_metadata": { "model": "llama-3.3-70b-versatile" }
      }
    }
  ]
}
```

---

## Performance & Latency Optimizations

### 1. Parallel Pinecone Queries

```
Before:  Q1 → Pinecone → Q2 → Pinecone → Q3 → Pinecone   = T1 + T2 + T3
After:   Q1 → Pinecone ┐
         Q2 → Pinecone ├─ asyncio.gather()                 = max(T1, T2, T3)
         Q3 → Pinecone ┘
```

### 2. Non-Blocking Docling (Thread Pool)

```python
# Docling is CPU-heavy + synchronous — would block event loop
docs = await loop.run_in_executor(None, loader.load)  # ✅ runs in thread pool
```

### 3. Parallel Multi-File Ingestion

Multiple uploaded files are all parsed simultaneously via `asyncio.gather()`.

### 4. Startup Singleton Warmup

All heavy singletons (HuggingFace model weights, Pinecone client, LLM connection, Neon DB pool) are pre-loaded during `lifespan` before the first request arrives — zero cold-start penalty per request.

### 5. `@lru_cache` Singletons

`get_llm()`, `get_embeddings()`, `get_pinecone_client()`, `get_graph()`, `get_graph_runner_pipeline()` — all cached. One initialization, reused forever.

---

## Developer Guide

### Layered Architecture Rules

```
api/routes/          ← THIN — only extract params, call services/pipelines
src/services/        ← business logic
src/nodes/           ← pure LangGraph node functions (no HTTP knowledge)
src/domain/          ← data structures only (no HTTP, no DB imports)
src/core/            ← infrastructure singletons (imported by everyone)
```

### Adding a New Node

1. Write the async function in `src/nodes/main_nodes.py` or `advance_nodes.py`
2. Register in `src/graphs/builder.py` with `workflow.add_node(...)`
3. Add edges / conditional edges
4. Update `State` in `src/domain/state.py` if new state fields needed

### Adding a New API Endpoint

1. Add route function to `api/routes/graph_routes.py` or `user_routes.py`
2. Add business logic to `src/services/`
3. Add `summary=`, `responses={...}` with `example` for Swagger docs

### `@filename` Filter — How It Works

```
Upload:   chunk.metadata["filename"] = "{thread_id}_{original.pdf}"
                                        stored in Pinecone ↑

Chat:     user types "@original.pdf"
          regex extracts "original.pdf"
          filter = ["{thread_id}_original.pdf"]
          Pinecone filter: {"filename": {"$in": ["{thread_id}_original.pdf"]}}
```

### Serialization

- **Messages → API:** `messages_to_dict(raw_messages)` — preserves ALL fields
- **Metadata → Pinecone:** `_sanitize_metadata()` — flattens complex types to JSON strings
- **Never use:** `dumps()` for API responses (adds LangChain internals)

---

## Running Locally

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Fill in your keys (see Environment Variables section)

# 3. Start the server
uv run main.py

# 4. Open Swagger docs
open http://localhost:8000/docs
```

---

## Environment Variables

```env
# Required
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
POSTGRES_SQL_URL=postgresql+psycopg://user:pass@host/db

# Optional — LangSmith monitoring
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=qri-rag

# Optional
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

---

## Docker

```bash
docker build -t qri .
docker run -p 8000:8000 --env-file .env qri
```

---

## Universal Cost Calculator & Formula Model

Use this dynamic formula model to recalculate system operational costs whenever constants in `src/core/constants.py`, model pricing, user count, or limits change.

### Constants to Token Formulas (Node Breakdown)

In QRI, each user question executes up to 3 LLM nodes (`orchestrator_node`, `query_generation_node`, and `chat_node`). All token calculations are derived directly from the parameters in `src/core/constants.py`:

1. **Total Input Tokens ($T_{in}$)**:
   $$T_{in} = \text{SYSTEM\_PROMPTS\_BASE} + \left( \text{RETRIEVER\_TOP\_K} \times \frac{\text{CHUNK\_SIZE}}{4} \right) + (\text{NO\_OF\_LAST\_MESSAGES\_TO\_KEEP} \times 100) + (\text{LENGTH\_OF\_SUMMARY\_GENERATED} \times 1.3)$$

   - **Formula Terms & Token Derivations**:
     - $\frac{\text{CHUNK\_SIZE}}{4} = \frac{800}{4} = \mathbf{200 \text{ tokens/chunk}}$ *(Rule: 1 token $\approx$ 4 characters in English NLP tokenization)*.
     - $\text{NO\_OF\_LAST\_MESSAGES\_TO\_KEEP} \times 100$: **100 tokens/message** (Upper-bound estimation: User question ~35 tokens + AI response ~90 tokens = Avg ~60-100 tokens per history message turn).
     - $\text{LENGTH\_OF\_SUMMARY\_GENERATED} \times 1.3 = 50 \times 1.3 = \mathbf{65 \text{ tokens}}$ *(Rule: 1 word $\approx$ 1.3 tokens in Llama-3 BPE tokenizer)*.
     - $\text{SYSTEM\_PROMPTS\_BASE} = \mathbf{500 \text{ tokens}}$ *(Fixed overhead for system instructions in Orchestrator, QueryGen, and Chat prompts)*.

   - **Current Values Calculation** ($\text{RETRIEVER\_TOP\_K}=4, \text{CHUNK\_SIZE}=800, \text{NO\_OF\_LAST\_MESSAGES\_TO\_KEEP}=4, \text{LENGTH\_OF\_SUMMARY\_GENERATED}=50$):
     $$T_{in} = 500 + \left(4 \times \frac{800}{4}\right) + (4 \times 100) + (50 \times 1.3) = 500 + 800 + 400 + 65 = \mathbf{1,765 \text{ Tokens}}$$

2. **Total Output Tokens ($T_{out}$)**:
   $$T_{out} = \text{OTHER\_NODES\_OUTPUT\_BASE} + (\text{LLM\_OUTPUT\_MAX\_WORDS} \times 1.35)$$
   - *Current Values Calculation* ($\text{OTHER\_NODES\_OUTPUT\_BASE}=50, \text{LLM\_OUTPUT\_MAX\_WORDS}=100$):
     $$T_{out} = 50 + (100 \times 1.35) = 50 + 135 = \mathbf{185 \text{ Tokens}}$$

---

### Variables & Inputs

| Symbol | Meaning | Default Value |
|---|---|---|
| **$U$** | Total Paid Users | 10,000 |
| **$Q_{user}$** | Monthly Question Limit / Usage per User | 150 (Max Limit) / 75 (Average Usage) |
| **$P_{plan}$** | Monthly Plan Price (INR) | ₹9 |
| **$T_{in}$** | Total Input Tokens per Question (All Nodes) | 1,765 |
| **$T_{out}$** | Total Output Tokens per Question (All Nodes) | 185 |
| **$R_{in}$** | Groq Input Rate per 1M Tokens (USD) | $0.59 |
| **$R_{out}$** | Groq Output Rate per 1M Tokens (USD) | $0.79 |
| **$FX$** | USD to INR Exchange Rate | 84.0 |
| **$C_{infra}$** | Fixed Monthly Infra (Server + DB + Pinecone + Logging) in USD | $150 |

---

### Step-by-Step Mathematical Formulas

#### 1. Per-Question AI Cost ($C_{Q}$)
$$C_{Q\_USD} = \left( \frac{T_{in} \times R_{in}}{1,000,000} \right) + \left( \frac{T_{out} \times R_{out}}{1,000,000} \right)$$
$$C_{Q\_INR} = C_{Q\_USD} \times FX$$

#### 2. Total Monthly Questions ($Q_{total}$)
$$Q_{total} = U \times Q_{user}$$

#### 3. Total Monthly AI/LLM Cost ($C_{AI\_total}$)
$$C_{AI\_total} = Q_{total} \times C_{Q\_INR}$$

#### 4. Total Monthly Infrastructure Cost ($C_{Infra\_total}$)
$$C_{Infra\_total} = C_{infra} \times FX$$

#### 5. Total Technical Cost ($C_{Total}$)
$$C_{Total} = C_{AI\_total} + C_{Infra\_total}$$

#### 6. Technical Cost Per Paid User ($C_{user}$)
$$C_{user} = \frac{C_{Total}}{U}$$

#### 7. Total Revenue & Net Technical Margin
$$\text{Revenue} = U \times P_{plan}$$
$$\text{Net Profit} = \text{Revenue} - C_{Total}$$

### Step-by-Step Worked Example & Tabular Matrix (₹9 Plan @ 10,000 Paid Users)

#### Inputs
- $U = 10,000$ users | $P_{plan} = \text{₹9/month}$
- $T_{in} = 1,765$ tokens | $T_{out} = 185$ tokens
- $R_{in} = \$0.59 / 1\text{M}$ | $R_{out} = \$0.79 / 1\text{M}$
- $FX = 84.0$ | $C_{infra} = \$150 / \text{month}$

#### Line-by-Line Execution

1. **Per-Question AI Cost ($C_Q$)**:
   $$C_{Q\_USD} = \left( \frac{1,765 \times 0.59}{1,000,000} \right) + \left( \frac{185 \times 0.79}{1,000,000} \right) = 0.00104135 + 0.00014615 = \mathbf{\$0.0011875}$$
   $$C_{Q\_INR} = \$0.0011875 \times 84.0 = \mathbf{\text{₹0.09975 / question (9.97 Paise)}}$$

2. **Total Monthly Questions ($Q_{total}$)**:
   - Scenario A (Low - 35 Qs/mo): $Q_{total} = 10,000 \times 35 = \mathbf{350,000}$
   - Scenario B (Avg - 75 Qs/mo): $Q_{total} = 10,000 \times 75 = \mathbf{750,000}$
   - Scenario C (Max - 150 Qs/mo): $Q_{total} = 10,000 \times 150 = \mathbf{1,500,000}$

3. **Total Monthly AI Cost ($C_{AI\_total}$)**:
   - Scenario A: $350,000 \times \text{₹0.09975} = \mathbf{\text{₹34,912.50}}$
   - Scenario B: $750,000 \times \text{₹0.09975} = \mathbf{\text{₹74,812.50}}$
   - Scenario C: $1,500,000 \times \text{₹0.09975} = \mathbf{\text{₹149,625.00}}$

4. **Total Monthly Infra Cost ($C_{Infra\_total}$)**:
   $$C_{Infra\_total} = \$150 \times 84.0 = \mathbf{\text{₹12,600.00 / month}}$$

5. **Total Technical Cost ($C_{Total}$)**:
   - Scenario A: $\text{₹34,912.50} + \text{₹12,600.00} = \mathbf{\text{₹47,512.50}}$
   - Scenario B: $\text{₹74,812.50} + \text{₹12,600.00} = \mathbf{\text{₹87,412.50}}$
   - Scenario C: $\text{₹149,625.00} + \text{₹12,600.00} = \mathbf{\text{₹162,225.00}}$

6. **Technical Cost Per Paid User ($C_{user}$)**:
   - Scenario A: $\frac{\text{₹47,512.50}}{10,000} = \mathbf{\text{₹4.75 / user / month}}$
   - Scenario B: $\frac{\text{₹87,412.50}}{10,000} = \mathbf{\text{₹8.74 / user / month}}$
   - Scenario C: $\frac{\text{₹162,225.00}}{10,000} = \mathbf{\text{₹16.22 / user / month}}$

7. **Total Revenue & Net Margin (Revenue = ₹90,000.00)**:
   - Scenario A: $\text{₹90,000} - \text{₹47,512.50} = \mathbf{+\text{₹42,487.50 Profit (47.2\% Margin)}}$
   - Scenario B: $\text{₹90,000} - \text{₹87,412.50} = \mathbf{+\text{₹2,587.50 Profit (2.9\% Margin)}}$
   - Scenario C: $\text{₹90,000} - \text{₹162,225.00} = \mathbf{-\text{₹72,225.00 Loss}}$

---

#### 📊 Tabular Summary Matrix

| Metric | Scenario A (Low - 35 Qs) | Scenario B (Avg - 75 Qs) | Scenario C (Max - 150 Qs) |
|---|---|---|---|
| **Total Monthly Questions** | 350,000 | 750,000 | 1,500,000 |
| **AI LLM Cost (Groq)** | ₹34,912.50 | ₹74,812.50 | ₹149,625.00 |
| **Infra Cost (DB+Server)** | ₹12,600.00 | ₹12,600.00 | ₹12,600.00 |
| **Total Technical Cost** | **₹47,512.50** | **₹87,412.50** | **₹162,225.00** |
| **Cost Per Paid User** | **₹4.75** | **₹8.74** | **₹16.22** |
| **Total Plan Revenue** | ₹90,000.00 | ₹90,000.00 | ₹90,000.00 |
| **Net Monthly Margin** | **+₹42,487.50** (Profit) | **+₹2,587.50** (Profit) | **-₹72,225.00** (Loss) |

---

### Excel / Google Sheets Formula Setup

```text
Cell A1 (Users): 10000
Cell A2 (Qs per user): 75
Cell A3 (Plan Price): 9
Cell A4 (Input Tokens): 1765
Cell A5 (Output Tokens): 185
Cell A6 (Input Rate): 0.59
Cell A7 (Output Rate): 0.79
Cell A8 (USD to INR): 84
Cell A9 (Infra Cost USD): 150

Formulas:
- Cost Per Question INR = ((A4*A6/1000000)+(A5*A7/1000000))*A8
- Total AI Cost INR = A1 * A2 * Cost_Per_Question_INR
- Total Infra Cost INR = A9 * A8
- Total Technical Cost INR = Total_AI_Cost_INR + Total_Infra_Cost_INR
- Cost Per User INR = Total_Technical_Cost_INR / A1
- Net Monthly Profit = (A1 * A3) - Total_Technical_Cost_INR
```

---

*Built by VashuTheGreat*
