# QRI — Multi-Threaded Stateful RAG Architecture

An industrial-grade, stateful Multi-Tenant RAG (Retrieval-Augmented Generation) backend built with **FastAPI**, **LangGraph**, **Pinecone**, **SQLite**, and **Groq LLM**.

---

## Industrial Architecture & Design Patterns

This codebase follows strict Clean Layered Architecture and Dependency Injection (DI) principles for high-performance agentic systems:

1. **Central Dependency Injection Hub (`src/core/dependencies.py`)**:
   - Infrastructure singletons (`AppConfig`, `ChatGroq`, `HuggingFaceEmbeddings`, `Pinecone`, `MemorySaver`, `ThreadManager`, `Retriever`) are managed via `@lru_cache` factories.
   - All modules import singletons exclusively through `src.core.dependencies`, enforcing strict decoupling and single-responsibility principles.

2. **Boot-Time Server Warmup (`warmup_dependencies()`)**:
   - During FastAPI startup (`lifespan`), `warmup_dependencies()` pre-loads heavy singletons (HuggingFace weights, Pinecone client, LLM connection pool, and SQLite database) before accepting HTTP traffic.
   - Eliminates cold-start latency for `/chat` streaming and `/ingest` endpoints.

3. **Multi-Tenant State & Thread Eviction**:
   - User sessions are isolated via `thread_id` namespaces in Pinecone and `MemorySaver` checkpointer memory.
   - SQLite enforces user-thread mapping and thread limits (`MAX_THREADS_PER_USER`). When exceeded, the oldest thread graph state and vector namespace are automatically purged asynchronously.

4. **Zero Code Comments & Clean Code Enforcement**:
   - Python code strictly adheres to clean naming, type annotations, and explicit layered control flow without inline comments.

---

## Architectural Layer Graph

![LangGraph Workflow Visualization](graph_visualization.png)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 7 — API & Presentation (`api/`)                                          │
│                                                                                 │
│   api/main.py                   FastAPI lifespan warmup & router setup          │
│   api/routes/graph_routes.py  /chat, /ingest, /delete HTTP handlers               │
│   api/middlewares/            Auth validator & multi-part file handler          │
│   api/models/chat_model.py    ChatRequest Pydantic schema                       │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼────────────────────────────────────────┐
│  LAYER 5 — Service & Pipeline (`src/pipelines/`)                                │
│                                                                                 │
│   get_pipeline()                @lru_cache factory for GraphRunnerPipeline      │
│   graph_runner_pipeline.py      Execution stream orchestrator (astream_events)  │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼────────────────────────────────────────┐
│  LAYER 4 — Workflow Graph & Nodes (`src/graphs/`, `src/nodes/`, `src/prompt/`)   │
│                                                                                 │
│   src/graphs/builder.py         StateGraph compiler & checkpointer attachment   │
│   src/nodes/main_nodes.py       ingestion, orchestrator, query_gen, retriever   │
│   src/nodes/advance_nodes.py    thread_manager, summarizer, async cleanup       │
│   src/nodes/conditional_nodes.py Pure state routing functions                     │
│   src/prompt/__init__.py        LLM system prompt registry                      │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼────────────────────────────────────────┐
│  LAYER 3 — Domain Components (`src/components/`, `src/retriever/`, `src/llm/`)  │
│                                                                                 │
│   src/llm/llm_loader.py              ChatGroq factory                           │
│   src/memory/__init__.py             MemorySaver checkpoint manager             │
│   src/retreiver/retreiver.py         Pinecone vector CRUD operations             │
│   src/components/data_ingestion.py   DoclingLoader → chunker → vector store     │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼────────────────────────────────────────┐
│  LAYER 2 — Core DI Hub (`src/core/dependencies.py`) ★ Singletons Hub ★          │
│                                                                                 │
│   warmup_dependencies()      Pre-loads singletons at FastAPI boot               │
│   get_app_config()           AppConfig settings                                 │
│   get_llm()                  ChatGroq instance                                  │
│   get_embeddings()           HuggingFaceEmbeddings instance                     │
│   get_pinecone_client()      Pinecone client instance                           │
│   get_checkpointer()         MemorySaver checkpoint instance                    │
│   get_thread_manager()       SQLite ThreadManager instance                      │
│   get_retriever()            Retriever instance factory                         │
└──────────────────┬─────────────────────────────────────┬────────────────────────┘
                   │                                     │
┌──────────────────▼──────────┐       ┌──────────────────▼────────────────────────┐
│  LAYER 6 — Database (`db/`) │       │  LAYER 1 — Entities & State Schemas       │
│                             │       │            (`src/entity/`, `src/models/`) │
│  thread_manager.py          │       │                                           │
│  SQLite CRUD for threads.   │       │  entity/config.py   RetrieverConfig       │
│  Purges checkpointer state  │       │  entity/artifact.py IngestionArtifact     │
│  on eviction or deletion.   │       │  models/workflow.py LangGraph State       │
└─────────────────────────────┘       └───────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────────────────────┐
│  LAYER 0 — Core Utilities (`src/constants/`, `src/logger/`, `src/exception/`)    │
│                                                                                 │
│  constants/__init__.py   System config constants & defaults                     │
│  logger/__init__.py      Rotating file logger ('app')                           │
│  exception/__init__.py   Trace-enhanced MyException error wrapper               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed System Sequence & Data Flow Graph

### End-to-End Chat & Ingestion Request Flow

```mermaid
graph TD
    Client[Client HTTP Request] -->|Auth & Query Params| AuthMiddleware[Authentication Middleware]
    AuthMiddleware -->|Validate user_id & thread_id| MulterMiddleware[File Upload Middleware]
    MulterMiddleware -->|Save file to uploads/thread_id| GraphRoutes[FastAPI Graph Routes]
    
    GraphRoutes -->|Get Singleton| Pipeline[GraphRunnerPipeline]
    Pipeline -->|Initialize State| LangGraph[LangGraph Engine]
    
    subgraph Execution Loop
        LangGraph --> ThreadManagerNode[Thread Manager Node]
        ThreadManagerNode -->|Enforce Limits & Evict| DB[(SQLite App DB)]
        
        ThreadManagerNode --> Router{Route Entry}
        Router -->|File Uploaded| IngestionNode[Ingestion Node]
        Router -->|No File| OrchestratorNode[Orchestrator Node]
        
        IngestionNode -->|Docling Loader & Chunker| RetrieverComp[Retriever Component]
        RetrieverComp -->|Upsert Vectors| VectorDB[(Pinecone Vector DB)]
        
        OrchestratorNode --> DBCheck{Require Search?}
        DBCheck -->|Yes| QueryGenNode[Query Generation Node]
        DBCheck -->|No| ChatNode[Chat Node]
        
        QueryGenNode --> RetrieverNode[Retriever Node]
        RetrieverNode -->|Similarity Search| VectorDB
        RetrieverNode --> ChatNode
        
        ChatNode -->|Groq LLM Stream| EventStream[SSE Response Stream]
    end
    
    EventStream --> Client
```

### Async Thread Eviction & Lifecycle Management Flow

```mermaid
sequenceDiagram
    autonumber
    participant User as User Request
    participant TM as ThreadManager Node
    participant DB as SQLite DB (app.db)
    participant CP as Checkpointer (MemorySaver)
    participant PC as Pinecone Vector Store

    User->>TM: Request (user_id, thread_id)
    TM->>DB: Check active thread count for user
    alt Count >= MAX_THREADS_PER_USER
        DB->>TM: Return oldest thread_id for eviction
        TM->>DB: DELETE FROM user_threads WHERE thread_id=oldest
        TM->>CP: delete_thread(oldest_thread_id)
        TM->>PC: Async Purge Namespace (oldest_thread_id)
    end
    TM->>DB: INSERT user_id, new_thread_id
    TM-->>User: Proceed with workflow execution
```

---

## Directory Layout

```
.
├── main.py                               Uvicorn web server entry point
│
├── api/
│   ├── main.py                           FastAPI application boot & lifespan warmup
│   ├── routes/
│   │   └── graph_routes.py               HTTP routes: /ingest, /chat, /delete
│   ├── middlewares/
│   │   ├── authentication_middleware.py  User & thread ID parameter authentication
│   │   └── multi_middleware.py           Multipart form upload processor
│   ├── helper/
│   │   └── graph_helper.py               Async thread cleanup helper
│   └── models/
│       └── chat_model.py                 ChatRequest Pydantic payload model
│
├── db/
│   ├── __init__.py                       Re-exports ThreadManager
│   └── thread_manager.py                 SQLite thread persistence & eviction logic
│
├── src/
│   ├── constants/__init__.py             System constants and threshold defaults
│   ├── logger/__init__.py                Rotating file logging configuration
│   ├── exception/__init__.py             Custom exception trace decorator
│   │
│   ├── config/
│   │   └── app_config.py                 Pydantic Settings env loader (.env)
│   │
│   ├── entity/
│   │   ├── config.py                     RetrieverConfig & DataIngestionConfig
│   │   └── artifact.py                   DataIngestionArtifact wrapper
│   │
│   ├── models/
│   │   └── workflow_models.py            LangGraph state schema & node output models
│   │
│   ├── core/
│   │   └── dependencies.py              ★ DI Hub: Singleton registry & boot warmup
│   │
│   ├── llm/
│   │   └── llm_loader.py                 ChatGroq singleton factory
│   │
│   ├── memory/
│   │   └── __init__.py                   MemorySaver checkpointer singleton
│   │
│   ├── retreiver/
│   │   ├── pinecone_client.py            Pinecone client singleton
│   │   └── retreiver.py                  Pinecone vector store CRUD manager
│   │
│   ├── components/
│   │   └── data_ingestion.py             Document processing pipeline
│   │
│   ├── prompt/__init__.py                Prompt template registry
│   │
│   ├── nodes/
│   │   ├── conditional_nodes.py          Pure conditional routing logic
│   │   ├── main_nodes.py                 Ingestion, orchestrator, retriever, chat nodes
│   │   └── advance_nodes.py              Thread manager, summarizer & cleanup nodes
│   │
│   ├── graphs/
│   │   └── builder.py                    StateGraph construction & compilation
│   │
│   ├── pipelines/
│   │   ├── __init__.py                   Pipeline factory
│   │   └── graph_runner_pipeline.py      Graph execution pipeline wrapper
│   │
│   └── utils/
│       └── abstract_class.py             Abstract base class contracts
│
└── data/
    └── app.db                            SQLite database storage
```

---

## API Documentation

### 1. File & Data Ingestion
- **URL**: `POST /api/v1/ingest?user_id={user_id}&thread_id={thread_id}`
- **Headers**: `Content-Type: multipart/form-data`
- **Body**: `file` (Binary File Upload)
- **Response**: `{"success": true, "message": "Data ingested successfully"}`

### 2. Conversational Chat (SSE Stream)
- **URL**: `POST /api/v1/chat?user_id={user_id}&thread_id={thread_id}`
- **Headers**: `Content-Type: application/json`
- **Body**: `{"message": "Summarize the uploaded document"}`
- **Response**: Server-Sent Events stream (`text/event-stream`)

### 3. Delete Thread & Clear Storage
- **URL**: `DELETE /api/v1/delete?user_id={user_id}&thread_id={thread_id}`
- **Response**: `{"success": true, "message": "Thread deleted successfully"}`

---

## Running locally

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Configure Environment (`.env`)**:
   ```env
   PINECONE_API_KEY=your_pinecone_key
   GROQ_API_KEY=your_groq_key
   ```

3. **Start the FastAPI Backend**:
   ```bash
   uv run main.py
   ```
