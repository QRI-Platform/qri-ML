# 📘 Pyrhon_Backend — Important Architecture & Developer Guide (`README_imp.md`)

Welcome to the **Pyrhon_Backend** codebase! This document serves as the master guide for any developer working on or joining this project. It explains the exact purpose of every single directory, file naming convention, architectural layer, and developer rule.

---

## 🏗️ Architectural Layer Overview

The codebase is structured according to the **Layered Clean Architecture**:

```
                       Client / HTTP Request
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Presentation / API Layer (`api/`)                                  │
│    Routes, Middlewares, API Request/Response Schemas (DTOs)            │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Pipeline & Service Layer (`src/pipelines/`, `src/services/`)         │
│    Execution Pipelines, Data Ingestion Service, Conversation Service   │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Workflow & Nodes (`src/graphs/`, `src/nodes/`)                      │
│    LangGraph Workflow Compiler, Pure Stateless Graph Node Functions    │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Domain Layer (`src/domain/`)                                       │
│    LangGraph State, Domain Entities, Dataclasses, Enums & Interfaces   │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. Core & Infrastructure (`src/core/`, `src/db/`, `src/retrievers/`)   │
│    Config, Logger, Exceptions, Singletons Hub, Vector Store, SQLite    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Folder-by-Folder & File-by-File Breakdown

### 1. `api/` — API Transport & Presentation Layer
This folder contains everything related to HTTP web communication via FastAPI.

- **`api/routes/`**: Contains FastAPI APIRouters (`graph_routes.py`, `user_routes.py`).
  - **Purpose**: Maps HTTP endpoints (`POST /api/v1/graph/chat`, `GET /api/v1/user/conversation`, `DELETE /api/v1/user/pine_cone`) to Python async functions.
  - **Rule**: Routers must stay *thin*. They extract headers/state and call `src/services/` or `src/pipelines/`.
- **`api/middlewares/`**: Request interceptors.
  - `authentication_middleware.py`: Extracts and validates `user_id` and `thread_id` from request headers or query params.
  - `multi_middleware.py`: Handles multipart file uploads for PDF/document ingestion.
- **`api/schemas/`**: Pydantic data validation models for API HTTP Request and Response payloads (Data Transfer Objects / DTOs).
  - `chat_schema.py`: Validates input payload for `/chat` (`{"message": "..."}`).
  - `user_schema.py`: Validates user-related request bodies.
- **`api/main.py`**: The FastAPI application entrypoint. Configures lifespan singleton warmups and mounts router routes.

---

### 2. `src/core/` — Foundation, Singletons & Configurations
This directory contains system-wide utilities and singleton factories shared across all layers.

- **`src/core/config.py`**: Reads `.env` environment variables using Pydantic `AppConfig`.
- **`src/core/constants.py`**: Global constants and default configuration values (e.g. `DEFAULT_INDEX_NAME`, `EMBEDDING_DIM`, `CHUNK_SIZE`, `NO_OF_LAST_MESSAGES_TO_KEEP`).
- **`src/core/logger.py`**: Centralized rotating file logger & colored console logger.
- **`src/core/exceptions.py`**: Custom exception wrapper (`MyException`) capturing detailed line numbers and tracebacks.
- **`src/core/memory.py`**: Singletons for LangGraph `MemorySaver` checkpointer and `InMemoryStore`.
- **`src/core/dependencies.py`**: Central Dependency Injection (DI) hub with `warmup_dependencies()` to eagerly initialize system singletons during startup.

---

### 3. `src/domain/` — Business Entities & Domain Models
The `domain` layer defines the pure data structures, state schemas, and interfaces of our AI application. It has zero dependencies on HTTP or FastAPI.

- **`src/domain/state.py`**: Contains the `State` class — the single source of truth flowing through every node in the LangGraph graph, plus structured output schemas (`OrchastratorOutput`, `QueryGenerationOutput`, `ChatOutput`).
- **`src/domain/config_entities.py`**: Internal business dataclasses (`RetrieverConfig`, `DataIngestionConfig`).
- **`src/domain/artifacts.py`**: Service result containers (`DataIngestionArtifact`).
- **`src/domain/enums.py`**:
  - **What does "enums" mean?**: "Enums" stands for *Enumerations / Abstract Types / Base Interfaces*. In computer science, an Enum or Interface defines a fixed set of constants, statuses, or abstract contracts.
  - **What content belongs here?**: Abstract base classes (such as `Pipeline(ABC)`), domain status enums (e.g. `ThreadStatus`, `TaskType`), and shared contracts.

---

### 4. `src/services/` — Core Business Logic Layer
The Service Layer encapsulates all business operations.

- **`src/services/data_ingestion_service.py`**: `DataIngestion` class — handles document loading (`DoclingLoader`), text chunking (`RecursiveCharacterTextSplitter`), and embedding storage in Pinecone.
- **`src/services/conversation_service.py`**: Business functions to load conversation history (`load_conversation`), delete Pinecone namespaces (`delete_pinecone_namespace`), and query user long-term memories.

---

### 5. `src/nodes/` — Pure LangGraph Node Functions
Contains pure, stateless node execution functions for the LangGraph workflow.

- **`src/nodes/main_nodes.py`**: Core workflow nodes: `ingestion_node`, `orchastrator_node`, `query_generation_node`, `retreiver_node`, `chat_node`, `summary_node`.
- **`src/nodes/advance_nodes.py`**: Advanced nodes: `summerizer`, `thread_manager_node`, `_cleanup_evicted_thread`.
- **`src/nodes/conditional_nodes.py`**: Edge routing functions (`route_entry`, `route_after_orchastrator`, `route_summary_node`) that decide the next node path.

---

### 6. `src/graphs/` — LangGraph Workflow Compiler
Assembles and compiles LangGraph workflows.

- **`src/graphs/builder.py`**: Instantiates `StateGraph(State)`, connects nodes with edges and conditional routes, attaches checkpointer/store singletons, and exports compiled `get_graph()`.
- **`src/graphs/subgraphs/` (Future Expansion)**: Dedicated folder for sub-workflows (e.g., `rag_subgraph.py`, `search_subgraph.py`).

---

### 7. `src/db/` — Database Repositories
Database persistence layer.

- **`src/db/thread_manager.py`**: `ThreadManager` class — manages SQLite operations (`data/app.db`) for tracking user active threads, enforcing thread limits, and handling TTL evictions.

---

### 8. `src/pipelines/` — Streaming & Execution Orchestration
Bridge between HTTP API routes and the compiled LangGraph workflow.

- **`src/pipelines/graph_runner_pipeline.py`**: `GraphRunnerPipeline` class — streams real-time graph events (`astream_events`) to HTTP event streams.

---

### 9. `src/retrievers/` — Vector Store Connectors
- **`pinecone_client.py`**: Pinecone client singleton factory.
- **`pinecone_retriever.py`**: Vector store initialization, document upserts, similarity search, and namespace deletion.

---

### 10. `src/llm/` & `src/embeddings/` — Model Loaders
- **`src/llm/llm_loader.py`**: `get_llm()` singleton for ChatGroq.
- **`src/embeddings/embedding_loader.py`**: `get_embeddings()` singleton for HuggingFaceEmbeddings.

---

### 11. `src/prompts/` & `src/tools/` — Prompts & Agent Tools
- **`src/prompts/templates.py`**: All system prompt templates (Orchestrator, Query Generator, Summarizer, Chat Prompt).
- **`src/tools/web_search.py`**: Math solver and external search tools for LLM tool calling.

---

## 📜 Developer Rules & Coding Standards

1. **Top-Level Imports Only (PEP 8)**: All `import` statements MUST be located at the top of the file.
2. **Thin API Controllers**: `api/routes/` must never contain raw DB queries or heavy LLM logic. Always delegate to `src/services/` or `src/pipelines/`.
3. **No Package Circular Imports**: Package `__init__.py` files must remain clean to prevent circular dependency triggers during module initialization.
4. **Schemas vs Domain Models**:
   - `api/schemas/` ➔ Pydantic models for HTTP Request/Response validation.
   - `src/domain/` ➔ Data structures representing internal domain state.
