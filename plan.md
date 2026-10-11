# Phase 2 Strategic Implementation Plan: Gamified Learning & High-Throughput Test Generation SubGraph

> **Document Status**: Production Roadmap & Architecture Blueprint  
> **Author**: VashuTheGreat 
> **Start Date**: Saturday, September 26, 2026  
> **Target Completion / Production Launch**: Sunday, November 1, 2026  
> **Reporting Cadence**: Weekly Team Deliverable every Sunday  

---

## 1. Executive Summary & Baseline Analysis

### 1.1 Current Implementation Audit (Baseline)
The current test-generation implementation lives in the following locations within the codebase:
- **API Route**: `api/routes/sub_graph_routes.py` (`GET /test`), which accepts query parameters (`total_no_of_questions`, `level`, `subject_name`, `exam_type`) under authenticated user context.
- **Pipeline Runner**: `src/pipelines/graph_runner_pipeline.py` (`initiate_test_generation`), which wraps parameters into a generic dictionary state with `need_test_paper: True` and invokes the root graph `self.graph.ainvoke(state)`.
- **Graph Assembly**: `src/graphs/builder.py`, where `route_entry` branches at `START` directly into a single `test_generation_node` if `need_test_paper` is detected.
- **Node Execution**: `src/nodes/main_nodes.py` (`test_generation_node`), which loads a Langfuse prompt (`TEST_PAPER_GENERATION_PROMPT`) and attempts to generate all requested questions in a single synchronous Groq LLM call via `.with_structured_output(Questions_generation_schema)`.
- **Domain State**: `src/domain/state.py`, where `need_test_paper`, `test_paper`, `total_no_of_questions`, `level`, `subject_name`, and `exam_type` are appended directly to the general chat conversation `State(TypedDict)`.

### 1.2 Limitations of the Current Implementation
1. **Monolithic Prompt Generation Bottleneck**:
   - Calling LLM for >5–10 questions in a single prompt causes output token exhaustion (Groq `max_tokens: 8192`), severe schema truncation, formatting drift, and latency degradation (15–30s blocking).
   - High risk of question repetition, lack of sub-topic diversity, and zero verification of distractors/options.
2. **Coupling with Root Chat Graph**:
   - Test generation is currently an artificial branch on the RAG conversation graph. It shares checkpointer checkpoints, memory stores, and conversational state overhead instead of operating as an isolated, reusable **LangGraph SubGraph**.
3. **No Parallel Batching Execution**:
   - No map-reduce mechanism to generate batches of questions concurrently (e.g., generating 30 questions as 3 parallel tasks of strictly 10 questions each).
4. **Complete Absence of the Test-Taking & Post-Test Lifecycle**:
   - No data models or routes for taking tests, answering questions, or recording user telemetry (time per question, confidence level).
   - No grading engine, negative marking calculation (e.g., IIT-JEE +4/-1 scheme), or accuracy analytics.
5. **No Gamified Learning System**:
   - No XP calculations, no streak tracking, no badges/achievements, no cognitive diagnostic feedback (Bloom’s taxonomy), and no adaptive weak-area remedial test recommendations.

### 1.3 Phase 2 Vision
Phase 2 transforms the test-generation prototype into an industrial-grade **Gamified Learning & Intelligent Assessment Engine**:
1. **Dedicated LangGraph SubGraphs**:
   - `TestPaperGenerationSubGraph`: Deconstructs test blueprints into parallel 10-question micro-batches via LangGraph `Send` API, validates outputs, deduplicates questions, and builds comprehensive test papers.
   - `TestEvaluationAndGamificationSubGraph`: Processes user submissions, executes deterministic scoring and LLM cognitive diagnosis, calculates XP and streak progress, updates skill radar charts, and recommends adaptive remedial micro-tests.
2. **Strict Modular State Contracts**:
   - Dedicated `InputState`, `InternalState`, and `OutputState` for each Subgraph, completely isolated from conversation chat state.
3. **Gamification Loops**:
   - XP, Level Tiers, Daily Streaks with streak-freeze mechanics, Badges (e.g., *Speed Demon*, *Accuracy Sniper*, *Concept Master*), and Weak-Area Diagnostic Remediation.

---

## 2. Target System Architecture

```
                       ┌────────────────────────────────────────────────────────┐
                       │               CLIENT / FRONTEND APPLICATION            │
                       └──────────┬─────────────────────────────────┬───────────┘
                                  │                                 │
           1. Generate Test Paper │                                 │ 2. Submit Answers &
             (Target, Level, N)   │                                 │    Telemetry
                                  ▼                                 ▼
         ┌───────────────────────────────────┐    ┌───────────────────────────────────┐
         │ POST /api/v1/subgraph/test/generate│    │  POST /api/v1/subgraph/test/submit│
         └─────────────────┬─────────────────┘    └─────────────────┬─────────────────┘
                           │                                        │
                           ▼                                        ▼
    ╔════════════════════════════════════════╗   ╔════════════════════════════════════════╗
    ║   TEST GENERATION SUBGRAPH             ║   ║   TEST EVALUATION & GAMIFICATION       ║
    ║   (TestPaperGenerationSubGraph)        ║   ║   SUBGRAPH                             ║
    ╠════════════════════════════════════════╣   ╠════════════════════════════════════════╣
    ║                                        ║   ║                                        ║
    ║  1. Blueprint & Batch Partitioner      ║   ║  1. Test Session & Answer Loader       ║
    ║     - Splits N into <=10 Question      ║   ║                                        ║
    ║       chunks with topic/difficulty     ║   ║  2. Deterministic Scoring Node         ║
    ║       allocation                       ║   ║     - Answer key match, negative       ║
    ║                                        ║   ║       marking, timing efficiency       ║
    ║  2. Parallel Generation Workers        ║   ║                                        ║
    ║     (LangGraph Send API / Workers)     ║   ║  3. Cognitive Diagnostic Analyser      ║
    ║     - Batch 1 (10 Qs) [Worker 1]       ║   ║     - LLM error diagnosis (concept vs  ║
    ║     - Batch 2 (10 Qs) [Worker 2]       ║   ║       calculation vs time pressure)    ║
    ║     - Batch 3 (10 Qs) [Worker 3]       ║   ║                                        ║
    ║                                        ║   ║  4. Gamification Engine Node           ║
    ║  3. Question Aggregator & Guardrails   ║   ║     - XP calculation & Level Tiers     ║
    ║     - Deduplication (Semantic Cosine)  ║   ║     - Daily streak update & freeze     ║
    ║     - Schema validation (4 options,    ║   ║     - Achievement Badge triggers       ║
    ║       valid key, pedagogical feedback) ║   ║                                        ║
    ║                                        ║   ║  5. Adaptive Remedial Generator        ║
    ║  4. Formatter & Test Paper Exporter    ║   ║     - Synthesizes weak-area remedial   ║
    ║     - Generates test_id & metadata     ║   ║       study guide & 5-question drill   ║
    ╚════════════════════════════════════════╝   ╚════════════════════════════════════════╝
                           │                                        │
                           ▼                                        ▼
         ┌───────────────────────────────────┐    ┌───────────────────────────────────┐
         │       NEON POSTGRES STORE         │    │       NEON POSTGRES STORE         │
         │ - test_papers                     │    │ - test_attempts                   │
         │ - questions_bank                  │    │ - user_gamification_profiles      │
         │ - topic_taxonomies                │    │ - user_badges & xp_ledger         │
         └───────────────────────────────────┘    └───────────────────────────────────┘
```

---

## 3. Modular State Contracts Specification

To ensure independent extensibility and clean boundaries, Phase 2 implements dedicated State contracts inside `src/domain/subgraph_states.py`:

### 3.1 Test Generation State Contracts
```python
# 1. Input Contract: Public API payload translated to SubGraph entry
class TestGenInputState(TypedDict):
    user_id: str
    subject_name: str
    exam_type: str
    total_no_of_questions: int                  # e.g., 30 questions
    level: Literal['easy', 'medium', 'hard']
    topic_focus: Optional[List[str]]            # e.g., ["Calculus", "Vectors"]
    include_hints: bool

# 2. Worker Parallel Task Payload (dispatched via Send API)
class QuestionBatchTask(BaseModel):
    batch_index: int
    batch_size: int                             # Strictly <= 10 questions
    difficulty: Literal['easy', 'medium', 'hard']
    target_topics: List[str]
    subject_name: str
    exam_type: str

# 3. Internal Subgraph Working State
class TestGenInternalState(TypedDict):
    input_params: TestGenInputState
    batch_tasks: List[QuestionBatchTask]
    raw_question_batches: Annotated[List[List[Question]], operator.add]
    deduplicated_questions: List[Question]
    validation_passed: bool
    generation_latency_seconds: float

# 4. Output Contract: Final Clean Data returned to API & Saved to DB
class TestGenOutputState(TypedDict):
    test_id: str
    subject_name: str
    exam_type: str
    total_questions: int
    level: str
    questions: List[Question]
    created_at: str
```

### 3.2 Test Evaluation & Gamification State Contracts
```python
# 1. Input Contract: User Answer Submission
class QuestionSubmission(BaseModel):
    question_id: str
    selected_option: Optional[Literal['option1', 'option2', 'option3', 'option4']]
    time_spent_seconds: float
    confidence_level: Optional[Literal['low', 'medium', 'high']]

class TestEvalInputState(TypedDict):
    user_id: str
    test_id: str
    attempt_id: str
    submissions: List[QuestionSubmission]
    total_time_taken_seconds: float

# 2. Internal Subgraph Working State
class GradedQuestion(BaseModel):
    question_id: str
    is_correct: bool
    user_answer: Optional[str]
    correct_answer: str
    marks_awarded: float
    time_spent_seconds: float
    explanation: str
    topic: str
    difficulty: str
    cognitive_error_type: Optional[str]         # "conceptual", "calculation", "timeout"

class TestEvalInternalState(TypedDict):
    input_data: TestEvalInputState
    test_paper: TestGenOutputState
    graded_questions: List[GradedQuestion]
    raw_score: float
    total_marks: float
    accuracy_percentage: float
    topic_mastery_map: Dict[str, float]
    gamification_profile_before: Dict[str, Any]
    xp_earned: int
    streak_updated: Dict[str, Any]
    newly_unlocked_badges: List[str]
    weak_areas: List[str]

# 3. Output Contract: Comprehensive Scorecard & Gamified Feedback
class TestEvalOutputState(TypedDict):
    attempt_id: str
    test_id: str
    score: float
    total_possible_score: float
    accuracy_percentage: float
    time_efficiency_rating: str
    detailed_graded_questions: List[GradedQuestion]
    topic_performance: Dict[str, float]
    gamification_delta: {
        "xp_gained": int,
        "new_total_xp": int,
        "current_level": int,
        "streak_days": int,
        "badges_awarded": List[Dict[str, str]],
    }
    remedial_recommendations: List[Dict[str, Any]]
```

---

## 4. 5-Week Implementation Roadmap (Sep 26 – Nov 1, 2026)

```
2026 Calendar Overview:
• Week 1: Mon Sep 28 – Sun Oct 04 | Deliverable & Report: Sunday, Oct 04, 2026
• Week 2: Mon Oct 05 – Sun Oct 11 | Deliverable & Report: Sunday, Oct 11, 2026
• Week 3: Mon Oct 12 – Sun Oct 18 | Deliverable & Report: Sunday, Oct 18, 2026
• Week 4: Mon Oct 19 – Sun Oct 25 | Deliverable & Report: Sunday, Oct 25, 2026
• Week 5: Mon Oct 26 – Sun Nov 01 | Deliverable & Report: Sunday, Nov 01, 2026
```

---

### Week 1 (Sep 28 – Oct 04, 2026): Domain Modeling, Modular State Architecture & Database Foundations

**Goal**: Establish isolated, extensible domain state contracts for Phase 2, decouple test generation from the main conversation graph, and set up database persistence schemas in Postgres.

#### Daily Step-by-Step Breakdown:
- **Monday, Sep 28**:
  - Audit existing state schemas in `src/domain/state.py`.
  - Create dedicated domain schema module `src/domain/subgraph_states.py`.
  - Implement Pydantic V2 models for `Options`, `Question`, `QuestionBatchTask`, `TestGenInputState`, `TestGenInternalState`, and `TestGenOutputState`.
- **Tuesday, Sep 29**:
  - Implement evaluation domain models: `QuestionSubmission`, `GradedQuestion`, `TestEvalInputState`, `TestEvalInternalState`, and `TestEvalOutputState`.
  - Define gamification data models: `UserGamificationProfile`, `XPTransaction`, `Badge`, `StreakTracker`, and `TopicMastery`.
- **Wednesday, Sep 30**:
  - Design Postgres tables & LangGraph BaseStore namespaces for Phase 2:
    - Table `test_papers` (storing generated test papers with JSONB question schemas).
    - Table `test_attempts` (storing user answer submissions and evaluation results).
    - Table `user_gamification` (storing total XP, current level, active streak, and badge arrays).
  - Draft async repository layer `src/db/test_repository.py` and `src/db/gamification_repository.py`.
- **Thursday, Oct 01**:
  - Architectural decoupling: Define a new SubGraph builder pattern in `src/graphs/test_generation_subgraph.py` and `src/graphs/evaluation_subgraph.py`.
  - Isolate graph compilation checkpointer & state schema so Subgraphs can execute independently without requiring conversational `messages` or Pinecone thread filters.
- **Friday, Oct 02**:
  - Write comprehensive unit tests for schema validation, type coercions, and serialization in `tests/unit/test_subgraph_states.py`.
  - Verify seamless interoperability with LangGraph `StateGraph(state_schema=..., input_schema=..., output_schema=...)`.
- **Saturday, Oct 03**:
  - End-to-end dry run of mock state passing between Input, Worker, and Output schemas.
  - Prepare documentation, architecture diagrams, and benchmark baseline for Sunday team report.
- **Sunday, Oct 04 (Team Milestone 1)**:
  - **Sunday Report & Presentation to Team**:
    - *Deliverable*: Production-ready `subgraph_states.py`, database schema migrations, and decoupled SubGraph architectural blueprint.
    - *Demonstration*: Schema validation test suite passing with 100% type coverage.

---

### Week 2 (Oct 05 – Oct 11, 2026): High-Performance Parallel Question Generation SubGraph (10-Question Chunking Engine)

**Goal**: Replace the single monolithic LLM prompt call with a parallelized map-reduce generator enforcing strict batches of at most 10 questions per worker, integrated with semantic deduplication and quality guardrails.

#### Daily Step-by-Step Breakdown:
- **Monday, Oct 05**:
  - Build `blueprint_partitioner_node`:
    - Takes `total_no_of_questions` (e.g., 25 or 50) and partitions into discrete worker tasks of **max 10 questions** (e.g., 25 questions -> 10 + 10 + 5).
    - Balances difficulty levels across batches (e.g., distributing easy/medium/hard proportionally).
    - Maps target sub-topics across batches to guarantee comprehensive syllabus coverage.
- **Tuesday, Oct 06**:
  - Implement `question_generator_worker_node`:
    - Optimized prompt loaded via Langfuse with few-shot IIT-JEE / exam examples.
    - Structured output validation with `Questions_generation_schema`.
    - Mathematical LaTeX formatting guardrails (`$$...$$` for formulas).
    - Enforce strictly 4 options with exactly 1 unambiguous correct answer and pedagogical explanations.
- **Wednesday, Oct 07**:
  - Parallel Orchestration via LangGraph `Send` API / `asyncio.gather`:
    - Connect `blueprint_partitioner_node` to parallel `question_generator_worker_node` instances.
    - Implement Groq rate-limiting backoff and retry mechanism for concurrent requests.
- **Thursday, Oct 08**:
  - Implement `question_aggregator_and_guardrails_node`:
    - Combines worker outputs into a single consolidated pool.
    - Semantic Deduplication: Computes sentence-transformer embeddings or Jaccard similarity between questions to flag and remove duplicate questions.
    - Option Sanity Check: Verifies that no question has identical options or missing keys.
- **Friday, Oct 09**:
  - Complete graph compilation of `TestPaperGenerationSubGraph` with Langfuse trace observation (`@observe`).
  - Implement fallback handling: If a worker fails, retry only that specific 10-question chunk rather than aborting the entire test paper.
- **Saturday, Oct 10**:
  - Benchmark generation speed: Compare old single-call latency vs. parallel 10-question chunking for 10, 20, 30, and 50 questions.
  - Finalize integration tests in `tests/unit/test_generation_subgraph.py`.
- **Sunday, Oct 11 (Team Milestone 2)**:
  - **Sunday Report & Presentation to Team**:
    - *Deliverable*: High-speed parallel test generation SubGraph producing 30+ question papers with zero schema failures.
    - *Demonstration*: Live generation of a 30-question IIT-JEE Physics & Math test paper in <8 seconds with deduplication analytics.

---

### Week 3 (Oct 12 – Oct 18, 2026): Interactive Test Submission, Scoring & Diagnostic Evaluation SubGraph

**Goal**: Build the test attempt execution lifecycle and the intelligent evaluation engine that performs automated scoring, timing analysis, and LLM-powered cognitive diagnostic feedback.

#### Daily Step-by-Step Breakdown:
- **Monday, Oct 12**:
  - Build `load_test_and_validate_node`:
    - Validates incoming `TestEvalInputState` against the stored `test_id`.
    - Checks submission bounds (handling unattempted questions, timestamp validity).
- **Tuesday, Oct 13**:
  - Implement `deterministic_scoring_node`:
    - High-speed scoring engine: Matches `selected_option` against ground truth.
    - Configurable exam marking schemes: Standard (+1 / 0) and Competitive Exam negative marking (e.g., IIT-JEE +4 for correct, -1 for incorrect, 0 for unattempted).
    - Computes time efficiency metrics: Average time per question, time spent on correct vs. incorrect answers.
- **Wednesday, Oct 14**:
  - Build `diagnostic_evaluator_node`:
    - LLM-powered cognitive failure analysis: Classifies incorrect answers into:
      1. *Conceptual Misunderstanding* (fundamental theory missed).
      2. *Calculation Error* (correct approach, arithmetic slip).
      3. *Distractor Trap* (fell for common misleading option).
      4. *Time Rush* (answered in <10s under pressure).
    - Generates personalized remedial advice per question.
- **Thursday, Oct 15**:
  - Implement `topic_mastery_aggregator_node`:
    - Groups performance by subject, chapter, and Bloom's cognitive level (Recall, Application, Analysis).
    - Computes topic mastery percentage (e.g., Calculus: 90%, Organic Chemistry: 35%).
- **Friday, Oct 16**:
  - Assemble `TestEvaluationSubGraph` in `src/graphs/evaluation_subgraph.py`.
  - Add Langfuse tracing for prompt accuracy and scoring evaluation.
- **Saturday, Oct 17**:
  - Write unit and integration tests in `tests/unit/test_evaluation_subgraph.py`.
  - Validate scoring accuracy with synthetic test submissions (perfect score, all wrong, partial completion, negative marking edge cases).
- **Sunday, Oct 18 (Team Milestone 3)**:
  - **Sunday Report & Presentation to Team**:
    - *Deliverable*: End-to-end evaluation engine with instant scorecard generation, negative marking, and cognitive diagnostic breakdown.
    - *Demonstration*: Submit a student test attempt and display the resulting diagnostic report with topic mastery percentages.

---

### Week 4 (Oct 19 – Oct 25, 2026): Gamified Learning Engine (XP, Streaks, Badges & Adaptive Remediation)

**Goal**: Implement the core gamification mechanics that reward learners, drive daily engagement, maintain streaks, award achievement badges, and dynamically recommend targeted remedial micro-tests.

#### Daily Step-by-Step Breakdown:
- **Monday, Oct 19**:
  - Implement `xp_and_progression_engine`:
    - Mathematical formula for XP reward:
      $$\text{XP} = (\text{Base Points} \times \text{Difficulty Multiplier}) \times \text{Accuracy Bonus} + \text{Speed Bonus}$$
    - Level progression curve: Level $L$ requires $100 \times L^{1.8}$ total XP.
    - Real-time level-up detection and reward multiplier trigger.
- **Tuesday, Oct 20**:
  - Build `streak_and_habit_engine`:
    - Daily streak tracking with UTC timezone normalization.
    - Consecutive day increment logic and streak protection / freeze mechanic.
    - Streak milestone multipliers (+5% XP bonus per 3 consecutive days, capped at +30%).
- **Wednesday, Oct 21**:
  - Implement `badge_and_achievement_evaluator`:
    - Event-driven rule engine for achievement unlocks:
      - *Speed Demon*: Average answer time <30s with accuracy >80%.
      - *Flawless Victory*: 100% score on a Medium/Hard test.
      - *JEE Conqueror*: Scored >85% on IIT-JEE Hard mock test.
      - *Consistency King*: Reached a 7-day active streak.
      - *Curiosity Spark*: Completed first test across 3 distinct subjects.
- **Thursday, Oct 22**:
  - Implement `adaptive_remedial_node`:
    - Analyzes weak topics (mastery <50%) from the evaluation output.
    - Automatically builds input state for an **Adaptive Remedial Micro-Test** (5 questions strictly focused on the identified weak concepts).
- **Friday, Oct 23**:
  - Integrate Gamification and Adaptive Remediation nodes directly into `TestEvaluationAndGamificationSubGraph`.
  - Persist updated user gamification profiles in LangGraph `BaseStore` / Postgres.
- **Saturday, Oct 24**:
  - Write test suite `tests/unit/test_gamification_engine.py` simulating XP gains, level-up transitions, streak freezes, and badge triggers.
- **Sunday, Oct 25 (Team Milestone 4)**:
  - **Sunday Report & Presentation to Team**:
    - *Deliverable*: Fully functioning gamification loop with XP, streaks, level progression, badges, and adaptive remedial triggers.
    - *Demonstration*: Complete student test cycle showing XP award, badge unlock popup data, and auto-generated remedial quiz recommendation.

---

### Week 5 (Oct 26 – Nov 01, 2026): API Routes Integration, Langfuse Evals, Stress Testing & Production Launch

**Goal**: Expose production FastAPI endpoints, establish automated Langfuse evaluation benchmarks, perform high-concurrency load testing, finalize documentation, and achieve production deployment signoff.

#### Daily Step-by-Step Breakdown:
- **Monday, Oct 26**:
  - Build and mount the Phase 2 API Router in `api/routes/sub_graph_routes.py`:
    - `POST /api/v1/subgraph/test/generate`: Starts parallel test generation.
    - `POST /api/v1/subgraph/test/submit`: Submits answers, executes evaluation, updates gamification.
    - `GET /api/v1/subgraph/test/performance/{attempt_id}`: Retrieves comprehensive scorecard.
    - `GET /api/v1/subgraph/test/gamification/profile`: Retrieves user XP, level, badges, and active streak.
    - `POST /api/v1/subgraph/test/remedial/generate`: Spawns instant weak-area adaptive quiz.
- **Tuesday, Oct 27**:
  - Connect `GraphRunnerPipeline` with dedicated async subgraph invokers (`initiate_parallel_test_generation` and `evaluate_test_attempt`).
  - Wire authentication middleware `authenticate_only_user` ensuring strict multi-tenant isolation.
- **Wednesday, Oct 28**:
  - Create Langfuse automated evaluation dataset in `evals_langfuse/data/test_generation_eval_dataset.json`.
  - Implement LLM Judge metrics in `evals_langfuse/metrics/test_quality_eval.py`:
    - Question clarity and scientific correctness score (0.0 to 1.0).
    - Distractor quality and plausibility score.
    - Explanation educational depth score.
- **Thursday, Oct 29**:
  - High-Concurrency Stress & Load Testing:
    - Simulate 50 concurrent test generation requests.
    - Benchmark Neon Postgres connection pool under concurrent evaluation writes.
    - Verify Groq token rate limits and backoff stability.
- **Friday, Oct 30**:
  - Clean up deprecated code paths and finalize documentation.
  - Update OpenAPI Swagger schemas, tags, and request/response examples in `api/main.py`.
- **Saturday, Oct 31**:
  - Production readiness checklist, security review, and full regression test suite run (`pytest tests/`).
  - Code freeze and staging deployment verification.
- **Sunday, Nov 01 (Final Phase 2 Launch Day)**:
  - **Sunday Report & Executive Presentation to Team**:
    - *Deliverable*: Production-ready Phase 2 Gamified Learning & Test Generation Engine live on `/api/v1/subgraph/test/*`.
    - *Demonstration*: Full live demo from test paper generation -> student taking test -> instant evaluation -> gamification level-up & badge unlock -> adaptive remedial quiz generation.

---

## 5. Sunday Team Reporting Checklist & Deliverables

Every Sunday at the team sync, present the deliverables following this standardized structure:

| Sunday | Date | Milestone Title | Key Deliverable & Demo Script | Verification Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **Sunday 1** | Oct 04, 2026 | **Contracts, Schemas & Decoupled State** | Demo decoupled `TestGenInputState`, `TestEvalInputState`, and Neon DB schema migrations. | 100% type-checked models; zero coupling with conversation state. |
| **Sunday 2** | Oct 11, 2026 | **Parallel 10-Question Chunking Engine** | Live generation of a 30-question test paper across 3 parallel LLM workers with deduplication. | <8s latency for 30 questions; 0 schema parse failures; 0 duplicate questions. |
| **Sunday 3** | Oct 18, 2026 | **Interactive Scoring & Diagnostic Engine** | Student submission demo showing instant negative marking, time efficiency, and LLM diagnostic feedback. | 100% grading accuracy on synthetic test attempts; cognitive error tagging verified. |
| **Sunday 4** | Oct 25, 2026 | **Gamification Engine & Remedial Loop** | Complete gamification cycle showing XP gain, streak calculation, badge unlock, and remedial test suggestion. | Deterministic level curve validation; streak freeze handling; remedial quiz triggering. |
| **Sunday 5** | Nov 01, 2026 | **Phase 2 Production Launch** | Full end-to-end integration demo across all REST endpoints with Langfuse traces and stress test report. | All integration tests passing; API response p95 < 2.5s; load tested to 50 concurrent users. |

---

## 6. Technical Safeguards, Risk Management & Fallback Strategies

1. **Groq Rate Limits (TPM / RPM) during Parallel 10-Question Chunks**:
   - *Risk*: Firing 5 concurrent workers generating 10 questions each may exceed Groq TPM rate limits.
   - *Safeguard*: Implement an asynchronous semaphore (`asyncio.Semaphore(3)`) combined with exponential jittered backoff. If rate-limited, fail over to backup model or serialize batches smoothly without crashing the request.
2. **Schema Hallucination & Malformed JSON**:
   - *Risk*: LLM occasionally drops options or returns invalid keys in complex mathematical problems.
   - *Safeguard*: Use Pydantic field validators with automatic correction (e.g., coercing string keys, defaulting missing points to 10). If validation fails on a chunk, retry only that specific 10-question chunk.
3. **Database Connection Pool Exhaustion on Test Submissions**:
   - *Risk*: Simultaneous student submissions could exhaust Neon Postgres pool connections.
   - *Safeguard*: Leverage the existing `AsyncConnectionPool` with `max_size=MAXIMUM_CONNECTION_POOL_SIZE` and execute evaluation scoring in-memory before writing batch updates in a single atomic transaction.
4. **State Isolation Guarantee**:
   - *Risk*: Developers inadvertently mixing conversation state (`messages`) into test generation.
   - *Safeguard*: Enforce strict type checking in Subgraph definitions with separate `input_schema` and `output_schema` contracts in `src/graphs/test_generation_subgraph.py` and `src/graphs/evaluation_subgraph.py`.

---
*Plan created and verified on Saturday, September 26, 2026.*
