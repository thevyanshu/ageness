# Architecture

## Layer Model

```
┌─────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                      │
│            Deep Agents (orchestration, agents)            │
├─────────────────────────────────────────────────────────┤
│                 COGNITIVE MEMORY LAYER                    │
│   (Active Context Builder, Async Distillation,            │
│    Memory Salience, Hybrid Retrieval,                    │
│    Cognitive State Reconstruction)                       │
├─────────────────────────────────────────────────────────┤
│               EXECUTION SUBSTRATE (LangGraph)             │
│   (StateGraph, nodes, edges, checkpointers, Store,        │
│    interrupt/retry, Send API, time travel)                │
├─────────────────────────────────────────────────────────┤
│               FOUNDATION (LangChain)                      │
│   (models, tools, prompts, retrievers, RAG)               │
└─────────────────────────────────────────────────────────┘
```

LangGraph is the **operating kernel**. All workflow orchestration, state transitions, checkpointing, interrupt handling, and graph execution use LangGraph primitives directly. The project-specific innovation layer exists **above** LangGraph — it does not duplicate LangGraph's capabilities.

---

## LangGraph Responsibilities

| Capability | LangGraph Primitive |
|---|---|
| Graph execution flow | `StateGraph.compile()`, `invoke()`, `stream()` |
| Node orchestration | `add_node()` with functions or `Command` |
| Agent workflow management | Graphs with conditional edges, loops, fan-out via `Send` |
| Checkpoint persistence | `Checkpointer` (InMemorySaver, PostgresSaver, SqliteSaver) |
| Execution state transitions | Reducers, `update_state()`, `Overwrite` |
| Resumable workflows | Checkpointer + `thread_id` + `interrupt()` |
| Interrupt/retry handling | `interrupt()`, `Command(resume=...)`, `RetryPolicy` |
| Tool execution routing | `ToolNode` with `handle_tool_errors=True` |
| State propagation | TypedDict state schemas with `Annotated` reducers |
| Branchable execution DAGs | `add_conditional_edges()`, `Send` fan-out |
| Cross-thread long-term memory | `Store` (InMemoryStore, PostgresStore) |
| Time travel / history | `get_state_history()`, `update_state()` |

No custom wrappers, abstractions, or reimplementations of the above. If LangGraph provides it, we use it directly.

---

## Custom Innovation Layers

### 1. Active Context Builder

**Purpose:** Dynamically assembles a minimal working context from memory, relevant to the current execution scope.

**Location:** `src/cognition/active_context/`

**Responsibilities:**
- On each graph node entry, queries the Hybrid Retrieval system for relevant memories
- Compresses retrieved context into a bounded window (token budget)
- Prioritizes information by salience score and temporal recency
- Injects reconstructed context into the node's state before execution

**Interface:**
```python
class ActiveContextBuilder:
    async def build_context(
        self, current_state: dict, thread_id: str, goals: list[str]
    ) -> ContextWindow: ...
```

### 2. Async Distillation Pipeline

**Purpose:** Background cognition system that runs after key execution steps to extract, compress, and persist structured memories.

**Location:** `src/cognition/distillation/`

**Responsibilities:**
- Extracts decisions, rationale, and outcomes from execution traces
- Compresses verbose state into compact episodic memories
- Identifies unresolved goals and open tasks
- Persists structured memories to the Store via the Memory Salience Engine

**Execution Model:**
- Triggered as a LangGraph node with low priority or via background task
- Does not block the main execution flow
- Uses LangGraph's checkpoint data as input (via `get_state_history`)

**Interface:**
```python
class AsyncDistillationPipeline:
    async def distill(
        self, thread_id: str, checkpoint_id: str
    ) -> DistillationResult: ...
```

### 3. Memory Salience Engine

**Purpose:** Determines what should persist, decay, or be ignored.

**Location:** `src/cognition/salience/`

**Responsibilities:**
- Scores each potential memory item on relevance, importance, and novelty
- Applies decay functions to existing memories over time
- Filters low-salience items before storage
- Triggers memory consolidation (merging related items)

**Interface:**
```python
class MemorySalienceEngine:
    async def score(self, item: RawMemory) -> SalienceScore: ...
    async def decay(self, store, namespace: tuple) -> None: ...
    async def consolidate(self, store, namespace: tuple) -> None: ...
```

### 4. Hybrid Retrieval System

**Purpose:** Combines multiple retrieval strategies to find relevant memories.

**Location:** `src/cognition/retrieval/`

**Strategies:**
- **Semantic retrieval:** Embedding-based similarity search against memory vectors
- **Temporal retrieval:** Recency-based search across checkpoint history
- **Checkpoint traversal:** Direct state inspection via LangGraph's time travel API
- **Graph retrieval (optional):** Relationship traversal across linked memories

**Interface:**
```python
class HybridRetrievalSystem:
    async def retrieve(
        self, query: RetrievalQuery, strategies: list[RetrievalStrategy]
    ) -> list[MemoryItem]: ...
```

### 5. Cognitive State Reconstruction

**Purpose:** Rebuilds an agent's full cognitive state from distributed memory.

**Location:** `src/cognition/reconstruction/`

**Capabilities:**
- Reconstructs active goals from episodic memories
- Surfaces relevant past decisions and their outcomes
- Identifies unresolved tasks and contextual dependencies
- Produces a structured cognitive state object for injection into graph state

**Interface:**
```python
class CognitiveStateReconstruction:
    async def reconstruct(
        self, thread_id: str, query: str
    ) -> CognitiveState: ...
```

---

## Execution Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│  LangGraph Graph Entry              │
│  (checkpointer loads thread state)  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Active Context Builder             │  ← COGNITIVE LAYER
│  (retrieves relevant memories,      │
│   assembles context window)         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Execution Node                     │  ← LANGGRAPH
│  (LLM call, tool use, etc.)         │
└─────────────────────────────────────┘
    │
    ├──▶ Async Distillation Pipeline  │  ← COGNITIVE LAYER (background)
    │    (extract decisions, goals,    │
    │     compress traces)             │
    │         │                        │
    │         ▼                        │
    │    Memory Salience Engine        │
    │    (score, filter, persist)      │
    │         │                        │
    │         ▼                        │
    │    Store (long-term memory)      │  ← LANGGRAPH PRIMITIVE
    │
    ▼
┌─────────────────────────────────────┐
│  Next Node / END                    │
└─────────────────────────────────────┘
```

---

## Storage Architecture

### Short-term Memory (Thread-scoped)

- **Mechanism:** LangGraph Checkpointer (PostgresSaver in production)
- **Scope:** Per `thread_id`
- **Content:** Full graph state at each super-step
- **Access:** `graph.get_state()`, `graph.get_state_history()`
- **Retention:** Configurable TTL per thread

### Long-term Memory (Cross-thread)

- **Mechanism:** LangGraph Store (PostgresStore in production)
- **Scope:** Namespace-based (e.g., `(user_id, memory_type)`)
- **Content:** Distilled episodic memories, preferences, facts, decisions
- **Access:** `runtime.store.get()`, `runtime.store.put()`, `runtime.store.search()`
- **Retention:** Governed by Memory Salience Engine (decay, consolidation)

### Memory Namespace Convention

```
(user_id, "episodic")     → Distilled execution episodes
(user_id, "semantic")     → Facts, preferences, learned patterns
(user_id, "goals")        → Active and resolved goals
(user_id, "decisions")    → Key decisions with rationale
```

---

## Key Design Constraints

1. **No custom orchestration** — every graph, edge, loop, fan-out, interrupt, and retry uses LangGraph primitives
2. **No custom checkpointing** — use Checkpointer directly; no wrapper abstractions
3. **No custom state management** — use TypedDict + Annotated reducers
4. **No custom runtime** — use LangGraph's `Runtime` object for store access
5. **Innovation lives in the cognitive layer** — context building, distillation, salience, retrieval, reconstruction

---

## Technology Stack

| Layer | Technology |
|---|---|
| Foundation | LangChain (models, tools, prompts) |
| Orchestration | LangGraph (StateGraph, Checkpointer, Store) |
| Agent Framework | Deep Agents (middleware, planning, subagents) |
| Persistence | PostgreSQL (via PostgresSaver, PostgresStore) |
| Embeddings | OpenAI / any LangChain embedding model |
| Vector Store | Optional — can use Store's built-in search or external vector DB |
| Language | Python 3.11+ |
