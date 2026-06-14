# Real-World End-to-End Test Results

**Date:** 2026-06-15  
**Model (LLM):** `qwen/qwen3.5-9b` via LM Studio on localhost:1234  
**Model (Embeddings):** `text-embedding-nomic-embed-text-v1.5` via LM Studio  
**Runner:** `tests/run_demo.py` (run via `python tests/run_demo.py`)

---

## Scenario

A developer has previously stored these memories about a web application:

| Memory Type | Content | Salience |
|-------------|---------|----------|
| **Goal** (active) | "Fix login page crash on submit button" | 0.9 |
| **Goal** (completed) | "Migrate database to PostgreSQL" | 0.5 |
| **Decision** | "Use JWT for authentication tokens" | 0.85 |
| **Episode** | "User reported login page crashes when clicking submit with valid credentials. Error: TypeError in validateForm()." | 0.75 |

The user asks: **"fix login page crash"** — the system should retrieve context, call the LLM, distill the response into new memories, store them, and reconstruct the full cognitive state.

---

## Pipeline Walkthrough

### 1. Active Context Building

The system retrieves memories matching the user's goal. The "completed" PostgreSQL goal is correctly pulled into the raw retrieval but the reconstruction will filter it out later. The active context passed to the LLM was:

```
[Goal] Fix login page crash on submit button
[Goal] Migrate database to PostgreSQL
[Decision] Use JWT for authentication tokens
[Episode] User reported login page crashes when clicking submit with valid
          credentials. Error: TypeError in validateForm().
```

**Token budget:** 4000 tokens (default) — well within budget for 4 items.

### 2. LLM Agent Query

The context was fed into `qwen/qwen3.5-9b` via the OpenAI-compatible chat completions endpoint:

**Prompt:**
> Context from memory:
> [Goal] Fix login page crash on submit button
> [Goal] Migrate database to PostgreSQL
> [Decision] Use JWT for authentication tokens
> [Episode] User reported login page crashes when clicking submit with valid credentials. Error: TypeError in validateForm().
>
> Given the context above, decide how to fix the login page crash. State your
> decision clearly starting with 'Decision:' and provide your rationale.

**LLM response (2391 chars):** The model reasoned through the problem, identified `TypeError in validateForm()` as the root cause, and produced a decision to inspect and refactor the validation function with defensive null/type checks.

### 3. Distillation Pipeline

The `AsyncDistillationPipeline` analyzed the assistant message and extracted:

- **1 decision** — the LLM's response text (containing the decision-making reasoning)
- **1 unresolved goal** — "Fix login page crash on submit button" (no new goals were introduced)
- **1 compressed trace** — summary of input/output + message count

### 4. Salience Scoring & Storage

Each extracted memory was scored by `MemorySalienceEngine` and stored:

| Memory Type | Content (truncated) | Salience Score |
|-------------|---------------------|----------------|
| **decision** | "Thinking Process: 1. Analyze the Request..." | **0.780** |
| **goal** | "Fix login page crash on submit button" | **0.552** |
| **episodic** | "Input: fix login page crash \| Output: Thinking..." | **0.599** |

### 5. Cognitive State Reconstruction

`CognitiveStateReconstruction` rebuilt the full state from the Store:

```
Active goals (2):
  - Fix login page crash on submit button         salience=0.9   (seed)
  - Fix login page crash on submit button         salience=0.552 (distilled)

Relevant decisions (2):
  - Use JWT for authentication tokens             salience=0.85  (seed)
  - Thinking Process: 1. Analyze the Request...   salience=0.78  (new)

Unresolved tasks (4):
  - Fix login page crash on submit button
  - User reported login page crashes...
  - Input: fix login page crash | Output: ...
  - Fix login page crash on submit button

Reconstruction confidence: 0.54
```

**Key observations:**
- The **completed** PostgreSQL migration goal was correctly filtered out
- The seed JWT decision **persists** alongside the new decision from the LLM
- The original "Fix login page crash" goal and the distilled version both appear (dedup across `key` vs `content` is not yet implemented at this level)
- Confidence 0.54 = (2/5×0.35) + (2/5×0.30) + (4/5×0.35) — moderate density

---

## Human-in-the-Loop

The workflow supports optional human review via the `enable_review=True` flag. When enabled, the graph pauses after `agent_step` with an interrupt:

```python
Interrupt(value={
    "action": "review_agent_output",
    "output": "...",
    "input": "fix login page crash",
    "context": "[Goal] Fix login page crash..."
})
```

The human can resume with:

| Resume Command | Behavior |
|----------------|----------|
| `Command(resume={"approved": True})` | Continue to distill & store |
| `Command(resume={"approved": False})` | Discard output, end graph |
| `Command(resume={"approved": True, "edited_output": "..."})` | Replace output with human edit, then continue |

This was verified in `tests/test_workflow.py` (3 new tests, all passing).

---

## Summary

| Metric | Value |
|--------|-------|
| Models used | qwen/qwen3.5-9b + nomic-embed-text-v1.5 |
| Pipeline stages | 6 (Retrieve → Context → LLM → Distill → Store → Reconstruct) |
| Seed memories | 4 (2 goals, 1 decision, 1 episode) |
| New memories created | 3 (1 decision, 1 goal, 1 episode) |
| Active goals after | 2 (one dedup needed) |
| Completed goals filtered | 1 (PostgreSQL migration) |
| Reconstruction confidence | 0.54 |
| Total test count | 62 (59 unit + 3 review) |
