## Goal state (what you’re building)

A **GraphRAG workspace** where:

* **All LLM “chat” calls** (indexing + query: `local`, `global`, `drift`, `basic`) go to **your local Ollama** serving
  `nemotron-3-nano:30b-a3b-q4_K_M-220k` at `http://172.17.0.1:11434/v1` via **OpenAI-compatible** `/v1/chat/completions`.
* **All embeddings** go to **your vLLM OpenAI-compatible server** hosting `nvidia/llama-embed-nemotron-8b` via `/v1/embeddings`.
* DRIFT is configured for **maximum recall + stability** (high-K, deeper traversal, multiple repeats), while respecting your **single-inference-at-a-time** constraint.

Also: the earlier Gemini doubts about “hallucinated DRIFT YAML keys” are resolvable: GraphRAG’s official YAML schema explicitly defines a `drift_search` section with keys like `drift_k_followups`, `primer_folds`, `primer_llm_max_tokens`, `n_depth`, and `concurrency`.

---

## 0) Non-negotiable constraints (so it doesn’t explode later)

### A. OpenAI-compatible endpoints are *real* and required here

* Ollama exposes `/v1/chat/completions` and documents OpenAI client usage via `base_url=http://localhost:11434/v1` (API key required-but-ignored).
* vLLM explicitly supports the **Embeddings API** at `/v1/embeddings` for embedding models.

### B. Context length is *not* set via OpenAI API

OpenAI-style calls don’t include “context size”; Ollama’s OpenAI-compat docs explicitly note you must set context size in the **model definition (Modelfile/template)** if you need a different context.
So: GraphRAG must be configured to **stay under** your 220k window by budgeting its internal “max_context_tokens / max_input_length / data_max_tokens” fields (details below).

### C. You cannot run two GPU requests at once

GraphRAG DRIFT mode provides an explicit `concurrency` knob. Set it to **1**.
However, **global search map-reduce** may still issue multiple calls depending on implementation; don’t trust it. The robust fix is: **put a single-flight proxy in front of both Ollama + vLLM** so *the whole system* is serialized (one in-flight request total). That guarantees correctness under your “one request at a time” rule regardless of GraphRAG internals.

I’m going to give you that proxy (small, boring, works).

---

## 1) Bring up the two model servers (expected network contracts)

### 1.1 Ollama (chat LLM)

You already have it, but the contract must hold:

* **OpenAI base URL:** `http://172.17.0.1:11434/v1`
* **Chat endpoint used:** `/v1/chat/completions`
* **Model name GraphRAG will request:** `nemotron-3-nano:30b-a3b-q4_K_M-220k`

**Sanity test (must succeed):**

```bash
curl http://172.17.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron-3-nano:30b-a3b-q4_K_M-220k",
    "messages": [{"role":"user","content":"Reply with exactly: OK"}],
    "temperature": 0
  }'
```

> Why I’m confident about this endpoint shape: Ollama’s OpenAI-compat blog and docs show `/v1/chat/completions` and using `base_url=http://…/v1` with a dummy key.

### 1.2 vLLM (embeddings)

Contract:

* **OpenAI base URL:** `http://172.17.0.1:8000/v1` (use port 8000; that’s vLLM’s documented default pattern)
* **Embeddings endpoint:** `/v1/embeddings`
* **Embedding model name:** `nvidia/llama-embed-nemotron-8b`

**Sanity test (must succeed):**

```bash
curl http://172.17.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-abc123" \
  -d '{
    "model": "nvidia/llama-embed-nemotron-8b",
    "input": "embedding smoke test"
  }'
```

vLLM explicitly lists `/v1/embeddings` as a supported OpenAI API for embedding models.

---

## 2) Enforce “only one request at a time” (single-flight proxy)

Run this on the same machine that can reach both `172.17.0.1:11434` and `172.17.0.1:8000`.

### 2.1 Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn httpx
```

### 2.2 Proxy (serializes ALL calls)

Save as `singleflight_openai_proxy.py`:

```python
import asyncio
from fastapi import FastAPI, Request, Response
import httpx

# Targets
OLLAMA_BASE = "http://172.17.0.1:11434"
VLLM_BASE   = "http://172.17.0.1:8000"

# One in-flight request across BOTH backends
SEM = asyncio.Semaphore(1)

app = FastAPI()

async def _forward(req: Request, target_url: str) -> Response:
    body = await req.body()
    headers = dict(req.headers)
    headers.pop("host", None)

    async with SEM:
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.request(
                method=req.method,
                url=target_url,
                content=body,
                headers=headers,
            )
    return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))

# Ollama OpenAI-compatible routes
@app.api_route("/ollama/{path:path}", methods=["GET","POST","PUT","PATCH","DELETE"])
async def ollama(path: str, req: Request):
    return await _forward(req, f"{OLLAMA_BASE}/{path}")

# vLLM OpenAI-compatible routes
@app.api_route("/vllm/{path:path}", methods=["GET","POST","PUT","PATCH","DELETE"])
async def vllm(path: str, req: Request):
    return await _forward(req, f"{VLLM_BASE}/{path}")
```

Run it:

```bash
uvicorn singleflight_openai_proxy:app --host 0.0.0.0 --port 9000
```

You now have:

* Ollama proxied at `http://<this-host>:9000/ollama/v1/...`
* vLLM proxied at `http://<this-host>:9000/vllm/v1/...`

This is the cleanest way to **guarantee** GraphRAG never overlaps GPU work, while still allowing you to scale later by changing `Semaphore(1)` to a higher number.

---

## 3) Create a GraphRAG workspace

### 3.1 Initialize

```bash
mkdir graphrag_nemotron && cd graphrag_nemotron
# Use GraphRAG's official init command for your installed version
graphrag init --root .
```

(Your workspace will include `settings.yml`, `prompts/`, and an `input/` folder.)

### 3.2 Put your corpus into `./input/`

* Plain text files are ideal.
* If you have PDFs/HTML, convert to clean text first (GraphRAG quality is extremely sensitive to garbage characters and layout detritus).

---

## 4) **Opinionated “max quality” settings.yml** (Nemotron-only)

You will edit only fields that GraphRAG’s schema explicitly defines (so it validates cleanly).

### 4.1 Models: route through the proxy (recommended)

Set these in `settings.yml` under your `models:` section (IDs can be any names; I’ll use these consistently):

```yaml
models:
  nemotron_chat:
    type: openai_chat
    model: nemotron-3-nano:30b-a3b-q4_K_M-220k
    api_base: http://127.0.0.1:9000/ollama/v1
    api_key: ollama
    temperature: 0
    max_tokens: 20000

  nemotron_embed:
    type: openai_embedding
    model: nvidia/llama-embed-nemotron-8b
    api_base: http://127.0.0.1:9000/vllm/v1
    api_key: token-abc123
```

Why this shape is grounded:

* Ollama’s OpenAI-compat uses `base_url=.../v1` and a dummy key.
* vLLM supports OpenAI-compatible APIs including `/v1/embeddings` for embedding models.

### 4.2 Community reports: tuned for DRIFT @ high-K

Community reports must be **dense** but not so huge that DRIFT primer can’t fit K=100.

GraphRAG defines:

* `community_reports.max_length` (output tokens per report)
* `community_reports.max_input_length` (input tokens when generating reports)

Set:

```yaml
community_reports:
  model_id: nemotron_chat
  max_length: 1400
  max_input_length: 200000
```

Rationale:

* With `drift_k_followups: 100`, you cannot safely shove “long essays” ×100 into a 220k window and still have room for the question, instructions, and the model’s own output.
* 1400-token reports are still rich, but the math stays survivable.

(You asked for “no options”, so this is the single tightrope-walk that keeps “K=100” compatible with “220k context”.)

### 4.3 Query configs (the heart of “full quality”)

GraphRAG’s schema-defined query knobs are here: `local_search`, `global_search`, `drift_search`, `basic_search`.
And GraphRAG CLI supports running these methods via `--method local|global|drift|basic`, plus `--community-level` and `--dynamic-community-selection` for global search.

#### Local search (wide + deep)

```yaml
local_search:
  chat_model_id: nemotron_chat
  embedding_model_id: nemotron_embed
  top_k_entities: 120
  top_k_relationships: 400
  max_context_tokens: 180000
  conversation_history_max_turns: 0
```

#### Global search (static + dynamic-ready)

```yaml
global_search:
  chat_model_id: nemotron_chat
  max_context_tokens: 200000
  data_max_tokens: 200000
  map_max_length: 3500
  reduce_max_length: 7000

  # Stability for dynamic selection
  dynamic_search_threshold: 0
  dynamic_search_keep_parent: true
  dynamic_search_num_repeats: 3
  dynamic_search_use_summary: false
  dynamic_search_max_level: 8
```

Those dynamic selection fields are explicitly part of the schema.

#### DRIFT search (the “full quality” engine)

This is the “K=100 stable” build:

```yaml
drift_search:
  chat_model_id: nemotron_chat
  embedding_model_id: nemotron_embed

  concurrency: 1  # you said: cannot do two requests at once
  drift_k_followups: 100
  primer_folds: 12
  primer_llm_max_tokens: 20000
  n_depth: 4

  # Make local phases huge and thorough
  local_search_top_k_mapped_entities: 120
  local_search_top_k_relationships: 400
  local_search_max_data_tokens: 180000

  # Stability: low temperature, multiple completions
  local_search_temperature: 0
  local_search_top_p: 1
  local_search_n: 3
  local_search_llm_max_gen_tokens: 12000

  # Reduce phase budgets (non “o-series” models)
  data_max_tokens: 200000
  reduce_max_tokens: 20000
```

Every key above is directly named in the schema.

#### Basic search (baseline “did we miss obvious chunks?”)

```yaml
basic_search:
  chat_model_id: nemotron_chat
  embedding_model_id: nemotron_embed
  k: 200
```

---

## 5) Index (high quality, Nemotron-only)

Run:

```bash
graphrag index --root .
```

What this does: build the entity/relationship graph, community structure, and reports (which DRIFT uses as its “global primer substrate”). DRIFT’s strength is exactly that it can start from community summaries and then drive local follow-ups. That whole mechanism is the reason GraphRAG is not “just cosine similarity.” (DRIFT described conceptually in the GraphRAG ecosystem; and GraphRAG explicitly documents DRIFT configuration and query mode.)

---

## 6) Run your 5-method “compute-unlimited ensemble” (no user choice)

These commands are grounded in GraphRAG’s CLI docs: `--method`, `--community-level`, and `--dynamic-community-selection`.

```bash
QUERY="…your question…"

# 1) DRIFT
graphrag query --root . --method drift  --query "$QUERY" > out_drift.txt

# 2) Global (static) at deepest community level you have
# (You will know the max level from your index outputs; choose the deepest.)
graphrag query --root . --method global --community-level 8 --query "$QUERY" > out_global_static.txt

# 3) Global (dynamic community selection)
graphrag query --root . --method global --dynamic-community-selection --query "$QUERY" > out_global_dynamic.txt

# 4) Local (wide)
graphrag query --root . --method local --query "$QUERY" > out_local.txt

# 5) Basic (high-k baseline)
graphrag query --root . --method basic --query "$QUERY" > out_basic.txt
```

---

## 7) Merge the 5 outputs (your “cross-reference” step), still Nemotron-only

You said: “all of the results will then be cross-referenced … to get the best answer regardless of question type.”

Do that with a final LLM pass that:

* deduplicates facts,
* resolves conflicts by preferring facts that appear in **2+ methods**,
* and keeps citations (GraphRAG outputs typically include provenance; preserve it).

Skeleton script (uses OpenAI-compatible chat; Ollama supports that format):

```python
from openai import OpenAI
from pathlib import Path

client = OpenAI(base_url="http://127.0.0.1:9000/ollama/v1", api_key="ollama")

parts = []
for name in ["out_drift.txt","out_global_static.txt","out_global_dynamic.txt","out_local.txt","out_basic.txt"]:
    parts.append(f"=== {name} ===\n{Path(name).read_text(encoding='utf-8')}\n")

prompt = """You are merging 5 GraphRAG answers from different methods (drift/global/local/basic).
Rules:
- Produce one final answer.
- Any factual claim must be supported by evidence present in at least one method output.
- Prefer claims supported by multiple methods; explicitly note disagreements.
- Preserve and carry over citations/provenance text when present.
- Be comprehensive. Do not omit relevant details.
"""

resp = client.chat.completions.create(
    model="nemotron-3-nano:30b-a3b-q4_K_M-220k",
    messages=[
        {"role":"system","content":prompt},
        {"role":"user","content":"\n\n".join(parts)}
    ],
    temperature=0,
    max_tokens=20000,
)
print(resp.choices[0].message.content)
```

---

## 8) Context-length reality check (why 220k is “enough” for *this* DRIFT build)

Nemotron 3 Nano is explicitly positioned as a long-context model (the Ollama library listing emphasizes long context and reports 1M context-window variants).
But GraphRAG DRIFT’s *practical* context usage is dominated by:

1. **Primer input size**: `drift_k_followups × (community report length)`
2. **Primer output**: `primer_llm_max_tokens`
3. **Local step context**: `local_search_max_data_tokens` + the number of entities/relations you map (`*_top_k_*`)
4. **Depth**: `n_depth` multiplies steps; not a single-context blowup, but a total cost blowup.

This is exactly why I forced:

* community report length down to 1400 tokens (so K=100 fits),
* local search context to 180k (so you actually exploit the 220k window),
* and kept headroom by not setting everything to 220,000 exactly (token counting is never perfectly aligned across systems).

---

## 9) If anything fails, it will be one of these (and what it *means*)

1. **“Context length exceeded” / truncated outputs**

   * Means your 220k window is real, but your budgets are too tight to the edge.
   * The fix is not “lower K” (you demanded K=100); the correct fix is: community reports must stay near the 1400 target, because that’s the only lever that reduces the primer payload without sacrificing K.

2. **JSON/structure parsing errors during indexing**

   * Means Nemotron is emitting extra text when GraphRAG expects strict structure.
   * The only correct fix (given your “Nemotron-only” rule) is: make sure your Ollama model template respects OpenAI JSON mode / response formatting (Ollama supports JSON mode in OpenAI-compat).

3. **GPU VRAM spikes / overlapping inference**

   * Means you bypassed the proxy or DRIFT concurrency wasn’t 1.
   * DRIFT exposes `concurrency` explicitly; it must remain 1.
   * The proxy guarantees serialization even if other parts parallelize.

---

If you want, paste your **generated** `settings.yml` after `graphrag init` (the whole file). I’ll rewrite it into the final “Nemotron-only, 220k-aware, single-flight” version while keeping every key strictly schema-valid.
