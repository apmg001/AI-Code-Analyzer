# AI Code Analyzer

> The only free local code analyzer that tracks data flow across functions,
> detects statistical anomalies, learns your codebase conventions,
> traces bugs to their git origin, and verifies its own patches
> against your test suite.

![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![LLM](https://img.shields.io/badge/LLM-llama.cpp%20%7C%20Qwen%2014B-orange)

---

## Results on Flask 3.0

| Metric | Result |
|---|---|
| Files scanned | 35 |
| Functions extracted | 421 |
| Chunks analyzed | 356 |
| Issues detected | 65 |
| Patch coverage | 75.4% |
| LLM patches generated | 43 |
| Runtime | 4 min 23 sec |
| Cloud dependency | None |

---

## What makes this different

| Capability | pylint | SonarQube | This tool |
|---|---|---|---|
| Rule-based detection | ✅ | ✅ | ✅ 9 rules |
| Taint / data flow analysis | ❌ | Paid only | ✅ Free |
| Statistical anomaly detection | ❌ | ❌ | ✅ IsolationForest |
| Codebase convention learning | ❌ | ❌ | ✅ DNA fingerprinting |
| Git archaeology (who/when) | ❌ | ❌ | ✅ |
| Patch verification via tests | ❌ | ❌ | ✅ |
| Local LLM patch generation | ❌ | ❌ | ✅ Qwen 14B |
| Runs 100% offline | ❌ | ❌ | ✅ |

---

## Quickstart

```bash
git clone https://github.com/apmg001/AI-Code-Analyzer
cd AI-Code-Analyzer

pip install -r requirements.txt

# Run on any GitHub repo
python analyze_repo.py https://github.com/pallets/flask

# With specific options
python analyze_repo.py https://github.com/pallets/flask --provider groq
python analyze_repo.py https://github.com/pallets/flask --min-confidence 0.80

# Quick local test (no clone needed)
python analyze_repo.py --test
```

---

## 13-Stage Pipeline

```
[1]  Clone repository
[2]  Scan Python files
[3]  Taint analysis          ← novel: tracks data flow across functions
[4]  Extract functions (AST)
[5]  Chunk code
[6]  Filter production code
[7]  Generate embeddings     ← sentence-transformers, all-MiniLM-L6-v2
[8]  Rule-based + semantic detection
[9]  Anomaly detection       ← novel: IsolationForest on embeddings
[10] Codebase DNA            ← novel: learns this repo's conventions
[11] Patch generation        ← local LLM via llama.cpp
[12] Git archaeology         ← novel: who introduced each bug + when
[13] Patch verification      ← novel: runs test suite to confirm fixes
```

---

## 5 Novel Capabilities

### 1. Taint Analysis
Tracks untrusted user input across function call chains.
Finds SQL injection and command injection vulnerabilities
that pattern-based tools cannot detect.

```python
# This is what taint analysis catches — pattern matching cannot:

def handle_request(request):
    user_id = request.args.get("id")   # ← source (untrusted input)
    return get_user(user_id)           # ← passes to database

def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)           # ← sink (SQL injection)
```

### 2. Statistical Anomaly Detection
Uses IsolationForest to learn what "normal" code looks like
in this specific codebase, then flags functions that are
statistically unusual compared to their neighbours.

### 3. Codebase DNA Fingerprinting
Learns the conventions of the codebase being analyzed:
- Does it use `logging` or `print()`?
- What % of functions have docstrings?
- Are type hints consistent?
- What naming convention is used?

Flags functions that violate the dominant convention.

### 4. Git Archaeology
For every high-confidence issue, finds:
- Which commit introduced it
- Who wrote it
- How many days ago
- What the code looked like before

### 5. Patch Verification
Automatically runs the repository's test suite before and
after applying a suggested patch. Verifies the fix doesn't
break anything before recommending it.

---

## What it detects

| Issue Type | Severity | Detection Method |
|---|---|---|
| Taint flow (injection risk) | High | Inter-procedural analysis |
| Hardcoded secrets | High | Regex heuristic |
| `eval()` / `exec()` usage | High | AST |
| Bare `except: pass` | High | AST |
| Statistical anomaly | Medium | IsolationForest |
| Mutable default arguments | Medium | AST |
| Semantic similarity to bad patterns | Medium | Cosine similarity |
| DNA convention violation | Low | Pattern analysis |
| `assert` in production | Low | AST |
| `print()` in production | Low | AST |
| Division by zero | High | Regex + AST |
| Infinite loop | Medium | AST |

---

## LLM Providers

Switch providers with a single flag — no code changes needed.

```bash
# Local — free, no internet (default)
python analyze_repo.py <url> --provider llamacpp

# Groq — free API tier, very fast
export GROQ_API_KEY=your_key
python analyze_repo.py <url> --provider groq

# Anthropic Claude — highest quality
export ANTHROPIC_API_KEY=your_key
python analyze_repo.py <url> --provider anthropic
```

---

## Dashboard

A FastAPI dashboard reads your analysis results and
displays them visually.

```bash
pip install fastapi uvicorn jinja2
uvicorn dashboard.app:app --reload --port 8000
```

Then open: http://localhost:8000

---

## CLI Reference

```bash
python analyze_repo.py --help

positional arguments:
  repo_url              GitHub repository URL

optional arguments:
  --provider            LLM provider: llamacpp | anthropic | groq | ollama
  --threshold           Similarity detection threshold (default: 0.75)
  --min-confidence      Only report issues above this confidence (e.g. 0.80)
  --output              Results directory (default: analysis_results/)
  --verbose             Show DEBUG logs
  --test                Run quick local test
```

---

## Output Files

| File | Contents |
|---|---|
| `analysis_results/issues.json` | All issues with severity, confidence, line numbers |
| `analysis_results/patches.json` | Fix suggestions with patch source and verification |
| `analysis_results/benchmark.json` | Detection rate, coverage, top files |
| `analysis_results/issues_high_confidence.json` | Issues ≥ 80% confidence only |

---

## Project Structure

```
AI-Code-Analyzer/
├── analyze_repo.py              # 13-stage pipeline entry point
├── config.py                    # Frozen dataclass — all settings
├── exceptions.py                # Typed custom exceptions
├── pipeline_types.py            # Shared TypedDicts
│
├── analysis/                    # Novel detection modules
│   ├── call_graph.py            # Taint analysis
│   ├── anomaly_detector.py      # IsolationForest anomaly detection
│   ├── codebase_dna.py          # Convention fingerprinting
│   ├── git_archaeologist.py     # Bug origin tracing
│   └── patch_verifier.py        # Test-suite patch validation
│
├── ingestion/
│   ├── clone_repo.py            # Git clone, idempotent
│   └── scan_files.py            # Recursive scanner with exclusions
│
├── parsing/
│   ├── extract_function_code.py # AST function extractor
│   └── code_chunker.py          # Sliding window chunker
│
├── embeddings/
│   ├── embed_functions.py       # sentence-transformers wrapper
│   └── similarity_search.py     # Cosine similarity, 15 patterns
│
├── bug_detector/
│   └── detect_patterns.py       # 9 rules + confidence scores
│
├── patch_generator/
│   └── generate_patch.py        # LLM-first, rule-based fallback
│
├── evaluation/
│   └── benchmark.py             # Metrics and terminal report
│
├── dashboard/
│   ├── app.py                   # FastAPI backend
│   └── templates/index.html     # Dashboard UI
│
└── requirements.txt
```

---

## Configuration

All settings live in `config.py` as a frozen dataclass.
Nothing is hardcoded anywhere else.

| Setting | Default | Description |
|---|---|---|
| `llm_provider` | `llamacpp` | Active LLM provider |
| `embedding_model` | `all-MiniLM-L6-v2` | Embedding model name |
| `similarity_threshold` | `0.75` | Semantic detection cutoff |
| `llamacpp_url` | `http://127.0.0.1:8080` | llama.cpp server |
| `llamacpp_model` | `qwen2.5-coder-14b` | Model name |
| `max_chunk_lines` | `80` | Max lines per chunk |
| `min_chunk_lines` | `1` | Min lines to keep a chunk |
| `contamination` | `0.05` | Anomaly detector sensitivity |

---

## Local LLM Setup (llama.cpp)

```bash
# Clone and build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release

# Download model (8GB — Qwen 14B recommended)
huggingface-cli download \
  Qwen/Qwen2.5-Coder-14B-Instruct-GGUF \
  qwen2.5-coder-14b-instruct-q4_k_m.gguf \
  --local-dir models/

# Start server
./build/bin/llama-server \
  --model models/qwen2.5-coder-14b-q4.gguf \
  --port 8080 \
  --n-gpu-layers 99 \
  --ctx-size 2048
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Anomaly detection | scikit-learn IsolationForest |
| Local LLM | llama.cpp + Qwen2.5-Coder-14B |
| GPU acceleration | Apple Metal (M4) |
| Static analysis | Python AST module |
| Dashboard | FastAPI + Chart.js |
| Vector similarity | numpy cosine similarity |

---

## License

MIT