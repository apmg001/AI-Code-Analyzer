# AI Code Analyzer

An automated pipeline that clones any Python repository and produces a
structured report of bugs, security risks, and fix suggestions — using
both rule-based static analysis and AI-powered semantic detection.

---

## How it works

```
GitHub URL
    │
    ▼
Clone repo → Scan files → Extract functions → Chunk code
                                                    │
                                                    ▼
                                           Generate embeddings
                                                    │
                                          ┌─────────┴──────────┐
                                          │                    │
                                   Rule-based              Semantic
                                   detection             similarity
                                   (8 rules)             (15 patterns)
                                          │                    │
                                          └─────────┬──────────┘
                                                    │
                                             LLM patch gen
                                          (Claude/gpt API / fallback)
                                                    │
                                                    ▼
                                        issues.json / patches.json
                                            benchmark.json
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key (optional — enables LLM patches)
export ANTHROPIC_API_KEY=your_key_here

# 3. Run
python analyze_repo.py https://github.com/pallets/flask
```

Results are saved to `analysis_results/`.

---

## What it detects

| Rule | Severity |
|---|---|
| Division by zero | High |
| `eval()` usage | High |
| `exec()` usage | High |
| Hardcoded secrets | High |
| Bare `except: pass` | High |
| Mutable default arguments | Medium |
| `while True` with no exit | Medium |
| `assert` in production code | Low |
| Semantic similarity to 15 known anti-patterns | Medium |

---

## Output files

| File | Contents |
|---|---|
| `analysis_results/issues.json` | All detected issues with severity + line numbers |
| `analysis_results/patches.json` | Fix suggestions (LLM or rule-based) |
| `analysis_results/benchmark.json` | Detection rate, coverage metrics, top files |

---

## Project structure

```
AI-Code-Analyzer/
├── config.py                    # Single source of truth for all settings
├── exceptions.py                # Typed custom exceptions
├── types.py                     # Shared TypedDicts
├── analyze_repo.py              # Pipeline entry point
├── ingestion/
│   ├── clone_repo.py            # Git clone with idempotency
│   └── scan_files.py            # Recursive file scanner with exclusions
├── parsing/
│   ├── extract_function_code.py # AST-based function extractor
│   └── code_chunker.py          # Function → chunk splitter
├── embeddings/
│   ├── embed_functions.py       # sentence-transformers wrapper
│   └── similarity_search.py     # Cosine similarity bug detector
├── bug_detector/
│   └── detect_patterns.py       # 8 rule-based + semantic detection
├── patch_generator/
│   └── generate_patch.py        # LLM-first, rule-based fallback
├── evaluation/
│   └── benchmark.py             # Metrics and report generation
└── requirements.txt
```

---

## Configuration

All settings are in `config.py`. Key options:

| Setting | Default | Description |
|---|---|---|
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `similarity_threshold` | `0.75` | Cosine similarity cutoff |
| `llm_model` | `claude-sonnet-4-6` | Model for patch generation |
| `max_chunk_lines` | `80` | Max lines per chunk |

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | No | Enables LLM-powered patches. Falls back to rule-based if unset. |

---

## License

MIT
