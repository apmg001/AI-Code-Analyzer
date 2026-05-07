# dashboard/app.py

"""
FastAPI dashboard backend for AI Code Analyzer.

Reads analysis_results/ JSON files and serves them
to the dashboard template.

Also exposes endpoints to:
- Trigger a new analysis run
- Start / stop the llama.cpp server
- Stream progress logs back to the UI

Usage
-----
    pip install fastapi uvicorn jinja2

    # From your project root:
    uvicorn dashboard.app:app --reload --port 8000

    Then open: http://localhost:8000
"""

import json
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates


# ------------------------------------------------------------------
# App + paths
# ------------------------------------------------------------------

app = FastAPI(title="AI Code Analyzer Dashboard", version="1.0.0")

BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
RESULTS_DIR = PROJECT_DIR / "analysis_results"
TEMPLATES   = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# llama.cpp server binary and model — auto-detected from project
LLAMA_SERVER = PROJECT_DIR / "llama.cpp" / "build" / "bin" / "llama-server"
LLAMA_MODEL  = PROJECT_DIR / "llama.cpp" / "models" / "qwen2.5-coder-14b-q4.gguf"


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

_analysis_running: bool      = False
_analysis_log:     List[str] = []

_llama_process:    Optional[subprocess.Popen] = None
_llama_log:        List[str] = []


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------

def _load(filename: str) -> Any:
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _llama_server_running() -> bool:
    """Check if the llama.cpp server is actually responding."""
    try:
        urllib.request.urlopen(
            "http://127.0.0.1:8080/v1/models", timeout=2
        )
        return True
    except Exception:
        return False


def _build_dashboard_data() -> Dict[str, Any]:
    issues    = _load("issues.json")    or []
    patches   = _load("patches.json")   or []
    benchmark = _load("benchmark.json") or {}
    high_conf = _load("issues_high_confidence.json") or []

    if not issues:
        return {
            "has_results":    False,
            "llama_running":  _llama_server_running(),
        }

    severity_counts = Counter(i.get("severity", "unknown") for i in issues)
    type_counts     = Counter(i.get("type", "unknown")     for i in issues)

    file_counts: Dict[str, int] = defaultdict(int)
    for issue in issues:
        fname = issue.get("file", "unknown").split("/")[-1]
        file_counts[fname] += 1

    top_files = sorted(file_counts.items(), key=lambda x: -x[1])[:8]

    llm_patches  = sum(
        1 for p in patches
        if p.get("patch_source") not in ("rule_based", "none", None)
    )
    rule_patches = sum(
        1 for p in patches
        if p.get("patch_source") == "rule_based"
    )

    patched  = sum(1 for p in patches if p.get("patch_source") != "none")
    coverage = round(patched / len(issues) * 100, 1) if issues else 0.0

    conf_buckets = {">=90%": 0, "70-89%": 0, "50-69%": 0, "<50%": 0}
    for issue in issues:
        c = issue.get("confidence", 0)
        if   c >= 0.90: conf_buckets[">=90%"]  += 1
        elif c >= 0.70: conf_buckets["70-89%"] += 1
        elif c >= 0.50: conf_buckets["50-69%"] += 1
        else:           conf_buckets["<50%"]    += 1

    return {
        "has_results":     True,
        "total_issues":    len(issues),
        "total_patches":   len(patches),
        "high_confidence": len(high_conf),
        "coverage_pct":    coverage,
        "llm_patches":     llm_patches,
        "rule_patches":    rule_patches,
        "taint_count":     sum(1 for i in issues if i.get("type") == "taint_flow"),
        "anomaly_count":   sum(1 for i in issues if i.get("type") == "statistical_anomaly"),
        "dna_count":       sum(1 for i in issues if i.get("type") == "dna_violation"),
        "severity":        dict(severity_counts),
        "issue_types":     dict(type_counts),
        "top_files":       top_files,
        "conf_buckets":    conf_buckets,
        "recent_issues":   sorted(issues, key=lambda x: x.get("confidence", 0), reverse=True)[:20],
        "patches":         patches[:25],
        "benchmark":       benchmark,
        "llama_running":   _llama_server_running(),
    }


# ------------------------------------------------------------------
# Routes — dashboard
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    data = _build_dashboard_data()
    return TEMPLATES.TemplateResponse("index.html", {"request": request, "data": data})


@app.get("/api/summary")
async def api_summary():
    return JSONResponse(_build_dashboard_data())


@app.get("/api/issues")
async def api_issues():
    return JSONResponse(_load("issues.json") or [])


@app.get("/api/patches")
async def api_patches():
    return JSONResponse(_load("patches.json") or [])


# ------------------------------------------------------------------
# Routes — analysis
# ------------------------------------------------------------------

@app.post("/api/analyze")
async def api_analyze(background_tasks: BackgroundTasks, repo_url: str = ""):
    global _analysis_running, _analysis_log

    if _analysis_running:
        return JSONResponse({"status": "already_running"}, status_code=409)

    if not repo_url.strip():
        return JSONResponse({"status": "error", "message": "repo_url required"}, status_code=400)

    _analysis_log = [f"Starting: {repo_url}"]
    background_tasks.add_task(_run_pipeline, repo_url.strip())
    return JSONResponse({"status": "started", "repo_url": repo_url})


@app.get("/api/status")
async def api_status():
    return JSONResponse({
        "running":       _analysis_running,
        "log":           _analysis_log[-20:],
        "llama_running": _llama_server_running(),
    })


# ------------------------------------------------------------------
# Routes — llama.cpp server control
# ------------------------------------------------------------------

@app.post("/api/llama/start")
async def llama_start(background_tasks: BackgroundTasks):
    """
    Start the llama.cpp server as a background process.
    Returns immediately — poll /api/llama/status for readiness.
    """
    global _llama_process, _llama_log

    if _llama_server_running():
        return JSONResponse({"status": "already_running"})

    if not LLAMA_SERVER.exists():
        return JSONResponse(
            {"status": "error",
             "message": f"llama-server not found at {LLAMA_SERVER}"},
            status_code=404,
        )

    if not LLAMA_MODEL.exists():
        return JSONResponse(
            {"status": "error",
             "message": f"Model not found at {LLAMA_MODEL}"},
            status_code=404,
        )

    _llama_log = ["Starting llama.cpp server…"]
    background_tasks.add_task(_start_llama_server)
    return JSONResponse({"status": "starting"})


@app.post("/api/llama/stop")
async def llama_stop():
    """Terminate the llama.cpp server process."""
    global _llama_process

    if _llama_process is None:
        return JSONResponse({"status": "not_running"})

    try:
        _llama_process.terminate()
        _llama_process.wait(timeout=5)
        _llama_process = None
        return JSONResponse({"status": "stopped"})
    except Exception as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)


@app.get("/api/llama/status")
async def llama_status():
    """Return whether the server is running and recent log lines."""
    return JSONResponse({
        "running": _llama_server_running(),
        "log":     _llama_log[-15:],
    })


# ------------------------------------------------------------------
# Background tasks
# ------------------------------------------------------------------

def _start_llama_server() -> None:
    """Spawn llama-server and stream its output into _llama_log."""
    global _llama_process, _llama_log

    try:
        _llama_process = subprocess.Popen(
            [
                str(LLAMA_SERVER),
                "--model",         str(LLAMA_MODEL),
                "--port",          "8080",
                "--ctx-size",      "2048",
                "--n-gpu-layers",  "99",
                "--threads",       "8",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        for line in _llama_process.stdout:
            line = line.rstrip()
            if line:
                _llama_log.append(line)
                # Stop reading once server is ready
                if "server is listening" in line:
                    _llama_log.append("✓ Server ready at http://127.0.0.1:8080")
                    break

    except Exception as exc:
        _llama_log.append(f"✗ Failed to start server: {exc}")


def _run_pipeline(repo_url: str) -> None:
    """Run analyze_repo.py as a subprocess and capture logs."""
    global _analysis_running, _analysis_log

    _analysis_running = True

    try:
        analyzer = PROJECT_DIR / "analyze_repo.py"
        proc = subprocess.Popen(
            [sys.executable, str(analyzer), repo_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_DIR),
        )

        for line in proc.stdout:
            line = line.rstrip()
            if line:
                _analysis_log.append(line)

        proc.wait()
        _analysis_log.append(
            "✓ Analysis complete" if proc.returncode == 0
            else f"✗ Pipeline exited with code {proc.returncode}"
        )

    except Exception as exc:
        _analysis_log.append(f"✗ Error: {exc}")
    finally:
        _analysis_running = False