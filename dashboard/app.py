# dashboard/app.py

"""
FastAPI dashboard backend for AI Code Analyzer.

Reads analysis_results/ JSON files and serves them
to the dashboard template.

Also exposes endpoints to trigger a new analysis
run and stream progress logs back to the UI.

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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates


# ------------------------------------------------------------------
# App + paths
# ------------------------------------------------------------------

app = FastAPI(
    title="AI Code Analyzer Dashboard",
    version="1.0.0",
)

# dashboard/app.py lives inside the dashboard/ folder
# so BASE_DIR is the dashboard/ folder itself
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent                        # project root
RESULTS_DIR = PROJECT_DIR / "analysis_results"      # where JSONs live
TEMPLATES   = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ------------------------------------------------------------------
# Analysis state
# Simple in-memory state — one analysis at a time.
# For production you'd use Redis or a proper task queue.
# ------------------------------------------------------------------

_running: bool       = False
_log:     List[str]  = []


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------

def _load(filename: str) -> Any:
    """
    Load a JSON file from analysis_results/.
    Returns None silently if the file does not exist yet.
    """
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _build_dashboard_data() -> Dict[str, Any]:
    """
    Aggregate all result files into one dict the template can use.

    This is the single place that knows the JSON schemas — the
    template just reads values, it never touches files directly.
    """

    issues    = _load("issues.json")    or []
    patches   = _load("patches.json")   or []
    benchmark = _load("benchmark.json") or {}
    high_conf = _load("issues_high_confidence.json") or []

    if not issues:
        return {"has_results": False}

    # ── Severity breakdown ─────────────────────────────────────
    severity_counts = Counter(
        i.get("severity", "unknown") for i in issues
    )

    # ── Issue type breakdown ───────────────────────────────────
    type_counts = Counter(
        i.get("type", "unknown") for i in issues
    )

    # ── Top files by issue count ───────────────────────────────
    file_counts: Dict[str, int] = defaultdict(int)
    for issue in issues:
        # Use just the filename — not the full path
        fname = issue.get("file", "unknown").split("/")[-1]
        file_counts[fname] += 1

    top_files = sorted(file_counts.items(), key=lambda x: -x[1])[:8]

    # ── Patch source breakdown ─────────────────────────────────
    # patch_source can be "llamacpp", "llm", "rule_based", "none"
    llm_patches  = sum(
        1 for p in patches
        if p.get("patch_source") not in ("rule_based", "none", None)
    )
    rule_patches = sum(
        1 for p in patches
        if p.get("patch_source") == "rule_based"
    )

    # ── Patch coverage ─────────────────────────────────────────
    patched  = sum(1 for p in patches if p.get("patch_source") != "none")
    coverage = round(patched / len(issues) * 100, 1) if issues else 0.0

    # ── Confidence distribution ────────────────────────────────
    conf_buckets = {"≥90%": 0, "70–89%": 0, "50–69%": 0, "<50%": 0}
    for issue in issues:
        c = issue.get("confidence", 0)
        if   c >= 0.90: conf_buckets["≥90%"]    += 1
        elif c >= 0.70: conf_buckets["70–89%"]  += 1
        elif c >= 0.50: conf_buckets["50–69%"]  += 1
        else:           conf_buckets["<50%"]     += 1

    # ── Novel detection counts ─────────────────────────────────
    taint_count   = sum(1 for i in issues if i.get("type") == "taint_flow")
    anomaly_count = sum(1 for i in issues if i.get("type") == "statistical_anomaly")
    dna_count     = sum(1 for i in issues if i.get("type") == "dna_violation")

    # ── Issues sorted by confidence for the feed ──────────────
    recent_issues = sorted(
        issues,
        key=lambda x: x.get("confidence", 0),
        reverse=True,
    )[:20]

    return {
        "has_results":    True,
        "total_issues":   len(issues),
        "total_patches":  len(patches),
        "high_confidence": len(high_conf),
        "coverage_pct":   coverage,
        "llm_patches":    llm_patches,
        "rule_patches":   rule_patches,
        "taint_count":    taint_count,
        "anomaly_count":  anomaly_count,
        "dna_count":      dna_count,
        "severity":       dict(severity_counts),
        "issue_types":    dict(type_counts),
        "top_files":      top_files,
        "conf_buckets":   conf_buckets,
        "recent_issues":  recent_issues,
        "patches":        patches[:25],
        "benchmark":      benchmark,
    }


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the main dashboard page.
    Builds data fresh on every request so the page always
    reflects the latest analysis_results/ files.
    """
    data = _build_dashboard_data()
    return TEMPLATES.TemplateResponse(
        "index.html",
        {"request": request, "data": data},
    )


@app.get("/api/summary")
async def api_summary():
    """
    JSON version of the dashboard data.
    Used by the auto-refresh JS in the template.
    """
    return JSONResponse(_build_dashboard_data())


@app.get("/api/issues")
async def api_issues():
    """Full issues list — useful for debugging or external tools."""
    return JSONResponse(_load("issues.json") or [])


@app.get("/api/patches")
async def api_patches():
    """Full patches list."""
    return JSONResponse(_load("patches.json") or [])


@app.post("/api/analyze")
async def api_analyze(
    background_tasks: BackgroundTasks,
    repo_url: str = "",
):
    """
    Trigger a new analysis pipeline run in the background.

    Returns immediately with status "started".
    Poll /api/status to get progress logs.

    One analysis at a time — returns 409 if already running.
    """
    global _running, _log

    if _running:
        return JSONResponse(
            {"status": "already_running",
             "message": "An analysis is already in progress."},
            status_code=409,
        )

    if not repo_url.strip():
        return JSONResponse(
            {"status": "error", "message": "repo_url is required"},
            status_code=400,
        )

    _log     = [f"Starting: {repo_url}"]
    background_tasks.add_task(_run_pipeline, repo_url.strip())

    return JSONResponse({"status": "started", "repo_url": repo_url})


@app.get("/api/status")
async def api_status():
    """
    Return current analysis status and the last 20 log lines.
    The frontend polls this every 1.5 seconds when a run is active.
    """
    return JSONResponse({
        "running": _running,
        "log":     _log[-20:],
    })


# ------------------------------------------------------------------
# Background task — runs the pipeline
# ------------------------------------------------------------------

def _run_pipeline(repo_url: str) -> None:
    """
    Spawn analyze_repo.py as a subprocess and capture its output
    into _log so the frontend can stream it in real time.

    Runs in a FastAPI background task — does not block the server.
    """
    global _running, _log

    _running = True

    try:
        analyzer_path = PROJECT_DIR / "analyze_repo.py"

        process = subprocess.Popen(
            [sys.executable, str(analyzer_path), repo_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout
            text=True,
            cwd=str(PROJECT_DIR),
        )

        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if line:
                _log.append(line)

        process.wait()

        if process.returncode == 0:
            _log.append("✓ Analysis complete")
        else:
            _log.append(f"✗ Pipeline exited with code {process.returncode}")

    except FileNotFoundError:
        _log.append("✗ analyze_repo.py not found — check project structure")

    except Exception as exc:
        _log.append(f"✗ Unexpected error: {exc}")

    finally:
        _running = False