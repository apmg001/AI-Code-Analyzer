# analysis/call_graph.py

"""
Module: call_graph

Responsibility
--------------
Build a function call graph across the entire codebase and
perform taint analysis — tracking where untrusted user input
flows into dangerous operations.

Why this matters
----------------
Pattern-based tools (pylint, bandit) analyze each function
in isolation. They cannot detect:

    def get_user(user_id):
        query = f"SELECT * FROM users WHERE id = {user_id}"
        return db.execute(query)           # sink

    def handle_request(request):
        user_id = request.args.get("id")  # source
        return get_user(user_id)          # user input flows here

The vulnerability only exists because of the connection between
the two functions. Taint analysis finds exactly this.

Design notes
------------
- Sources: functions that receive external/untrusted input
- Sinks:   functions that perform dangerous operations
- Taint:   DFS from every source looking for a path to any sink
- One CallGraphBuilder per pipeline run — built once, queried many times
"""

import ast
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Sources — entry points for untrusted data
# ------------------------------------------------------------------

SOURCE_PATTERNS = {
    # Web frameworks
    "request.args",
    "request.form",
    "request.json",
    "request.data",
    "request.get_json",
    "request.files",
    "request.cookies",
    "request.headers",

    # Standard input
    "input(",
    "sys.argv",
    "os.environ",
    "os.getenv",

    # Network
    "socket.recv",
    "urllib.request",
    "requests.get",
    "requests.post",

    # File system (untrusted files)
    "open(",
    "Path(",
}


# ------------------------------------------------------------------
# Sinks — dangerous operations
# ------------------------------------------------------------------

SINK_PATTERNS = {
    # Code execution
    "eval(",
    "exec(",
    "compile(",

    # Shell execution
    "os.system(",
    "os.popen(",
    "subprocess.call(",
    "subprocess.run(",
    "subprocess.Popen(",

    # Database
    "execute(",
    "executemany(",
    "raw(",
    "RawSQL(",

    # File operations
    "open(",
    "write(",
    "pickle.loads(",
    "pickle.load(",

    # Serialization
    "yaml.load(",
    "json.loads(",

    # Template rendering
    "render_template_string(",
    "Template(",
    "Markup(",
}


# ------------------------------------------------------------------
# Call graph builder
# ------------------------------------------------------------------

class CallGraphBuilder:
    """
    Builds a directed call graph across all Python files
    and identifies taint flows from sources to sinks.

    Usage
    -----
    builder = CallGraphBuilder()
    builder.build(python_files)
    vulnerabilities = builder.find_source_to_sink_paths()
    """

    def __init__(self):
        # func_name → list of functions it calls
        self._call_graph: Dict[str, List[str]] = defaultdict(list)

        # func_name → source file path
        self._func_files: Dict[str, str] = {}

        # func_name → start line number
        self._func_lines: Dict[str, int] = {}

        # func_name → raw source code
        self._func_code: Dict[str, str] = {}

        # Functions that touch external input
        self.sources: Set[str] = set()

        # Functions that perform dangerous operations
        self.sinks: Set[str] = set()

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self, python_files: List[Path]) -> "CallGraphBuilder":
        """
        Analyze all files and build the call graph.
        Returns self for chaining.
        """
        for file_path in python_files:
            self._analyze_file(file_path)

        logger.info(
            "Call graph built — %d functions, %d sources, %d sinks",
            len(self._call_graph),
            len(self.sources),
            len(self.sinks),
        )

        return self

    def _analyze_file(self, file_path: Path) -> None:
        """Parse one file and extract call relationships."""
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree   = ast.parse(source)
        except Exception as exc:
            logger.debug("Skipping %s: %s", file_path, exc)
            return

        lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            func_name = node.name
            start     = node.lineno
            end       = getattr(node, "end_lineno", node.lineno)
            func_code = "\n".join(lines[start - 1 : end])

            # Register function metadata
            self._func_files[func_name] = str(file_path)
            self._func_lines[func_name] = start
            self._func_code[func_name]  = func_code

            # Classify as source or sink
            if any(p in func_code for p in SOURCE_PATTERNS):
                self.sources.add(func_name)

            if any(p in func_code for p in SINK_PATTERNS):
                self.sinks.add(func_name)

            # Record outgoing calls
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue

                callee = self._extract_callee_name(child)
                if callee:
                    self._call_graph[func_name].append(callee)

    @staticmethod
    def _extract_callee_name(call_node: ast.Call) -> Optional[str]:
        """Extract the function name from a call node."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        if isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
        return None

    # ------------------------------------------------------------------
    # Taint analysis
    # ------------------------------------------------------------------

    def find_source_to_sink_paths(self) -> List[Dict]:
        """
        Find all paths from user-input sources to dangerous sinks.

        Returns a list of vulnerability dicts — one per taint flow found.
        These represent the highest-value findings in the tool because
        they cannot be detected by any pattern-based approach.
        """
        vulnerabilities = []

        for source in self.sources:
            path = self._dfs(source, visited=set(), path=[])

            if path:
                sink      = path[-1]
                sink_file = self._func_files.get(sink, "unknown")
                sink_line = self._func_lines.get(sink, 0)

                vulnerabilities.append({
                    "type":          "taint_flow",
                    "severity":      "high",
                    "confidence":    0.80,
                    "function":      source,
                    "file":          self._func_files.get(source, "unknown"),
                    "line_number":   self._func_lines.get(source, 0),
                    "code_snippet":  self._func_code.get(source, "")[:120],
                    "chunk_id":      f"taint_{source}_{sink}",
                    "similarity_score": None,
                    "taint_path":    " → ".join(path),
                    "sink_function": sink,
                    "sink_file":     sink_file,
                    "sink_line":     sink_line,
                    "message": (
                        f"User-controlled input enters at '{source}' "
                        f"and reaches dangerous operation in '{sink}' "
                        f"via {len(path) - 1} call(s): {' → '.join(path)}. "
                        f"Potential injection vulnerability — verify all "
                        f"input is sanitized before reaching '{sink}'."
                    ),
                })

        logger.info("Taint analysis complete — %d flows found", len(vulnerabilities))
        return vulnerabilities

    def _dfs(
        self,
        func:    str,
        visited: Set[str],
        path:    List[str],
    ) -> List[str]:
        """
        Depth-first search from a source function looking for a sink.

        Returns the full path if a sink is reachable, empty list otherwise.
        visited prevents infinite loops in recursive call graphs.
        """
        if func in visited:
            return []

        visited.add(func)
        current_path = path + [func]

        # Found a sink — return the path (exclude source-only paths)
        if func in self.sinks and len(current_path) > 1:
            return current_path

        # Recurse into callees
        for callee in self._call_graph.get(func, []):
            result = self._dfs(callee, visited.copy(), current_path)
            if result:
                return result

        return []
