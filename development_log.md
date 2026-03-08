# AI Code Analyzer – Development Log

## Day 1

### Project Setup
- Created project directory for the AI Code Analyzer system.
- Initialized a Git repository.
- Connected the project to the GitHub repository:
  https://github.com/apmg001/AI-Code-Analyzer
- Created a Python virtual environment (CodeCorrector).
- Installed required libraries for the project.

### Repository Ingestion Module
Implemented the first component of the system: repository cloning.

File:
ingestion/clone_repo.py

Functionality:
- Accepts a GitHub repository URL.
- Clones the repository locally if it does not already exist.
- Returns the local path to the repository.

Test Result:
Successfully cloned the Flask repository for testing.

### File Scanning Module
Implemented scanning of Python files within a repository.

File:
ingestion/scan_files.py

Functionality:
- Walks through the repository directory.
- Detects all `.py` files.
- Returns a list of Python file paths.

Test Result:
Scanned the Flask repository and detected 83 Python files.

### Current Pipeline
The current system pipeline is:

GitHub repository URL
→ clone repository
→ scan repository for Python files
→ list of Python files ready for analysis

### Next Steps
- Implement code parsing module.
- Extract functions from Python files using AST.
- Begin building structured code chunks for later analysis.