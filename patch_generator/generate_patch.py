"""
Add this to the top of generate_patch.py
"""
import os
import json
import urllib.request


def _call_llm_for_patch(issue: Dict, chunk: Dict) -> str:
    """
    Call Claude API to generate a smart patch suggestion.
    Falls back to rule-based if API fails.
    """

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        return None  # fallback to rule-based

    prompt = f"""You are a senior Python engineer reviewing buggy code.

Bug type detected: {issue['type']}
Severity: {issue['severity']}
Issue message: {issue['message']}

Buggy code:
```python
{chunk['code']}
```

Give a concise, specific fix for this exact code.
Return ONLY the fixed code and a 1-line explanation.
Do not explain what the bug is — just fix it."""

    payload = json.dumps({
        "model": "claude-opus-4-6",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]

    except Exception:
        return None  # fallback gracefully