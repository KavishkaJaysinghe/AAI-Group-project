"""
llm_explain.py — TeleGuard Phase 4: LLM anomaly explanation engine.

Given a flagged connection's features, ask an LLM to explain — in plain English —
WHY it looks anomalous, for an analyst reading the dashboard.

Primary provider: ANTHROPIC (Claude Messages API), as requested for this phase.
Secondary provider: Google Gemini (free tier) — kept so the pipeline still runs
for $0 if you don't have Anthropic credits. Pick the provider with LLM_PROVIDER
in .env ("anthropic" or "gemini").

Keys are read from .env via python-dotenv. Never hard-code a key.
If no key / package is available, explain_anomaly() returns a graceful fallback
string instead of raising, so Stages 1-3 keep working without any API access.

Run the standalone demo:
    python src/llm_explain.py

------------------------------------------------------------------------------
PROMPT TEMPLATE (chain-of-thought, kept short to keep token usage modest)
------------------------------------------------------------------------------
SYSTEM:
    You are a telecom security analyst. Explain network anomalies clearly and
    concisely for fellow analysts.

USER:
    A network connection was flagged as a possible anomaly by our detector.
    Here are its key features:
      - <feature>: <value>
      ...

    Think step by step:
      1. Identify which feature values look unusual or suspicious, and why.
      2. Consider what kind of attack or misbehaviour they might indicate.
      3. Then give a concise plain-English explanation (2-3 sentences) an
         analyst can act on, ending with the single most likely category
         (DoS, probe/scan, brute-force/R2L, or "unsure").
------------------------------------------------------------------------------
"""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path

# --- Load .env so ANTHROPIC_API_KEY / GEMINI_API_KEY are available ---------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    # python-dotenv not installed: fall back to whatever is already in os.environ
    pass

# Model defaults (override per provider in .env).
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"   # cheaper option: claude-haiku-4-5
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"     # free tier
MAX_TOKENS = 512          # modest cap — these explanations are short
REQUEST_TIMEOUT = 30.0    # per-request timeout passed to the SDK (Anthropic)
OVERALL_TIMEOUT = 35.0    # hard cap so the dashboard never hangs on a slow API

# Single shared worker so a slow/blocked API call can be abandoned without
# blocking the caller (Streamlit) — the UI shows the fallback instead of spinning.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")

SYSTEM_PROMPT = (
    "You are a telecom security analyst. Explain network anomalies clearly and "
    "concisely for fellow analysts. Be precise and avoid unnecessary jargon."
)


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def build_user_prompt(features: dict) -> str:
    """Render the flagged connection's features into the user message."""
    feature_lines = "\n".join(f"  - {k}: {v}" for k, v in features.items())
    return (
        "A network connection was flagged as a possible anomaly by our "
        "detector.\nHere are its key features:\n"
        f"{feature_lines}\n\n"
        "Think step by step:\n"
        "  1. Identify which feature values look unusual or suspicious, and why.\n"
        "  2. Consider what kind of attack or misbehaviour they might indicate.\n"
        "  3. Then give a concise plain-English explanation (2-3 sentences) an "
        "analyst can act on, ending with the single most likely category "
        "(DoS, probe/scan, brute-force/R2L, or \"unsure\").\n"
    )


def _fallback(reason: str) -> str:
    """Graceful message used whenever the LLM call can't be made."""
    return (
        f"[Automated explanation unavailable: {reason}. "
        f"Please review the flagged feature values manually.]"
    )


# --------------------------------------------------------------------------- #
# Provider: Anthropic (primary)
# --------------------------------------------------------------------------- #
def _explain_with_anthropic(features: dict) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback("ANTHROPIC_API_KEY not set in .env")

    try:
        import anthropic
    except ImportError:
        return _fallback("anthropic package not installed (pip install anthropic)")

    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.with_options(timeout=REQUEST_TIMEOUT).messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(features)}],
        )
        # response.content is a list of blocks; take the text block(s).
        text = "".join(b.text for b in response.content if b.type == "text")
        return text.strip() or _fallback("empty response from model")
    except anthropic.AuthenticationError:
        return _fallback("invalid ANTHROPIC_API_KEY")
    except anthropic.APITimeoutError:
        return _fallback("Anthropic request timed out")
    except anthropic.APIError as exc:
        return _fallback(f"Anthropic API error: {exc}")
    except Exception as exc:  # last-resort safety net
        return _fallback(f"unexpected error: {exc}")


# --------------------------------------------------------------------------- #
# Provider: Google Gemini (free-tier secondary)
# --------------------------------------------------------------------------- #
def _explain_with_gemini(features: dict) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback("GEMINI_API_KEY not set in .env")

    try:
        from google import genai
    except ImportError:
        return _fallback("google-genai not installed (pip install google-genai)")

    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    # Gemini has no separate system role here; prepend it to the prompt.
    prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(features)}"

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        return (response.text or "").strip() or _fallback("empty response from model")
    except Exception as exc:  # google-genai raises various provider errors
        return _fallback(f"Gemini error: {exc}")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def explain_anomaly(features: dict) -> str:
    """Return a plain-English explanation of why `features` looks anomalous.

    Dispatches to the provider named by LLM_PROVIDER in .env
    ("anthropic" by default, or "gemini"). Never raises — on any failure it
    returns a fallback string so the dashboard stays usable.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    fn = _explain_with_gemini if provider == "gemini" else _explain_with_anthropic

    # Run with a hard timeout so a slow/blocked network never freezes the UI.
    future = _EXECUTOR.submit(fn, features)
    try:
        return future.result(timeout=OVERALL_TIMEOUT)
    except concurrent.futures.TimeoutError:
        return _fallback(
            f"{provider} did not respond within {OVERALL_TIMEOUT:.0f}s "
            f"(check your network / API key)"
        )
    except Exception as exc:  # pragma: no cover - last-resort safety net
        return _fallback(f"unexpected error: {exc}")


# --------------------------------------------------------------------------- #
# Standalone demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Sample anomalous record (rejected connection, no data, repeated failed logins).
    sample_features = {
        "protocol_type": "tcp",
        "service": "private",
        "flag": "REJ",
        "src_bytes": 0,
        "dst_bytes": 0,
        "num_failed_logins": 5,
        "logged_in": 0,
        "count": 120,
        "serror_rate": 1.0,
    }

    print("Flagged connection features:")
    for key, value in sample_features.items():
        print(f"  {key}: {value}")

    print(f"\nProvider: {os.getenv('LLM_PROVIDER', 'anthropic')}")
    print("\n--- LLM explanation ---")
    print(explain_anomaly(sample_features))
