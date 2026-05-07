"""Call the Ollama API at localhost:11434 using httpx.

Performs a preflight health_check() before any generation:
  - verifies Ollama is reachable
  - verifies the requested model is pulled
"""

import sys
import time
from dataclasses import dataclass

import httpx


OLLAMA_BASE    = "http://localhost:11434"
OLLAMA_GENERATE = f"{OLLAMA_BASE}/api/generate"
OLLAMA_TAGS     = f"{OLLAMA_BASE}/api/tags"
DEFAULT_MODEL  = "gemma3:4b"
TIMEOUT        = httpx.Timeout(120.0)   # generation can be slow
MAX_TOKENS     = 2048                    # hard cap — prevents verbose models inflating latency


@dataclass
class LLMResponse:
    text: str
    latency_ms: float


def health_check(model: str = DEFAULT_MODEL) -> None:
    """Exit with a clear message if Ollama isn't running or model isn't pulled."""
    try:
        with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
            resp = client.get(OLLAMA_TAGS)
            resp.raise_for_status()
    except httpx.ConnectError:
        print(
            f"ERROR: Cannot reach Ollama at {OLLAMA_BASE}\n"
            "  → Start it with:  ollama serve",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Ollama health check failed — {exc}", file=sys.stderr)
        sys.exit(1)

    available = [m["name"] for m in resp.json().get("models", [])]
    base_names = [n.split(":")[0] for n in available]
    if model not in available and model not in base_names:
        print(
            f"ERROR: Model '{model}' is not pulled.\n"
            f"  Available: {available or ['(none)']}\n"
            f"  → Pull it with:  ollama pull {model}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"✓ Ollama reachable — model '{model}' is ready.\n")


def generate(prompt: str, model: str = DEFAULT_MODEL) -> LLMResponse:
    """Send *prompt* to Ollama and return the response text + latency."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": MAX_TOKENS},  # hard token cap
    }

    t0 = time.perf_counter()
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(OLLAMA_GENERATE, json=payload)
        resp.raise_for_status()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return LLMResponse(
        text=resp.json().get("response", ""),
        latency_ms=round(latency_ms, 2),
    )
