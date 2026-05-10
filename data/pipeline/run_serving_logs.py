"""
run_serving_logs.py — Run LMSYS prompts through Ollama, record actual Gemma output tokens

Reads:   data/lmsys_prompts.jsonl
Writes:  data/serving_logs.jsonl  (checkpoint-friendly — safe to resume)

Each output record:
  {
    "prompt":              str,
    "response":            str,
    "response_token_count": int,   ← actual Gemma output token count
    "conversation_id":     str,
    "model":               str,
    "elapsed_sec":         float
  }

Resume behaviour: already-processed conversation_ids are skipped on restart.
Ctrl-C saves cleanly — re-run to continue.

Usage:
    python data/pipeline/run_serving_logs.py \\
        [--prompts data/lmsys_prompts.jsonl] \\
        [--out     data/serving_logs.jsonl] \\
        [--model   gemma3:4b] \\
        [--url     http://localhost:11434] \\
        [--timeout 300]
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

from transformers import AutoTokenizer

# ── tokenizer ────────────────────────────────────────────────────────────────
_TOKENIZER = None

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    print("Loading bert-base-uncased tokenizer...")
    _TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("  ✓ Tokenizer loaded")
    return _TOKENIZER

def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


# ── Ollama API call ───────────────────────────────────────────────────────────
def call_ollama(prompt: str, model: str, base_url: str, timeout: int) -> str:
    """Send prompt to Ollama, return full response text."""
    url = base_url.rstrip("/") + "/api/generate"
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "")


# ── Main ─────────────────────────────────────────────────────────────────────
def main(prompts_path, out_path, model, base_url, timeout):
    # load prompts
    if not os.path.exists(prompts_path):
        print(f"ERROR: {prompts_path} not found. Run collect_lmsys.py first.", file=sys.stderr)
        sys.exit(1)

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(prompts):,} prompts from {prompts_path}")

    # load already-done conversation_ids for resume
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec.get("conversation_id", ""))
                except json.JSONDecodeError:
                    pass
        print(f"Resuming — {len(done_ids):,} already done, "
              f"{len(prompts) - len(done_ids):,} remaining")

    # warm up tokenizer before the loop
    get_tokenizer()

    total     = len(prompts)
    processed = len(done_ids)
    errors    = 0

    print(f"\nSending prompts to {base_url} (model: {model})")
    print("Ctrl-C to stop cleanly and resume later.\n")

    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, record in enumerate(prompts):
            conv_id = record.get("conversation_id", str(i))
            if conv_id in done_ids:
                continue

            prompt = record["prompt"]
            t0 = time.time()

            try:
                response = call_ollama(prompt, model, base_url, timeout)
                elapsed  = time.time() - t0

                if not response.strip():
                    errors += 1
                    print(f"  [{processed+1}/{total}] EMPTY response — skipping")
                    continue

                token_count = count_tokens(response)
                processed  += 1

                out_rec = {
                    "prompt":               prompt,
                    "response":             response,
                    "response_token_count": token_count,
                    "conversation_id":      conv_id,
                    "model":                model,
                    "elapsed_sec":          round(elapsed, 2),
                }
                out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                out_f.flush()

                # progress every 10 requests
                if processed % 10 == 0:
                    pct = processed / total * 100
                    eta = (total - processed) * (elapsed)  # rough estimate
                    print(f"  [{processed}/{total}] {pct:.1f}% "
                          f"| last: {elapsed:.1f}s "
                          f"| tokens: {token_count} "
                          f"| ETA: {eta/60:.0f}min")

            except KeyboardInterrupt:
                print(f"\nInterrupted. Progress saved — {processed:,} done. Re-run to resume.")
                sys.exit(0)

            except (urllib.error.URLError, TimeoutError) as e:
                errors += 1
                print(f"  [{i+1}/{total}] ERROR: {e} — skipping")
                continue

            except Exception as e:
                errors += 1
                print(f"  [{i+1}/{total}] UNEXPECTED ERROR: {e} — skipping")
                continue

    print(f"\nDone. Processed: {processed:,}  Errors: {errors:,}")
    print(f"Output → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LMSYS prompts through Ollama")
    parser.add_argument("--prompts", default="data/lmsys_prompts.jsonl")
    parser.add_argument("--out",     default="data/serving_logs.jsonl")
    parser.add_argument("--model",   default="gemma3:4b")
    parser.add_argument("--url",     default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-request timeout in seconds (default: 300)")
    args = parser.parse_args()
    main(args.prompts, args.out, args.model, args.url, args.timeout)
