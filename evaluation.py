"""
Evaluate generated JSON with two judges run sequentially:
  1) local LLaVA-Critic
  2) OpenRouter (skipped if no API key)

  python evaluation.py
  evaluation.evaluate(api_key=...)   # OpenRouter step uses this key

Use --backend local|openrouter to run a single judge only.
Outputs: metrics/judge_details_*_{local,openrouter}.jsonl and metrics_*_{local,openrouter}.pkl
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

from utils.evaluation_llava import (
    generate,
    make_openrouter_judge_fn,
    run,
    warmup,
)


def _normalize_backend(name: str) -> str:
    raw = name.strip().lower()
    if raw in ("local", "llava", "llava-critic"):
        return "local"
    if raw in ("openrouter", "gpt", "router"):
        return "openrouter"
    raise ValueError(f"Unknown backend: {name!r} (use local or openrouter)")


def _common_run_kwargs() -> Dict[str, Any]:
    only = os.getenv("EVAL_ONLY", "both").strip()
    resume = os.getenv("EVAL_RESUME", "1").strip() not in {"0", "false", "False"}
    return dict(
        sleep_s=float(os.getenv("EVAL_SLEEP_S", "0.2")),
        max_retries=int(os.getenv("EVAL_MAX_RETRIES", "2")),
        only=only,
        metrics_dir=os.getenv("METRICS_DIR", "metrics"),
        batch_size=int(os.getenv("EVAL_BATCH_SIZE", "50")),
        resume=resume,
    )


def _run_local(**run_kwargs: Any) -> None:
    print("\n=== Judge 1/2: local LLaVA-Critic ===")
    repo = os.getenv("HF_REPO_ID", "lmms-lab/LLaVA-Critic-R1-7B")
    print(f"Loading model ({repo})…")
    warmup()
    print("Model ready.")

    def judge_fn(*, prompt: str, image_path: Optional[str]) -> str:
        return generate(prompt=prompt, image_path=image_path)

    run(
        judge_fn=judge_fn,
        concurrency=1,
        run_label="local",
        row_extra={"judge_backend": "local"},
        **run_kwargs,
    )


def _run_openrouter(api_key: str, **run_kwargs: Any) -> None:
    key = api_key.strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("\n=== Judge 2/2: OpenRouter — skipped (no API key) ===")
        return

    print("\n=== Judge 2/2: OpenRouter ===")
    model = os.getenv("OPENROUTER_JUDGE_MODEL", "openai/gpt-5-mini").strip()
    if not model:
        raise ValueError("OpenRouter: set OPENROUTER_JUDGE_MODEL.")

    max_tokens_raw = os.getenv("OPENROUTER_MAX_TOKENS", "4096").strip()
    max_tokens = int(max_tokens_raw) if max_tokens_raw else None
    concurrency = int(os.getenv("EVAL_CONCURRENCY", "6"))

    print(f"Model: {model}, concurrency={concurrency}")
    judge_fn = make_openrouter_judge_fn(api_key=key, model=model, max_tokens=max_tokens)
    run(
        judge_fn=judge_fn,
        concurrency=concurrency,
        run_label="openrouter",
        row_extra={"judge_backend": "openrouter", "judge_router_model": model},
        **run_kwargs,
    )


def run_evaluation(*, backend: Optional[str] = None, api_key: str = "") -> None:
    run_kwargs = _common_run_kwargs()

    if backend is not None and str(backend).strip():
        choice = _normalize_backend(str(backend))
        if choice == "local":
            _run_local(**run_kwargs)
        else:
            _run_openrouter(api_key, **run_kwargs)
        return

    _run_local(**run_kwargs)
    _run_openrouter(api_key, **run_kwargs)


def evaluate(api_key: str = "") -> None:
    """Entry point for framework.py: runs local, then OpenRouter (if key is set)."""
    run_evaluation(api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated JSON (local LLaVA-Critic, then OpenRouter)."
    )
    parser.add_argument(
        "--backend",
        choices=["local", "openrouter"],
        default=None,
        help="Run only one judge; default runs both in sequence",
    )
    args = parser.parse_args()
    try:
        run_evaluation(backend=args.backend)
    except ValueError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
