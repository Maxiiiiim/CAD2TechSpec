from __future__ import annotations

import json
import os
import re
import time
from typing import Any


def _safe_slug(s: str, *, max_len: int = 80) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "unknown"
    return s[:max_len]


def log_final_prompt(
    *,
    log_root_dir: str,
    provider: str,
    model_name: str,
    data_type: str,
    image_path: str,
    part_number: str | None,
    full_prompt: str,
    append_section_name: str | None = None,
    append_section_text: str | None = None,
) -> str:
    """
    Persist the final prompt text that is sent to the model (including RAG additions).
    Returns the created log file path.
    """
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    provider_slug = _safe_slug(provider)
    model_slug = _safe_slug(model_name)
    dtype_slug = _safe_slug(data_type)
    part_slug = _safe_slug(part_number or "unknown")
    img_slug = _safe_slug(os.path.splitext(os.path.basename(image_path))[0])

    out_dir = os.path.join(log_root_dir, provider_slug, dtype_slug, model_slug)
    os.makedirs(out_dir, exist_ok=True)

    filename = f"{ts}Z__{part_slug}__{img_slug}.txt"
    out_path = os.path.join(out_dir, filename)

    payload: dict[str, Any] = {
        "timestamp_utc": f"{ts}Z",
        "provider": provider,
        "model": model_name,
        "data_type": data_type,
        "part_number": part_number,
        "image_path": image_path,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))
        f.write("\n\n----- FINAL_PROMPT -----\n")
        f.write(full_prompt)
        f.write("\n")
        if append_section_name and append_section_text is not None:
            f.write(f"\n----- {append_section_name} -----\n")
            f.write(append_section_text)
            if not append_section_text.endswith("\n"):
                f.write("\n")

    return out_path


def format_fewshot_log_section(
    *,
    query_image_path: str,
    fewshot_root: str,
    top_k: int,
    min_similarity: float | None,
    results: list[dict],
    infer_collages_n,
) -> str:
    collages_n = infer_collages_n(query_image_path)
    index_path = (
        os.path.join(fewshot_root, f"index_{collages_n}.pkl") if collages_n is not None else None
    )
    header = {
        "fewshot_root": fewshot_root,
        "index_path": index_path,
        "query_image_path": query_image_path,
        "collages_n": collages_n,
        "top_k": top_k,
        "min_similarity": min_similarity,
        "retrieved_count": len(results),
    }
    out_lines: list[str] = []
    out_lines.append(json.dumps(header, ensure_ascii=False, indent=2))
    out_lines.append("")
    for i, r in enumerate(results, start=1):
        out_lines.append(f"## EXAMPLE {i}")
        out_lines.append(f"id: {r.get('id')}")
        out_lines.append(f"similarity: {r.get('similarity')}")
        out_lines.append(f"image_path: {r.get('image_path')}")
        out_lines.append(f"json_path: {r.get('json_path')}")
        out_lines.append("")
        try:
            with open(r.get("json_path", ""), "r", encoding="utf-8") as f:
                out_lines.append(f.read().strip())
        except Exception as e:
            out_lines.append(f"[failed to read json] {e}")
        out_lines.append("")
    return "\n".join(out_lines).rstrip() + "\n"

