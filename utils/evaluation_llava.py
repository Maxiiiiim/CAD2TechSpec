"""
LLaVA-Critic evaluation: local inference, OpenRouter judge, prompts, RAG, metrics.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import torch
from PIL import Image

from example_material.prompts.evaluation_prompt_rag_aug import get_evaluation_prompt
from utils.text_rag import retrieve_relevant_data

try:
    try:
        from transformers import AutoModelForImageTextToText as AutoVLModel, AutoProcessor
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLModel, AutoProcessor
except ValueError as e:
    msg = str(e)
    if "infer_schema" in msg or "unsupported type" in msg:
        raise RuntimeError(
            "Incompatible PyTorch and transformers versions on import.\n"
            "Upgrade PyTorch to >= 2.5, then transformers/accelerate, e.g.:\n"
            "  pip install -U 'torch>=2.5' transformers accelerate\n"
            "  pip install -r requirements.txt\n"
            f"Original error: {e}"
        ) from e
    raise

JudgeFn = Callable[..., str]

SCORES_RE = re.compile(r"(-?\d+)")
SCORES_TAG_RE = re.compile(r"<scores>\s*(.*?)\s*</scores>", re.DOTALL | re.IGNORECASE)
REASONING_END_RE = re.compile(r"</reasoning>", re.IGNORECASE)
ASSISTANT_PREFIX_RE = re.compile(r"\bassistant\b", re.IGNORECASE)
DEFAULT_RAG_CSV = os.path.join("example_material", "equipment_tooling_base.csv")
DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_BATCH_SIZE = 50


@dataclass(frozen=True)
class EvalItem:
    data_type: str
    model: str
    collages: str
    part_id: str
    generated_path: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_retrieved_context(records: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for r in records:
        t = str(r.get("type", "")).strip()
        name = str(r.get("name", "")).strip()
        iso = str(r.get("iso", "")).strip()
        iso_title = str(r.get("iso title", "")).strip()
        desc = str(r.get("description", "")).strip()
        lines.append(
            " | ".join(
                x
                for x in [
                    f"type={t}" if t else "",
                    f"name={name}" if name else "",
                    f"iso={iso}" if iso else "",
                    f"iso_title={iso_title}" if iso_title else "",
                    f"description={desc}" if desc else "",
                ]
                if x
            )
        )
    return "\n".join(f"- {l}" for l in lines if l)


def create_prompt(generated_json: Any) -> str:
    generated_str = json.dumps(generated_json, ensure_ascii=False, indent=2)

    rag_records = retrieve_relevant_data(generated_str, DEFAULT_RAG_CSV, top_k=10)
    retrieved_context = _format_retrieved_context(rag_records)
    if not retrieved_context:
        retrieved_context = "(empty)"

    return get_evaluation_prompt(generated_json=generated_str, retrieved_context=retrieved_context)


def find_collage_image_path(*, collages: str, part_id: str) -> Optional[str]:
    base = os.path.join("example_material", f"collages_{collages}")
    for ext in DEFAULT_IMAGE_EXTS:
        p = os.path.join(base, f"{part_id}{ext}")
        if os.path.isfile(p):
            return p
    return None


def _scores_from_segment(segment: str, *, expected: int) -> Optional[List[int]]:
    segment = (segment or "").strip()
    if not segment:
        return None
    nums = [int(x) for x in SCORES_RE.findall(segment)]
    in_range = [n for n in nums if 1 <= n <= 5]
    if len(in_range) >= expected:
        return in_range[:expected]
    return None


def _scores_from_lines_tail(blob: str, *, expected: int) -> Optional[List[int]]:
    blob = blob.strip()
    if len(blob) > 1500:
        blob = blob[-1500:]
    for line in reversed(blob.splitlines()):
        line = line.strip()
        if not line:
            continue
        nums = [int(x) for x in SCORES_RE.findall(line)]
        vin = [n for n in nums if 1 <= n <= 5]
        if len(vin) >= expected:
            return vin[:expected]
    return None


def _reply_after_last_assistant(text: str) -> str:
    matches = list(ASSISTANT_PREFIX_RE.finditer(text))
    if not matches:
        return text
    return text[matches[-1].end() :].strip()


def parse_scores(text: str, *, expected: int) -> Optional[List[int]]:
    text = (text or "").strip()
    if not text:
        return None

    candidates = [_reply_after_last_assistant(text), text]

    for blob in candidates:
        blob = blob.strip()
        if not blob:
            continue
        matches = list(SCORES_TAG_RE.finditer(blob))
        for m in reversed(matches):
            got = _scores_from_segment(m.group(1), expected=expected)
            if got:
                return got

        split_m = REASONING_END_RE.search(blob)
        tail = blob[split_m.end() :] if split_m else blob
        for m in reversed(list(SCORES_TAG_RE.finditer(tail))):
            got = _scores_from_segment(m.group(1), expected=expected)
            if got:
                return got
        got = _scores_from_lines_tail(tail, expected=expected)
        if got:
            return got

    return None


def iter_generated_items(*, data_type: str) -> Iterable[EvalItem]:
    base = os.path.join(data_type, "json_responses")
    if not os.path.isdir(base):
        return

    for dirpath, _, filenames in os.walk(base):
        for name in filenames:
            if not name.endswith(".json"):
                continue
            generated_path = os.path.join(dirpath, name)
            part_id = os.path.splitext(name)[0]

            rel = os.path.relpath(generated_path, base)
            parts = rel.split(os.sep)
            model = parts[0] if len(parts) >= 1 else "unknown_model"

            collages = "unknown"
            for p in parts:
                if p.startswith("collages_"):
                    collages = p.split("_", 1)[1]
                    break

            yield EvalItem(
                data_type=data_type,
                model=model,
                collages=collages,
                part_id=part_id,
                generated_path=generated_path,
            )


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_metrics_dir(out_dir: str = "metrics") -> str:
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _load_done_generated_paths(jsonl_path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.isfile(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("status") == "ok" and isinstance(row.get("generated_path"), str):
                done.add(row["generated_path"])
    return done


def _load_aggregated_if_exists(pkl_path: str) -> Dict[str, Dict[str, List[List[int]]]]:
    if not os.path.isfile(pkl_path):
        return {}
    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def _metrics_suffix(run_label: str) -> str:
    label = (run_label or "").strip()
    return f"_{label}" if label else ""


def _save_metrics(
    *,
    aggregated: Dict[str, Dict[str, List[List[int]]]],
    out_dir: str,
    data_type: str,
    run_label: str = "",
) -> Tuple[str, str]:
    suffix = _metrics_suffix(run_label)
    stem = f"metrics_{data_type.replace('results_', '')}{suffix}"
    pkl_path = os.path.join(out_dir, f"{stem}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(aggregated, f)

    summary_path = os.path.join(out_dir, f"{stem}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)

    return pkl_path, summary_path


# --- Local LLaVA-Critic inference ---

HF_REPO_ID = os.environ.get("HF_REPO_ID", "lmms-lab/LLaVA-Critic-R1-7B").strip()
TORCH_DTYPE = os.environ.get("TORCH_DTYPE", "bfloat16").strip()

_processor: Any = None
_model: Any = None


def _torch_dtype() -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(TORCH_DTYPE, torch.float16)


def _ensure_model_loaded() -> tuple[Any, Any]:
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    if not HF_REPO_ID:
        raise RuntimeError("Set HF_REPO_ID to the HuggingFace model repo id.")

    _processor = AutoProcessor.from_pretrained(HF_REPO_ID, trust_remote_code=True)
    _model = AutoVLModel.from_pretrained(
        HF_REPO_ID,
        trust_remote_code=True,
        torch_dtype=_torch_dtype(),
        device_map="auto",
    )
    _model.eval()
    return _processor, _model


def _decode_only_new_tokens(processor: Any, *, generated_ids: torch.Tensor, prompt_token_len: int) -> str:
    new_ids = generated_ids[prompt_token_len:]
    if new_ids.numel() == 0:
        return ""
    tok = getattr(processor, "tokenizer", None)
    ids_cpu = new_ids.detach().cpu()
    if tok is not None and hasattr(tok, "decode"):
        return tok.decode(ids_cpu, skip_special_tokens=True)
    return processor.decode(ids_cpu, skip_special_tokens=True)


def _build_model_inputs(*, processor: Any, prompt: str, image: Optional[Image.Image]) -> dict:
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": ([{"type": "image"}] if image is not None else [])
                + [{"type": "text", "text": prompt}],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return processor(text=text, images=image, return_tensors="pt")
    return processor(text=prompt, images=image, return_tensors="pt")


def warmup() -> None:
    """Load model weights into memory (idempotent)."""
    _ensure_model_loaded()


def generate(*, prompt: str, image_path: Optional[str] = None, max_new_tokens: int = 512) -> str:
    """
    Run LLaVA-Critic locally. Loads the model on first call.

    Env: HF_REPO_ID, TORCH_DTYPE, JUDGE_MAX_NEW_TOKENS
    """
    max_new_tokens = int(os.getenv("JUDGE_MAX_NEW_TOKENS", str(max_new_tokens)))
    processor, model = _ensure_model_loaded()

    image: Optional[Image.Image] = None
    if image_path:
        image = Image.open(image_path).convert("RGB")

    inputs = _build_model_inputs(processor=processor, prompt=prompt, image=image)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=int(max_new_tokens))

    return _decode_only_new_tokens(
        processor,
        generated_ids=out[0],
        prompt_token_len=prompt_len,
    )


# --- OpenRouter judge ---


def _b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _mime_for_image_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


def _openrouter_user_message_content(*, prompt: str, image_path: Optional[str]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_path:
        b64 = _b64_image(image_path)
        mime = _mime_for_image_path(image_path)
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return parts


def _text_from_chat_message(msg: Any) -> str:
    content = getattr(msg, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
            else:
                t = getattr(block, "text", None)
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts).strip()
    return str(content).strip()


def call_openrouter_judge(
    *,
    api_key: str,
    prompt: str,
    image_path: Optional[str],
    model: str,
    max_tokens: Optional[int] = None,
) -> str:
    from openrouter import OpenRouter

    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": _openrouter_user_message_content(prompt=prompt, image_path=image_path),
        }
    ]
    send_kwargs: Dict[str, Any] = {"model": model, "messages": messages}
    if max_tokens is not None:
        send_kwargs["max_tokens"] = int(max_tokens)

    with OpenRouter(api_key=api_key) as client:
        response = client.chat.send(**send_kwargs)

    choice = response.choices[0]
    msg = choice.message
    out = _text_from_chat_message(msg)
    if not out:
        bits: List[str] = []
        fr = getattr(choice, "finish_reason", None)
        if fr is not None:
            bits.append(f"finish_reason={fr!r}")
        refusal = getattr(msg, "refusal", None)
        if refusal:
            bits.append(f"refusal={refusal!r}")
        try:
            if hasattr(msg, "model_dump"):
                bits.append("message=" + json.dumps(msg.model_dump(), ensure_ascii=False)[:4000])
            else:
                bits.append(f"message={repr(msg)[:2000]}")
        except Exception as e:
            bits.append(f"message_dump_error={e!r}")
        out = "[EMPTY_ASSISTANT] " + " | ".join(bits)
    return out


def make_openrouter_judge_fn(
    *,
    api_key: str,
    model: str,
    max_tokens: Optional[int] = None,
) -> JudgeFn:
    def judge_fn(*, prompt: str, image_path: Optional[str]) -> str:
        return call_openrouter_judge(
            api_key=api_key,
            prompt=prompt,
            image_path=image_path,
            model=model,
            max_tokens=max_tokens,
        )

    return judge_fn


# --- Evaluation loop ---


def _eval_one_item(
    item: EvalItem,
    *,
    judge_fn: JudgeFn,
    sleep_s: float,
    max_retries: int,
    row_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    image_path = find_collage_image_path(collages=item.collages, part_id=item.part_id)
    extra = dict(row_extra or {})
    try:
        generated_json = load_json(item.generated_path)
    except Exception as e:
        return {
            "ts_utc": _utc_now_iso(),
            "data_type": item.data_type,
            "model": item.model,
            "collages": item.collages,
            "part_id": item.part_id,
            "generated_path": item.generated_path,
            "image_path": image_path,
            "status": "json_load_error",
            "error": str(e),
            **extra,
        }

    prompt = create_prompt(generated_json)
    last_text = ""
    scores: Optional[List[int]] = None
    for attempt in range(max_retries + 1):
        try:
            last_text = judge_fn(prompt=prompt, image_path=image_path)
            scores = parse_scores(last_text, expected=3)
            if scores is not None:
                break
        except Exception as e:
            last_text = f"ERROR: {e}"
        if attempt < max_retries:
            time.sleep(max(0.0, sleep_s))

    if scores is None:
        return {
            "ts_utc": _utc_now_iso(),
            "data_type": item.data_type,
            "model": item.model,
            "collages": item.collages,
            "part_id": item.part_id,
            "generated_path": item.generated_path,
            "image_path": image_path,
            "status": "judge_parse_error",
            "judge_output": last_text,
            **extra,
        }
    return {
        "ts_utc": _utc_now_iso(),
        "data_type": item.data_type,
        "model": item.model,
        "collages": item.collages,
        "part_id": item.part_id,
        "generated_path": item.generated_path,
        "image_path": image_path,
        "status": "ok",
        "scores": scores,
        "judge_output": last_text.strip(),
        **extra,
    }


def run(
    *,
    judge_fn: JudgeFn,
    sleep_s: float = 0.2,
    max_retries: int = 2,
    only: str = "both",
    metrics_dir: str = "metrics",
    batch_size: Optional[int] = None,
    resume: Optional[bool] = None,
    concurrency: int = 1,
    row_extra: Optional[Dict[str, Any]] = None,
    run_label: str = "",
) -> None:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    out_dir = ensure_metrics_dir(metrics_dir)
    if batch_size is None:
        batch_size = int(os.getenv("EVAL_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
    if resume is None:
        resume = os.getenv("EVAL_RESUME", "1").strip() not in {"0", "false", "False"}

    suffix = _metrics_suffix(run_label)
    targets = ["results_no_rag", "results_rag"] if only == "both" else [only]
    for data_type in targets:
        jsonl_path = os.path.join(out_dir, f"judge_details_{data_type}{suffix}.jsonl")
        pkl_path = os.path.join(out_dir, f"metrics_{data_type.replace('results_', '')}{suffix}.pkl")

        done = _load_done_generated_paths(jsonl_path) if resume else set()
        aggregated = _load_aggregated_if_exists(pkl_path) if resume else {}

        items = list(iter_generated_items(data_type=data_type))
        processed_since_save = 0
        pending = [it for it in items if it.generated_path not in done]

        with open(jsonl_path, "a", encoding="utf-8") as details_f:
            write_lock = threading.Lock()

            def commit_row(item: EvalItem, row: Dict[str, Any]) -> None:
                nonlocal processed_since_save
                with write_lock:
                    details_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    details_f.flush()
                    if row.get("status") == "ok" and isinstance(row.get("scores"), list):
                        aggregated.setdefault(item.model, {}).setdefault(item.collages, []).append(
                            row["scores"]
                        )
                    done.add(item.generated_path)
                    processed_since_save += 1
                    if processed_since_save >= batch_size:
                        saved_pkl, saved_json = _save_metrics(
                            aggregated=aggregated,
                            out_dir=out_dir,
                            data_type=data_type,
                            run_label=run_label,
                        )
                        print(f"[{data_type}] checkpoint wrote: {saved_pkl}, {saved_json} (+ {jsonl_path})")
                        processed_since_save = 0

            if concurrency <= 1:
                for item in pending:
                    row = _eval_one_item(
                        item,
                        judge_fn=judge_fn,
                        sleep_s=sleep_s,
                        max_retries=max_retries,
                        row_extra=row_extra,
                    )
                    commit_row(item, row)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as ex:
                    fut_to_item = {
                        ex.submit(
                            _eval_one_item,
                            item,
                            judge_fn=judge_fn,
                            sleep_s=sleep_s,
                            max_retries=max_retries,
                            row_extra=row_extra,
                        ): item
                        for item in pending
                    }
                    for fut in as_completed(fut_to_item):
                        item = fut_to_item[fut]
                        row = fut.result()
                        commit_row(item, row)

        saved_pkl, saved_json = _save_metrics(
            aggregated=aggregated, out_dir=out_dir, data_type=data_type, run_label=run_label
        )
        print(f"[{data_type}] wrote: {saved_pkl}, {saved_json}, {jsonl_path}")
