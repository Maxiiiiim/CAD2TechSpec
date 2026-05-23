from __future__ import annotations

import os
import re
import time
import pickle
from typing import Any

import numpy as np
from PIL import Image


CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

_FEWSHOT_INDEX_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_CLIP_DEVICE = None


def infer_collages_n_from_path(image_path: str) -> int | None:
    m = re.search(r"collages_(\d+)", image_path)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _load_clip_once() -> None:
    global _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE
    if _CLIP_MODEL is not None and _CLIP_PROCESSOR is not None:
        return

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except Exception as e:
        raise RuntimeError(
            "Local few-shot RAG requires torch+transformers. "
            "Install with: pip install -r requirements.txt"
        ) from e

    _CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    _CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(_CLIP_DEVICE)
    _CLIP_MODEL.eval()


def clip_image_embedding(image_path: str) -> np.ndarray:
    """Return L2-normalized CLIP image embedding (float32, shape [d])."""
    _load_clip_once()
    import torch

    img = Image.open(image_path).convert("RGB")
    inputs = _CLIP_PROCESSOR(images=img, return_tensors="pt")
    inputs = {k: v.to(_CLIP_DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        try:
            feats = _CLIP_MODEL.get_image_features(**inputs)
        except Exception:
            feats = _CLIP_MODEL(**inputs)

    if isinstance(feats, torch.Tensor):
        vec = feats
    else:
        if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
            vec = feats.image_embeds
        elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            vec = feats.pooler_output
        else:
            raise RuntimeError(f"Unexpected CLIP output type: {type(feats)}")

    vec = vec.detach().float().cpu().numpy().reshape(-1)
    norm = float(np.linalg.norm(vec) + 1e-12)
    return (vec / norm).astype(np.float32, copy=False)


def build_fewshot_index(
    *,
    collages_n: int,
    fewshot_root: str,
    output_path: str | None = None,
) -> str:
    if output_path is None:
        output_path = os.path.join(fewshot_root, f"index_{collages_n}.pkl")

    base_dir = os.path.join(fewshot_root, f"collages_{collages_n}")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Few-shot folder not found: {base_dir}")

    entries: list[dict[str, Any]] = []
    for dirpath, _, filenames in os.walk(base_dir):
        for fn in filenames:
            if fn.startswith("."):
                continue
            if not fn.lower().endswith(".jpg"):
                continue
            img_path = os.path.join(dirpath, fn)
            stem = os.path.splitext(fn)[0]
            json_path = os.path.join(dirpath, f"{stem}.json")
            if not os.path.exists(json_path):
                continue
            emb = clip_image_embedding(img_path)
            rel_id = os.path.relpath(os.path.join(dirpath, stem), fewshot_root)
            entries.append(
                {
                    "id": rel_id,
                    "image_path": img_path,
                    "json_path": json_path,
                    "embedding": emb,
                }
            )

    if not entries:
        raise RuntimeError(f"No (jpg,json) pairs found under {base_dir}")

    mat = np.stack([e["embedding"] for e in entries], axis=0).astype(np.float32, copy=False)
    payload: dict[str, Any] = {
        "collages_n": collages_n,
        "model_id": CLIP_MODEL_ID,
        "created_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "entries": entries,
        "matrix": mat,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    return output_path


def _load_fewshot_index(index_path: str) -> dict[str, Any]:
    abs_path = os.path.abspath(index_path)
    mtime = os.path.getmtime(abs_path)
    cached = _FEWSHOT_INDEX_CACHE.get(abs_path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    with open(abs_path, "rb") as f:
        payload = pickle.load(f)
    _FEWSHOT_INDEX_CACHE[abs_path] = (mtime, payload)
    return payload


def retrieve_fewshot_examples(
    *,
    query_image_path: str,
    top_k: int,
    fewshot_root: str,
    min_similarity: float | None = 0.25,
) -> list[dict[str, Any]]:
    collages_n = infer_collages_n_from_path(query_image_path)
    if collages_n is None:
        return []

    index_path = os.path.join(fewshot_root, f"index_{collages_n}.pkl")
    if not os.path.exists(index_path):
        return []

    idx = _load_fewshot_index(index_path)
    mat = idx["matrix"]  # normalized rows
    q = clip_image_embedding(query_image_path)  # normalized
    sims = (mat @ q.reshape(-1, 1)).reshape(-1)

    k = min(int(top_k), sims.shape[0])
    if k <= 0:
        return []
    top_idx = np.argsort(sims)[-k:][::-1]

    results: list[dict[str, Any]] = []
    for i in top_idx:
        sim = float(sims[i])
        if min_similarity is not None and sim < float(min_similarity):
            continue
        e = idx["entries"][int(i)]
        results.append(
            {
                "id": e["id"],
                "image_path": e["image_path"],
                "json_path": e["json_path"],
                "similarity": sim,
            }
        )
    return results

