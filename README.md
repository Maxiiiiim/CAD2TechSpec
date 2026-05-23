# CAD2TechSpec

![CAD2TechSpec overview](https://github.com/Maxiiiiim/CAD2TechSpec/blob/main/Framework_CAD2TechSpec.png)

## Description
CAD2TechSpec is a novel framework for automating design processes within CAD systems by leveraging multimodal large language models (LLMs). The framework enables the analysis and generation of detailed design specifications, including the automated creation of machining process plans. Our system architecture combines 3D model rendering, dimensionality reduction techniques, and the capabilities of multimodal LLMs to produce structured JSON representations of manufacturing workflows. Experiments conducted on the ABC dataset demonstrate that CAD2TechSpec significantly reduces design time while enhancing the accuracy and completeness of technical specifications. The proposed approach holds considerable promise for high-tech industries such as precision manufacturing and mechanical engineering, where efficiency and precision in design processes are critical.

## Dataset Information
**ABC Data.** The primary data utilized for the CAD2TechSpec framework were sourced from the [ABC Dataset](https://doi.ieeecomputersociety.org/10.1109/CVPR.2019.00983). The project pipeline employed two distinct data types: Obj and Stats. The Obj files contain models with ground truth normals and curvature values at each vertex, serving as the main input data for the framework. In contrast, the Stats files provide statistical information about the CAD model in parametric boundary representation, which was used for selecting single-component parts within the CAD representation. A detailed description of all data contained in the dataset is available at the following link: https://deep-geometry.github.io/abc-dataset/.

**Equipment & ISO knowledge base.** RAG tables list equipment, tooling, ISO numbers, and descriptions. The main file is `example_material/equipment_tooling_base.csv`. The external data for RAG consist of normative documents developed by the [International Organization for Standardization (ISO)](https://www.iso.org/home.html).

## Code Information

| Path | Role |
|------|------|
| `abc_dataset/` | Data from the [ABC dataset](https://archive.nyu.edu/handle/2451/44309) |
| `framework.py` | End-to-end entry: collages → JSON generation → evaluation |
| `dim_reduction_solved_3d_model.py` | Isomap selection + collage assembly (3 / 4 / 6 views) |
| `render_script.py` | Blender rendering (28 views per part) |
| `llm_benchmarks.py` | Generate JSON via Mistral Large 3, Qwen2.5-VL-72B, Qwen-VL-Max |
| `evaluation.py` | Score generated JSON (LLaVA-Critic, OpenRouter) |
| `utils/evaluation_llava.py` | Shared evaluation + local VLM inference |
| `utils/text_rag.py` | BM25 text RAG over equipment/ISO CSV |
| `utils/multimodal_rag.py` | CLIP-based few-shot retrieval for generation |
| `example_material/prompts/` | Generation and evaluation prompt templates |
| `results_no_rag/`, `results_rag/` | Generated JSON outputs (by model / collage count) |
| `metrics/` | Evaluation checkpoints |

### Prompts (`example_material/prompts/`)

- `machining_process_prompt.py` – JSON generation for machining plans
- `evaluation_prompt_plain.py` – judge prompt (JSON + collage, no KB block)
- `evaluation_prompt_rag_aug.py` — judge prompt with retrieved equipment/ISO context (used by `evaluation.py`)

### `example_material/` (runtime data)

- `rendered_imgs/` — multi-view renders per part
- `collages_3`, `collages_4`, `collages_6` — Isomap collages
- `equipment_tooling_base.csv` — text RAG knowledge base
- `example_object_path.pkl` — list of `.obj` paths for Blender

## Usage Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For **local LLaVA-Critic** on GPU, install a CUDA build of PyTorch if needed, e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Render multi-view images (Blender)

Install [Blender](https://www.blender.org/) (3.4.x recommended; Cap3D bundle: https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/blender.zip) and run from the project root:

```bash
blender -b -P render_script.py -- \
  --object_path_pkl './example_material/example_object_path.pkl' \
  --parent_dir './example_material'
```

`render_script.py` runs an 8-view pass (type 1) and a 20-view pass (type 2) per object (28 images), adapted from [Cap3D / DiffuRank](https://github.com/tiangeluo/DiffuRank).

### 3. Full pipeline

Set API keys in `framework.py` (or export as environment variables):

```python
DASHSCOPE_API_KEY = ""   # Qwen (DashScope)
MISTRAL_API_KEY = ""     # Mistral Large 3 (vision)
OPENROUTER_API_KEY = ""  # OpenRouter judge (optional step)
```

```bash
python framework.py
```

Steps executed:

1. **`dim_reduction_solved_3d_model.dim_reduction()`** — build collages in `example_material/collages_{3,4,6}/`
2. **`llm_benchmarks.llm_benchmark(...)`** — generate JSON under `results_no_rag/` and `results_rag/` for:
   - **Mistral Large 3** (`mistral_large_3/`)
   - **Qwen2.5-VL-72B** (`qwen2_5_vl_72b/`)
   - **Qwen-VL-Max** (`qwen_vl_max/`)
3. **`evaluation.evaluate(api_key=OPENROUTER_API_KEY)`** — run both judges sequentially (see below)

## Requirements
Core Python packages (see `requirements.txt`):

- **Data / ML:** numpy, pandas, scipy, scikit-learn, matplotlib, pillow, pyyaml, rank-bm25
- **Vision LLM (local judge):** torch (≥2.5), transformers (≥4.46), accelerate, safetensors
- **API clients:** openai, mistralai, openrouter

**Blender** (separate install): required for `render_script.py` (`bpy`, `mathutils`). 
