import base64
import json
import numpy as np
import os
import pickle
import time

from mistralai.client import Mistral
from openai import OpenAI

from example_material.prompts.machining_process_prompt import get_prompt

from utils.abc_meta import get_part_meta
from utils.multimodal_rag import build_fewshot_index, infer_collages_n_from_path, retrieve_fewshot_examples
from utils.prompt_logging import format_fewshot_log_section, log_final_prompt
from utils.text_rag import retrieve_relevant_data

# Defaults / paths
ABC_STATS_ROOT = "./abc_dataset/abc_0000_stat_v00"
ABC_FEATURES_ROOT = "./abc_dataset/abc_0000_feat_v00"
EQUIPMENT_TOOLING_CSV = "./equipment_tooling_base.csv"
FEWSHOT_ROOT = "./few-shot"
PROMPT_LOG_ROOT = "./prompt_logs"


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_response_from_image_mistral(
    image_path,
    prompt,
    api_key,
    data_type,
    *,
    part_number: str | None = None,
    log_final_prompts: bool = True,
    final_prompt_log_dir: str = "./prompt_logs",
    fewshot_top_k: int = 1,
):
    # Vision-capable Mistral model.
    model = "mistral-large-2512"
    base64_image = encode_image_to_base64(image_path)

    fewshot_examples: list[dict] = []
    fewshot_min_similarity: float | None = 0.25
    if data_type == 'RAG':
        # Few-shot multimodal RAG (image->json)
        fewshot_examples = retrieve_fewshot_examples(
            query_image_path=image_path,
            top_k=fewshot_top_k,
            fewshot_root=FEWSHOT_ROOT,
            min_similarity=fewshot_min_similarity,
        )

        # Obtain additional context from RAG
        csv_path="./equipment_tooling_base.csv"
        rag_context = retrieve_relevant_data(prompt, csv_path)

        # Formulate context for the model
        rag_prompt = (
            "Important: if you are provided with a list of equipment/tooling/standards, it is meant as guidance only. You are NOT limited to that list. Choose plausible equipment and tooling even if it does not appear in the list. Use 'N/A' for equipment ONLY when the stage truly requires no specific equipment or it is impossible to infer from the input."
            "Use the following information about available equipment, tooling, and standards as guidance (not an exhaustive list). "
            "You may also use other plausible equipment/tooling if needed; avoid using \"N/A\" unless truly not applicable:\n"
        )
        for item in rag_context:
            item_type = item.get("type", "")
            name = item.get("name", "")
            process_steps = item.get("process_steps", "")
            iso = item.get("iso", "")
            iso_title = item.get("iso title", "")
            rag_prompt += (
                f"- Type: {item_type}, "
                f"Name: {name}, "
                f"Process steps: {process_steps}, "
                f"ISO: {iso}, ISO title: {iso_title}\n"
            )

        full_prompt = prompt + "\n\n" + rag_prompt
    else:
        full_prompt = prompt
        
    if log_final_prompts:
        fewshot_section = format_fewshot_log_section(
            query_image_path=image_path,
            fewshot_root=FEWSHOT_ROOT,
            top_k=fewshot_top_k,
            min_similarity=fewshot_min_similarity,
            results=fewshot_examples,
            infer_collages_n=infer_collages_n_from_path,
        )
        log_final_prompt(
            log_root_dir=final_prompt_log_dir,
            provider="mistral",
            model_name=model,
            data_type=data_type,
            image_path=image_path,
            part_number=part_number,
            full_prompt=full_prompt,
            append_section_name="FEWSHOT_RAG",
            append_section_text=fewshot_section,
        )

    # Build messages with optional retrieved few-shot demonstrations
    messages = []
    if fewshot_examples:
        for ex in fewshot_examples:
            ex_b64 = encode_image_to_base64(ex["image_path"])
            with open(ex["json_path"], "r", encoding="utf-8") as f:
                ex_json = f.read().strip()
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Example. Return JSON only in the same schema as shown."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{ex_b64}"
                        },
                    ],
                }
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": ex_json
                }
            )

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": full_prompt
                 },
                {
                    "type": "image_url", 
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                },
            ],
        }
    )

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model= model,
        messages=messages
    )
    return response.choices[0].message.content


def generate_response_from_image_qwen(
    image_path,
    model_name,
    prompt,
    my_api_key,
    data_type,
    *,
    part_number: str | None = None,
    log_final_prompts: bool = True,
    final_prompt_log_dir: str = "./prompt_logs",
    fewshot_top_k: int = 2,
):
    client = OpenAI(
        api_key= my_api_key, # DASHSCOPE_API_KEY
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    base64_image = encode_image_to_base64(image_path)

    fewshot_examples: list[dict] = []
    fewshot_min_similarity: float | None = 0.25
    if data_type == 'RAG':
        # Few-shot multimodal RAG (image->json)
        fewshot_examples = retrieve_fewshot_examples(
            query_image_path=image_path,
            top_k=fewshot_top_k,
            fewshot_root=FEWSHOT_ROOT,
            min_similarity=fewshot_min_similarity,
        )

        # Obtain additional context from RAG
        csv_path="./equipment_tooling_base.csv"
        rag_context = retrieve_relevant_data(prompt, csv_path)

        # Formulate context for the model
        rag_prompt = (
            "Important: if you are provided with a list of equipment/tooling/standards, it is meant as guidance only. You are NOT limited to that list. Choose plausible equipment and tooling even if it does not appear in the list. Use 'N/A' for equipment ONLY when the stage truly requires no specific equipment or it is impossible to infer from the input. "
            "Use the following information about available equipment, tooling, and standards as guidance (not an exhaustive list). "
            "You may also use other plausible equipment/tooling if needed; avoid using \"N/A\" unless truly not applicable:\n"
        )
        for item in rag_context:
            item_type = item.get("type", "")
            name = item.get("name", "")
            process_steps = item.get("process_steps", "")
            iso = item.get("iso", "")
            iso_title = item.get("iso title", "")
            rag_prompt += (
                f"- Type: {item_type}, "
                f"Name: {name}, "
                f"Process steps: {process_steps}, "
                f"ISO: {iso}, ISO title: {iso_title}\n"
            )

        full_prompt = prompt + "\n\n" + rag_prompt
    else:
        full_prompt = prompt

    if log_final_prompts:
        fewshot_section = format_fewshot_log_section(
            query_image_path=image_path,
            fewshot_root=FEWSHOT_ROOT,
            top_k=fewshot_top_k,
            min_similarity=fewshot_min_similarity,
            results=fewshot_examples,
            infer_collages_n=infer_collages_n_from_path,
        )
        log_final_prompt(
            log_root_dir=final_prompt_log_dir,
            provider="qwen",
            model_name=model_name,
            data_type=data_type,
            image_path=image_path,
            part_number=part_number,
            full_prompt=full_prompt,
            append_section_name="FEWSHOT_RAG",
            append_section_text=fewshot_section,
        )

    messages = []
    if fewshot_examples:
        for ex in fewshot_examples:
            ex_b64 = encode_image_to_base64(ex["image_path"])
            with open(ex["json_path"], "r", encoding="utf-8") as f:
                ex_json = f.read().strip()
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Example. Return JSON only in the same schema as shown."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex_b64}"}},
                    ],
                }
            )
            messages.append({"role": "assistant", "content": ex_json})

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
        )
    return completion.choices[0].message.content


def run_mistral(
    api_key,
    image_path,
    prompt,
    data_type,
    *,
    part_number: str | None = None,
    log_final_prompts: bool = True,
    final_prompt_log_dir: str = "./prompt_logs",
    fewshot_top_k: int = 1,
):
    try:
        print(f"image_path: {image_path}")
        response = generate_response_from_image_mistral(
            image_path,
            prompt,
            api_key,
            data_type,
            part_number=part_number,
            log_final_prompts=log_final_prompts,
            final_prompt_log_dir=final_prompt_log_dir,
            fewshot_top_k=fewshot_top_k,
        )
        return response
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def run_qwen(
    api_key,
    model_name,
    image_path,
    prompt,
    data_type,
    *,
    part_number: str | None = None,
    log_final_prompts: bool = True,
    final_prompt_log_dir: str = "./prompt_logs",
    fewshot_top_k: int = 2,
):
    try:
        print(f"image_path: {image_path}")
        response = generate_response_from_image_qwen(
            image_path,
            model_name,
            prompt,
            api_key,
            data_type,
            part_number=part_number,
            log_final_prompts=log_final_prompts,
            final_prompt_log_dir=final_prompt_log_dir,
            fewshot_top_k=fewshot_top_k,
        )
        return response
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def create_json_with_mistral(
    object_dirpath,
    n_dims: int,
    output_path,
    api_key_mistral,
    data_type,
    *,
    stats_root: str = "./abc_dataset/abc_0000_stat_v00",
    features_root: str = "./abc_dataset/abc_0000_feat_v00",
    scale_to_mm: float = 1.0,
):
    json_data = {}
    api_key =api_key_mistral
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}

    for dirpath, dirnames, filenames in os.walk(object_dirpath):
        for file_path in filenames:
          if file_path.startswith("."):
              continue
          _, ext = os.path.splitext(file_path)
          if ext.lower() not in allowed_exts:
              continue
          path = os.path.join(dirpath, file_path)
          part_number = file_path.split(".")[0]

          meta = get_part_meta(
              part_number,
              stats_root=stats_root,
              features_root=features_root,
              scale_to_mm=scale_to_mm,
          )
          prompt = get_prompt(
              n_dims,
              length=meta.get("length_mm", "unknown"),
              width=meta.get("width_mm", "unknown"),
              height=meta.get("height_mm", "unknown"),
              holes=meta.get("holes", []),
          )
          response = run_mistral(api_key, path, prompt, data_type, part_number=part_number)
          json_data[part_number] = response

    with open(output_path, "wb") as f:
        pickle.dump(json_data, f)


def create_json_with_qwen(
    api_key,
    model,
    object_dirpath,
    n_dims: int,
    output_path,
    data_type,
    *,
    stats_root: str = "./abc_dataset/abc_0000_stat_v00",
    features_root: str = "./abc_dataset/abc_0000_feat_v00",
    scale_to_mm: float = 1.0,
):
    json_data = {}
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}

    for dirpath, dirnames, filenames in os.walk(object_dirpath):
        for file_path in filenames:
          if file_path.startswith("."):
              continue
          _, ext = os.path.splitext(file_path)
          if ext.lower() not in allowed_exts:
              continue
          path = os.path.join(dirpath, file_path)
          part_number = file_path.split(".")[0]

          meta = get_part_meta(
              part_number,
              stats_root=stats_root,
              features_root=features_root,
              scale_to_mm=scale_to_mm,
          )
          prompt = get_prompt(
              n_dims,
              length=meta.get("length_mm", "unknown"),
              width=meta.get("width_mm", "unknown"),
              height=meta.get("height_mm", "unknown"),
              holes=meta.get("holes", []),
          )
          response = run_qwen(api_key, model, path, prompt, data_type, part_number=part_number)
          json_data[part_number] = response

    with open(output_path, "wb") as f:
        pickle.dump(json_data, f)


def save_jsons(json_pkl_paths, json_collages_paths):
    for i in range(len(json_pkl_paths)):
        json_data = pickle.load(open(json_pkl_paths[i], 'rb'))
        for key, value in json_data.items():
            if not isinstance(value, str) or not value.strip():
                # Skip failed generations (e.g., network errors returning None)
                continue
            cleaned_json = value.strip().replace('json', '').replace('```', '').strip()
            try:
                parsed = json.loads(cleaned_json)
            except Exception:
                # Skip non-JSON outputs
                continue
            os.makedirs(json_collages_paths[i], exist_ok=True)
            output_path = os.path.join(json_collages_paths[i], f"{key}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed, f, ensure_ascii=False, indent=4)


def llm_benchmark(api_key_qwen, api_key_mistral):
    # Ensure log dir exists even before first request
    os.makedirs(PROMPT_LOG_ROOT, exist_ok=True)

    object_path3, object_path4, object_path6 = './example_material/collages_3', './example_material/collages_4', './example_material/collages_6'

    for type in ['results_no_rag', 'results_rag']:
        data_type = 'RAG' if type == 'results_rag' else 'NO_RAG'
        
        mistral_pkl_path3 = f"./{type}/json_responses/json_mistral_3.pkl"
        mistral_pkl_path4 = f"./{type}/json_responses/json_mistral_4.pkl"
        mistral_pkl_path6 = f"./{type}/json_responses/json_mistral_6.pkl"

        vl_max_pkl_path3 = f"./{type}/json_responses/json_qwen_vl_max_3.pkl"
        vl_max_pkl_path4 = f"./{type}/json_responses/json_qwen_vl_max_4.pkl"
        vl_max_pkl_path6 = f"./{type}/json_responses/json_qwen_vl_max_6.pkl"

        qwen_72b_pkl_path3 = f"./{type}/json_responses/json_qwen2_vl_72b_instruct_3.pkl"
        qwen_72b_pkl_path4 = f"./{type}/json_responses/json_qwen2_vl_72b_instruct_4.pkl"
        qwen_72b_pkl_path6 = f"./{type}/json_responses/json_qwen2_vl_72b_instruct_6.pkl"

        create_json_with_mistral(object_path3, 3, mistral_pkl_path3, api_key_mistral, data_type)
        create_json_with_mistral(object_path4, 4, mistral_pkl_path4, api_key_mistral, data_type)
        create_json_with_mistral(object_path6, 6, mistral_pkl_path6, api_key_mistral, data_type)

        create_json_with_qwen(api_key_qwen, "qwen-vl-max", object_path3, 3, vl_max_pkl_path3, data_type)
        create_json_with_qwen(api_key_qwen, "qwen-vl-max", object_path4, 4, vl_max_pkl_path4, data_type)
        create_json_with_qwen(api_key_qwen, "qwen-vl-max", object_path6, 6, vl_max_pkl_path6, data_type)


        create_json_with_qwen(api_key_qwen, "qwen2.5-vl-72b-instruct", object_path3, 3, qwen_72b_pkl_path3, data_type)
        create_json_with_qwen(api_key_qwen, "qwen2.5-vl-72b-instruct", object_path4, 4, qwen_72b_pkl_path4, data_type)
        create_json_with_qwen(api_key_qwen, "qwen2.5-vl-72b-instruct", object_path6, 6, qwen_72b_pkl_path6, data_type)

        save_jsons([qwen_72b_pkl_path3, qwen_72b_pkl_path4, qwen_72b_pkl_path6], [f"./{type}/json_responses/qwen2_5_vl_72b/collages_3", 
                                                                                f"./{type}/json_responses/qwen2_5_vl_72b/collages_4", 
                                                                                f"./{type}/json_responses/qwen2_5_vl_72b/collages_6"])

        save_jsons([vl_max_pkl_path3, vl_max_pkl_path4, vl_max_pkl_path6], [f"./{type}/json_responses/qwen_vl_max/collages_3", 
                                                                            f"./{type}/json_responses/qwen_vl_max/collages_4", 
                                                                            f"./{type}/json_responses/qwen_vl_max/collages_6"])

        save_jsons([mistral_pkl_path3, mistral_pkl_path4, mistral_pkl_path6], [f"./{type}/json_responses/mistral_large_3/collages_3", 
                                                                            f"./{type}/json_responses/mistral_large_3/collages_4",
                                                                            f"./{type}/json_responses/mistral_large_3/collages_6"])