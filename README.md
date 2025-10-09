# CAD2TechSpec

![CAD2TechSpec overview](https://github.com/Maxiiiiim/3dLLM_Lambda/blob/main/Framework%20CAD2TechSpec.png)

CAD2TechSpec is a novel framework for automating design processes within CAD systems by leveraging multimodal large language models (LLMs). The framework enables the analysis and generation of detailed design specifications, including the automated creation of machining process plans. Our system architecture combines 3D model rendering, dimensionality reduction techniques, and the capabilities of multimodal LLMs to produce structured JSON representations of manufacturing workflows. Experiments conducted on the ABC dataset demonstrate that CAD2TechSpec significantly reduces design time while enhancing the accuracy and completeness of technical specifications. The proposed approach holds considerable promise for high-tech industries such as aerospace and mechanical engineering, where efficiency and precision in design processes are critical.

## Dataset
**ABC Data.** The primary data utilized for the CAD2TechSpec framework were sourced from the [ABC Dataset](https://doi.ieeecomputersociety.org/10.1109/CVPR.2019.00983). The project pipeline employed two distinct data types: Obj and Stats. The Obj files contain models with ground truth normals and curvature values at each vertex, serving as the main input data for the framework. In contrast, the Stats files provide statistical information about the CAD model in parametric boundary representation, which was used for selecting single-component parts within the CAD representation. A detailed description of all data contained in the dataset is available at the following link: https://deep-geometry.github.io/abc-dataset/.

**ISO Data.** The external data for RAG consist of normative documents developed by the [International Organization for Standardization (ISO)](https://www.iso.org/home.html). The knowledge base is structured as a table comprising three key columns: equipment used in technical operations, the corresponding ISO standard title, and its official document number. The collected data were obtained from open-source resources in collaboration with subject matter experts and systematically organized within a structured tabular format, as documented in the file **'example_material/equipment_iso.csv'**.

## Repository Description
The 'abc_dataset' folder contains data from the [ABC dataset](https://archive.nyu.edu/handle/2451/44309);

The 'example_material' folder contains the following:
   - The folders **'collages_3'**, **'collages_4'**, **'collages_6'** contain image collages consisting of 3, 4, or 6 images, respectively, created using Isomap.
   - The **'json_standard'** folder contains reference solutions in the form of JSON files.
   - The **'prompts'** folder contains prompts for generating JSON files describing mechanical processing planning processes for parts.   
   - The **'rendered_imgs'** folder contains multi-view images of the part.
   - The **'example_material/equipment_iso.csv'** file contains data of normative documents developed by the International Organization for Standardization.
   - The **'example_object_path.pkl'** file contains an array of all files in .obj format.

## Framework launch process
First, you need to run the scripts **'render_script_type1.py'** and **'render_script_type2.py'** to render the 3D model into 28 images.
   - [Blender](https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/blender.zip) must be installed and placed in the root of the project beforehand.
   - Commands to run the scripts:
```
# Create 8 views
blender -b -P render_script_type1.py -- --object_path_pkl './example_material/example_object_path.pkl' --parent_dir './example_material'

# Create 20 views
blender -b -P render_script_type2.py -- --object_path_pkl './example_material/example_object_path.pkl' --parent_dir './example_material'
```
The scripts **'render_script_type1.py'** and **'render_script_type2.py'** were adapted from the automatic method Cap3D (available at https://github.com/tiangeluo/DiffuRank?tab=readme-ov-file).

Next, you should run the **'framework.py'** file, which performs the following tasks:
   - Selects 3, 4, or 6 different images of the model from various angles and combines them into a single image.
   - Generates JSON files describing the planning processes for mechanical part machining. The inputs for the model are a prompt and the previously created multi-view collage. The Pixtral 12B, Qwen2.5-VL-72B, and Qwen-VL-Max models are used for JSON generation.
   - Evaluates the quality of the generated JSON files against specified criteria using the GPT-4o mini model.

## Requirements
This project requires the following **Python libraries** to be installed:
   - **matplotlib**
   - **numpy**
   - **pandas**
   - **scikit-learn**
   - **Pillow (PIL)**
   - **scipy**
   - **openai**
   - **mistralai**   
```
pip install openai llm_benchmarks matplotlib numpy pandas scikit-learn Pillow scipy mistralai
```  
**Blender Environment:**
   - **Blender** (recommended version 3.4.1 Linux x64) — required to run scripts that use `bpy` and `mathutils` libraries
   - **bpy & mathutils** — Blender's Python API libraries included with Blender  
