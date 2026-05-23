import dim_reduction_solved_3d_model
import llm_benchmarks
import evaluation

DASHSCOPE_API_KEY = ""
MISTRAL_API_KEY = ""
OPENROUTER_API_KEY = ""

dim_reduction_solved_3d_model.dim_reduction()
llm_benchmarks.llm_benchmark(DASHSCOPE_API_KEY, MISTRAL_API_KEY)
evaluation.evaluate(api_key=OPENROUTER_API_KEY)