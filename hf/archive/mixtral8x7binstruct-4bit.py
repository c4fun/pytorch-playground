import os
from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
model_name_or_path = os.path.expanduser("~/models/TheBloke/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf")
llm = AutoModelForCausalLM.from_pretrained(model_name_or_path, gpu_layers=50)

print(llm("11 developers and 1 tester went to the bar."))
