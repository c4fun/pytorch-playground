import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model_name_or_path = os.path.expanduser("~/models/TheBloke/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf")

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_4bit=True)

text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pt").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
