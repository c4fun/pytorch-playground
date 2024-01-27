from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

device = ("cuda" if torch.cuda.is_available() else "cpu")
# Cannot run because the model is too large
model_name_or_path = os.path.expanduser("~/models/TheBloke/mixtral-8x7b-instruct/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf")
max_new_tokens = 500
top_p = 0.9
temperature = 0.35
repetition_penalty = 1.0

torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    load_in_4bit=True,
    device_map=device
)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

text = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."

inst_begin_tokens = tokenizer.encode('[INST]', add_special_tokens=False)
inst_end_tokens = tokenizer.encode('[/INST]', add_special_tokens=False)
human_tokens = tokenizer.encode(text, add_special_tokens=False)
input_ids = [tokenizer.bos_token_id] + inst_begin_tokens + human_tokens + inst_end_tokens

# input_ids = human_tokens
input_ids = torch.tensor([input_ids], dtype=torch.long).cuda()

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
        top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id
    )
outputs = outputs.tolist()[0][len(input_ids[0]):]
response = tokenizer.decode(outputs)
response = response.strip().replace(tokenizer.eos_token, "").strip()
print("Chatbot：{}".format(response))
