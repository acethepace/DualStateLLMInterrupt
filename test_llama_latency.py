import torch
import time
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# 1000 tokens of text
text = "hello " * 1000
inputs = tokenizer(text=text, return_tensors="pt", add_special_tokens=False).to("cuda")

start = time.time()
with torch.no_grad():
    model.generate(input_ids=inputs.input_ids, max_new_tokens=1, use_cache=True, pad_token_id=tokenizer.eos_token_id)
lat = (time.time() - start) * 1000
print(f"Llama 1000-token Prefill + 1 token generation: {lat:.2f} ms")
