import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "unsloth/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    quantization_config=quantization_config
)

# 1000 tokens of text
text = "hello " * 1000
inputs = tokenizer(text=text, return_tensors="pt", add_special_tokens=False).to("cuda")

# Warmup
with torch.no_grad():
    model.generate(input_ids=inputs.input_ids[:, :10], max_new_tokens=1)

# FDB 1000-token Prefill + 1 token generation
start = time.time()
with torch.no_grad():
    model.generate(input_ids=inputs.input_ids, max_new_tokens=1, use_cache=True, pad_token_id=tokenizer.eos_token_id)
fdb_lat = (time.time() - start) * 1000

# FLEXI 50-token Prefill + 1 token generation
inputs_flexi = tokenizer(text="hello "*50, return_tensors="pt", add_special_tokens=False).to("cuda")
start = time.time()
with torch.no_grad():
    model.generate(input_ids=inputs_flexi.input_ids, max_new_tokens=1, use_cache=True, pad_token_id=tokenizer.eos_token_id)
flexi_lat = (time.time() - start) * 1000

# FLEXI 50-token Prefill + 15 token generation
start = time.time()
with torch.no_grad():
    model.generate(input_ids=inputs_flexi.input_ids, max_new_tokens=15, use_cache=True, pad_token_id=tokenizer.eos_token_id)
flexi_lat_15 = (time.time() - start) * 1000

# Dual LLM Assessor Latency (0 prefill)
inputs_dual = tokenizer(text="hello ", return_tensors="pt", add_special_tokens=False).to("cuda")
start = time.time()
with torch.no_grad():
    model.generate(input_ids=inputs_dual.input_ids, max_new_tokens=1, use_cache=True, pad_token_id=tokenizer.eos_token_id)
dual_lat = (time.time() - start) * 1000

print(f"Independent V1 (FLEXI 15-tokens): {flexi_lat_15:.2f} ms")
print(f"Independent V2/V4 (FLEXI 1-token): {flexi_lat:.2f} ms")
print(f"Independent V2/V4 (FDB 1000-tokens): {fdb_lat:.2f} ms")
print(f"Dual V4 (0 prefill + 1-token): {dual_lat:.2f} ms")
