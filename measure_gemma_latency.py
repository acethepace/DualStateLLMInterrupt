import torch
import time
from unsloth import FastLanguageModel

model_name = "unsloth/gemma-4-12b-it"
max_seq_length = 4096

print("Loading Gemma 4 12B...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Simulate 2000 token context
text = "hello world " * 1000
inputs = tokenizer(text, return_tensors="pt").to("cuda")
print(f"Context length: {inputs.input_ids.shape[1]}")

# 1. Independent Decider Model (Full context + reasoning generation)
print("Measuring Independent Decider Model (Prefill + 15 tokens)...")
start = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=15, use_cache=True)
independent_latency = (time.time() - start) * 1000
print(f"Latency: {independent_latency:.2f} ms")

# 2. Independent + Prefilled Thinking / Logit Gating (Full context + 1 token)
print("Measuring Independent + Logit Gating (Prefill + 1 token)...")
start = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1, use_cache=True)
independent_1tok_latency = (time.time() - start) * 1000
print(f"Latency: {independent_1tok_latency:.2f} ms")

# 3. Dual State + Logit Gating (0 prefill + 1 token)
print("Measuring Dual State (0 prefill + 1 token)...")
# Get KV cache first
with torch.no_grad():
    base_outputs = model(**inputs, use_cache=True)
kv_cache = base_outputs.past_key_values

new_inputs = tokenizer("STOP", return_tensors="pt").to("cuda")
start = time.time()
with torch.no_grad():
    out = model(input_ids=new_inputs.input_ids, past_key_values=kv_cache, use_cache=True)
dual_state_latency = (time.time() - start) * 1000
print(f"Latency: {dual_state_latency:.2f} ms")

