import torch
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
ids = tokenizer("CONTINUE<|eot_id|><|start_header_id|>", add_special_tokens=False).input_ids
print("Tokens:", [tokenizer.decode([i]) for i in ids])
print("IDs:", ids)
