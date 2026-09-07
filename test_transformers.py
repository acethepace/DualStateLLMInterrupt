import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
print("Successfully loaded model without unsloth Triton kernels!")
