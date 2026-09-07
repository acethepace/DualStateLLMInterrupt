import sys

with open("dual_state_interruptor.py", "r") as f:
    code = f.read()

code = code.replace('if "llama" in model_name.lower():', 'if "llama" in model_name.lower() or "qwen" in model_name.lower():\n        from transformers import BitsAndBytesConfig\n        quant_config = BitsAndBytesConfig(load_in_4bit=True)\n')
code = code.replace('device_map="auto"', 'device_map="auto", quantization_config=quant_config')

with open("dual_state_interruptor.py", "w") as f:
    f.write(code)
