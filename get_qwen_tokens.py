from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
print("STOP:", tokenizer.encode("STOP", add_special_tokens=False))
print("CONTINUE:", tokenizer.encode("CONTINUE", add_special_tokens=False))
