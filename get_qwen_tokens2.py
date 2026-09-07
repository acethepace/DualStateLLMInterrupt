from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
words = ["STOP", "CONTINUE", "IGNORE", "WAIT", "PROCEED", "HALT", "YES", "NO", "A", "B"]
for word in words:
    print(f"{word}: {tokenizer.encode(word, add_special_tokens=False)}")
