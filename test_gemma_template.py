from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-4-12b-it")
messages = [{"role": "user", "content": "Hello"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(repr(text))
