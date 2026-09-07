from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-4-12b-it")
messages = [{"role": "user", "content": "BASE\n[USER BARGE-IN]: INT"}]
full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

base_text = "<bos><|turn>user\nBASE"
interruption_text = "\n[USER BARGE-IN]: INT<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
manual = base_text + interruption_text
print("MATCH:", full == manual)
