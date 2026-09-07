from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
print("eos_token_id:", tokenizer.eos_token_id)
print("eot_id token id:", tokenizer.convert_tokens_to_ids("<|eot_id|>"))
