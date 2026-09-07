from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct", max_seq_length=4096)
print("tokenizer.eos_token_id:", tokenizer.eos_token_id)
print("eot_id token:", tokenizer.convert_tokens_to_ids("<|eot_id|>"))
print("end_of_text token:", tokenizer.convert_tokens_to_ids("<|end_of_text|>"))
