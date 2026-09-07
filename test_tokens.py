from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-4-12b-it")
print(tokenizer.decode([82652]))
print(tokenizer.decode([107968]))
