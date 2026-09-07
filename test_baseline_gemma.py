import json
import torch
import time
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "unsloth/gemma-4-12b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    quantization_config=quantization_config
)

def get_templates():
    with open("prompt_v2.txt", "r") as f:
        base_instructions = f.read()
    return {"max_new_tokens": 100, "base_instructions": base_instructions}

with open("real_flexi.json", "r") as f:
    data = json.load(f)[:10]

config = get_templates()
for item in data:
    user_input = ""
    if "transcript" in item:
        user_input = "Input: " + item["transcript"]
    elif "interruption_chunk_index" in item:
        user_input = "Input: " + "".join(item["base_stream"][:item["interruption_chunk_index"]])
    else:
        user_input = "Input: " + item["base_stream"]
    
    user_input += "\n[USER BARGE-IN]: " + item["user_interruption"]
    
    messages = [{"role": "user", "content": config["base_instructions"] + "\n\n" + user_input}]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if "<|channel>thought\n<channel|>" in text:
        text = text.replace("<|channel>thought\n<channel|>", "<|channel>thought\n")
    
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config["max_new_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
    pred = "CONTINUE"
    if "STOP" in generated_text.upper():
        pred = "STOP"
    elif "IGNORE" in generated_text.upper():
        pred = "CONTINUE"
    elif "CONTINUE" in generated_text.upper():
        pred = "CONTINUE"
        
    print(f"Expected: {item.get('expected_action', 'CONTINUE')}, Pred: {pred}")
    print(f"Generated: {generated_text}\n")
