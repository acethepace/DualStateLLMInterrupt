import json
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prompt_file", type=str, required=True)
args = parser.parse_args()

model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    quantization_config=quantization_config
)

with open("qwen_subset.json", "r") as f:
    data = json.load(f)

with open(args.prompt_file, "r") as f:
    base_instructions = f.read()

system_turn_start = "<|im_start|>system\n"
user_turn_start = "<|im_end|>\n<|im_start|>user\n"
assessor_trigger = "<|im_end|>\n<|im_start|>assistant\nClassification: "

stop_id = 50669
ignore_id = 35045

tp = 0
tn = 0
fp = 0
fn = 0

for item in data:
    full_text = system_turn_start + base_instructions
    full_text += user_turn_start + "Input: " + "".join(item["base_stream"][:item["interruption_chunk_index"]])
    full_text += "\n[USER BARGE-IN]: " + item["user_interruption"]
    full_text += assessor_trigger

    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to("cuda")

    with torch.no_grad():
        out = model(input_ids=inputs.input_ids)
        logits = out.logits[:, -1, :]
        stop_logit = logits[0, stop_id]
        ignore_logit = logits[0, ignore_id]
        
        combined = torch.tensor([stop_logit, ignore_logit])
        probs = torch.nn.functional.softmax(combined, dim=-1)
        
        if probs[0] > probs[1]:
            pred = "STOP"
        else:
            pred = "IGNORE"
            
    expected = item["expected_action"]
    print(f"User: {item['user_interruption']}")
    print(f"Pred: {pred} | Expected: {expected} | Probs: STOP={probs[0]:.2f}, IGNORE={probs[1]:.2f}")
    if pred == "STOP" and expected == "STOP":
        tp += 1
    elif pred == "IGNORE" and expected == "IGNORE":
        tn += 1
    elif pred == "STOP" and expected == "IGNORE":
        fp += 1
    elif pred == "IGNORE" and expected == "STOP":
        fn += 1

print(f"\nSubset Accuracy: {(tp+tn)/(tp+tn+fp+fn):.2f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
