import json
import torch
import time
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    quantization_config=quantization_config
)

def evaluate(prompt_file="prompt_llama.txt", dataset_path="real_flexi.json", limit=None):
    with open(prompt_file, "r") as f:
        base_instructions = f.read()

    with open(dataset_path, "r") as f:
        data = json.load(f)
        if limit:
            data = data[:limit]

    tp, tn, fp, fn = 0, 0, 0, 0

    for idx, item in enumerate(data):
        # Build prompt using apply_chat_template or manually.
        # Llama 3 uses a specific format. Let's do it manually to allow appending '[USER BARGE-IN]' and trailing parts easily
        
        system_turn_start = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        user_turn_start = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        assessor_trigger = "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        full_text = system_turn_start + base_instructions
        if "transcript" in item:
            full_text += user_turn_start + "Input: " + item["transcript"]
        elif "interruption_chunk_index" in item:
            full_text += user_turn_start + "Input: " + "".join(item["base_stream"][:item["interruption_chunk_index"]])
        else:
            full_text += user_turn_start + "Input: " + item["base_stream"]
            
        full_text += "\n[USER BARGE-IN]: " + item["user_interruption"]
        full_text += assessor_trigger

        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Simple heuristic mapping for Llama output
            pred = "CONTINUE"
            if "STOP" in generated_text.upper():
                pred = "STOP"
            elif "IGNORE" in generated_text.upper():
                pred = "CONTINUE"
                
        expected = item.get("expected_action", "CONTINUE")
        if pred == "STOP" and expected == "STOP":
            tp += 1
        elif pred == "CONTINUE" and expected == "CONTINUE":
            tn += 1
        elif pred == "STOP" and expected == "CONTINUE":
            fp += 1
        elif pred == "CONTINUE" and expected == "STOP":
            fn += 1
            
        if (idx + 1) % 20 == 0:
            print(f"[{idx+1}/{len(data)}] Acc so far: {(tp+tn)/(tp+tn+fp+fn):.2f}, Out: {generated_text}")

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    print(f"Final Acc: {acc:.4f} | TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

evaluate(limit=50)
