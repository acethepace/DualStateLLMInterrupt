import json
import torch
import time
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "unsloth/gemma-4-12b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

print("Loading Gemma-4-12B-IT...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    quantization_config=quantization_config
)
print("Model loaded.")

def get_templates():
    # Prompt that elicited structured reasoning. We'll use v1 or v2, v2 might be better as long as we allow thought.
    # Let's try v2.
    with open("prompt_v2.txt", "r") as f:
        base_instructions = f.read()
    
    config = {
        "max_new_tokens": 100, # Allow enough for thought and STOP/IGNORE
        "base_instructions": base_instructions,
    }
    return config

def evaluate(dataset_path):
    print(f"Evaluating {dataset_path} with Baseline...")
    with open(dataset_path, "r") as f:
        data = json.load(f)

    config = get_templates()
    tp, tn, fp, fn = 0, 0, 0, 0
    
    tp_latencies = []
    tn_latencies = []
    total_latencies = []

    for idx, item in enumerate(data):
        user_input = ""
        if "transcript" in item:
            user_input = "Input: " + item["transcript"]
        elif "interruption_chunk_index" in item:
            user_input = "Input: " + "".join(item["base_stream"][:item["interruption_chunk_index"]])
        else:
            user_input = "Input: " + item["base_stream"]
            
        user_input += "\n[USER BARGE-IN]: " + item["user_interruption"]
        
        messages = [
            {"role": "user", "content": config["base_instructions"] + "\n\n" + user_input}
        ]
        
        # Apply chat template
        # gemma-4-12b-it chat template will add <|turn>user\n...<turn|>\n<|turn>model\n
        # If enable_thinking is True, it will add <|channel>thought\n
        # Wait, the default chat template of gemma-4-12b-it might need enable_thinking=True, but let's see.
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Force it to think if it doesn't automatically.
        if "<|channel>thought\n<channel|>" in text:
            # Replace the closed thought block with an open one to force thinking.
            text = text.replace("<|channel>thought\n<channel|>", "<|channel>thought\n")
        
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")

        start = time.time()
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
                
        latency = (time.time() - start) * 1000
        total_latencies.append(latency)
                
        expected = item.get("expected_action", "CONTINUE")
        if pred == "STOP" and expected == "STOP":
            tp += 1
            tp_latencies.append(latency)
        elif pred == "CONTINUE" and expected == "CONTINUE":
            tn += 1
            tn_latencies.append(latency)
        elif pred == "STOP" and expected == "CONTINUE":
            fp += 1
        elif pred == "CONTINUE" and expected == "STOP":
            fn += 1
            
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(data)} items...")
            print(f"Current Acc: {(tp + tn) / (tp + tn + fp + fn):.4f}")

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    avg_latency = sum(total_latencies) / len(total_latencies) if total_latencies else 0
    avg_tp_latency = sum(tp_latencies) / len(tp_latencies) if tp_latencies else 0
    avg_tn_latency = sum(tn_latencies) / len(tn_latencies) if tn_latencies else 0
    
    print(f"Acc: {acc:.4f} | TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"TP Latency (Interruption): {avg_tp_latency:.2f} ms")
    print(f"TN Latency (No-Op): {avg_tn_latency:.2f} ms")
    
    return {
        "accuracy": acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "avg_latency_ms": avg_latency,
        "interruption_latency_ms": avg_tp_latency,
        "noop_latency_ms": avg_tn_latency,
        "dataset": dataset_path,
        "prompt": config["base_instructions"]
    }

def main():
    flexi_res = evaluate("real_flexi.json")
    if flexi_res["accuracy"] < 0.90:
        print("WARNING: FLEXI Accuracy below 90%. Prompt iteration might be needed.")
        
    fdb_res = evaluate("datasets/fdb_dataset_recovered.json")
    if fdb_res["accuracy"] < 0.90:
        print("WARNING: FDB Accuracy below 90%. Prompt iteration might be needed.")

    os.makedirs("results/unsloth_gemma-4-12b-it", exist_ok=True)
    
    timestamp = int(time.time())
    flexi_path = f"results/unsloth_gemma-4-12b-it/flexi_baseline_{timestamp}.json"
    fdb_path = f"results/unsloth_gemma-4-12b-it/fdb_baseline_{timestamp}.json"
    
    with open(flexi_path, "w") as f:
        json.dump(flexi_res, f, indent=2)
    with open(fdb_path, "w") as f:
        json.dump(fdb_res, f, indent=2)

    print(f"\nSaved results to:\n- {flexi_path}\n- {fdb_path}")
    
    print(f"FLEXI Accuracy: {flexi_res['accuracy']:.4f}")
    print(f"FLEXI Avg Latency: {flexi_res['avg_latency_ms']:.2f}")
    print(f"FDB Accuracy: {fdb_res['accuracy']:.4f}")
    print(f"FDB Avg Latency: {fdb_res['avg_latency_ms']:.2f}")

if __name__ == "__main__":
    main()
