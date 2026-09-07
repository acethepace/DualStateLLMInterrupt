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
    with open("prompt_v2.txt", "r") as f:
        base_instructions = f.read()
    
    config = {
        "base_instructions": base_instructions,
    }
    return config

def evaluate(dataset_path):
    print(f"Evaluating {dataset_path} with Logit Gating...")
    with open(dataset_path, "r") as f:
        data = json.load(f)

    config = get_templates()
    tp, tn, fp, fn = 0, 0, 0, 0
    
    tp_latencies = []
    tn_latencies = []
    total_latencies = []

    stop_id = 82652
    ignore_id = 107968

    for idx, item in enumerate(data):
        user_input = ""
        if "transcript" in item:
            user_input = "Input: " + item["transcript"]
        elif "interruption_chunk_index" in item:
            user_input = "Input: " + "".join(item["base_stream"][:item["interruption_chunk_index"]])
        else:
            user_input = "Input: " + item.get("base_stream", "")
            
        user_input += "\n[USER BARGE-IN]: " + item["user_interruption"]
        
        messages = [
            {"role": "user", "content": config["base_instructions"] + "\n\n" + user_input}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Note: We do NOT force thinking here. We just want to extract the logit for the very first token emitted.
        
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")

        start = time.time()
        with torch.no_grad():
            out = model(input_ids=inputs.input_ids)
            logits = out.logits[:, -1, :]
            stop_logit = logits[0, stop_id]
            ignore_logit = logits[0, ignore_id]
            
            combined = torch.tensor([stop_logit, ignore_logit])
            probs = torch.nn.functional.softmax(combined, dim=-1)
            
            pred = "STOP" if probs[0] > probs[1] else "IGNORE"
                
        latency = (time.time() - start) * 1000
        total_latencies.append(latency)
                
        expected = item.get("expected_action", "CONTINUE")
        expected_mapped = "STOP" if expected == "STOP" else "IGNORE"
        
        if pred == "STOP" and expected_mapped == "STOP":
            tp += 1
            tp_latencies.append(latency)
        elif pred == "IGNORE" and expected_mapped == "IGNORE":
            tn += 1
            tn_latencies.append(latency)
        elif pred == "STOP" and expected_mapped == "IGNORE":
            fp += 1
        elif pred == "IGNORE" and expected_mapped == "STOP":
            fn += 1
            
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(data)} items...")
            acc = (tp + tn) / (tp + tn + fp + fn)
            print(f"Current Acc: {acc:.4f}")

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
    fdb_res = evaluate("datasets/fdb_dataset_recovered.json")

    os.makedirs("results/unsloth_gemma-4-12b-it", exist_ok=True)
    
    timestamp = int(time.time())
    flexi_path = f"results/unsloth_gemma-4-12b-it/flexi_logit_{timestamp}.json"
    fdb_path = f"results/unsloth_gemma-4-12b-it/fdb_logit_{timestamp}.json"
    
    with open(flexi_path, "w") as f:
        json.dump(flexi_res, f, indent=2)
    with open(fdb_path, "w") as f:
        json.dump(fdb_res, f, indent=2)

    print(f"\nSaved results to:\n- {flexi_path}\n- {fdb_path}")
    print(f"RESULTS_SAVED")
    print(f"FLEXI_PATH:{os.path.abspath(flexi_path)}")
    print(f"FDB_PATH:{os.path.abspath(fdb_path)}")

if __name__ == "__main__":
    main()
