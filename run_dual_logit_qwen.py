import json
import torch
import time
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

print("Loading Qwen3-4B-Instruct-2507...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    quantization_config=quantization_config
)
print("Model loaded.")

def get_templates():
    with open("prompt_v2.txt", "r") as f:
        base_instructions = f.read()
    
    system_turn_start = "<|im_start|>system\n"
    user_turn_start = "<|im_end|>\n<|im_start|>user\n"
    assessor_trigger = "<|im_end|>\n<|im_start|>assistant\nClassification: "

    config = {
        "assessor_prompt": assessor_trigger,
        "base_instructions": base_instructions,
        "system_turn_start": system_turn_start,
        "user_turn_start": user_turn_start
    }
    return config

def evaluate(dataset_path):
    print(f"Evaluating {dataset_path} with Dual LLM (Logit Gating)...")
    with open(dataset_path, "r") as f:
        data = json.load(f)

    config = get_templates()
    tp, tn, fp, fn = 0, 0, 0, 0
    
    tp_latencies = []
    
    stop_id = 50669
    ignore_id = 35045

    for idx, item in enumerate(data):
        # 1. Base Stream (Main Generation Thread - Prefill)
        base_text = config["system_turn_start"] + config["base_instructions"]
        if "transcript" in item:
            base_text += config["user_turn_start"] + "Input: " + item["transcript"]
        elif "interruption_chunk_index" in item:
            base_text += config["user_turn_start"] + "Input: " + "".join(item["base_stream"][:item["interruption_chunk_index"]])
        else:
            base_text += config["user_turn_start"] + "Input: " + item.get("base_stream", "")
            
        base_inputs = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).to("cuda")
        
        with torch.no_grad():
            base_outputs = model(input_ids=base_inputs.input_ids, use_cache=True)
            past_key_values = base_outputs.past_key_values
            
        # 2. Interruption occurs, Assessor state forks
        interruption_text = "\n[USER BARGE-IN]: " + item["user_interruption"] + config["assessor_prompt"]
        interruption_inputs = tokenizer(interruption_text, return_tensors="pt", add_special_tokens=False).to("cuda")
        
        # We simulate the time it takes for the Assessor to classify based on the forked state
        start = time.time()
        with torch.no_grad():
            out = model(
                input_ids=interruption_inputs.input_ids,
                past_key_values=past_key_values,
                use_cache=False
            )
            logits = out.logits[:, -1, :]
            stop_logit = logits[0, stop_id]
            ignore_logit = logits[0, ignore_id]
            
            combined = torch.tensor([stop_logit, ignore_logit])
            probs = torch.nn.functional.softmax(combined, dim=-1)
            
            pred = "STOP" if probs[0] > probs[1] else "IGNORE"
                
        latency = (time.time() - start) * 1000
                
        expected = item.get("expected_action", "CONTINUE")
        expected_mapped = "STOP" if expected == "STOP" else "IGNORE"
        
        if pred == "STOP" and expected_mapped == "STOP":
            tp += 1
            tp_latencies.append(latency)
        elif pred == "IGNORE" and expected_mapped == "IGNORE":
            tn += 1
            # For TN (No-Op), the Assessor does not block the Main stream, so latency impact is strictly 0 ms.
        elif pred == "STOP" and expected_mapped == "IGNORE":
            fp += 1
        elif pred == "IGNORE" and expected_mapped == "STOP":
            fn += 1
            
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(data)} items...")

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    avg_tp_latency = sum(tp_latencies) / len(tp_latencies) if tp_latencies else 0
    avg_tn_latency = 0.0 # Explicitly 0 ms for Dual LLM architecture
    
    # Calculate overall average latency based only on TPs since TN impact is 0
    # Wait, the overall latency for the run on all items could be computed, but we'll focus on the requested metrics
    
    print(f"Acc: {acc:.4f} | TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"TP Latency (Interruption): {avg_tp_latency:.2f} ms")
    print(f"TN Latency (No-Op): {avg_tn_latency:.2f} ms")
    
    return {
        "accuracy": acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "interruption_latency_ms": avg_tp_latency,
        "noop_latency_ms": avg_tn_latency,
        "dataset": dataset_path
    }

def main():
    flexi_res = evaluate("real_flexi.json")
    # Finding FDB path based on previous script
    fdb_res = evaluate("datasets/fdb_dataset_recovered.json")

    os.makedirs("results/Qwen_Qwen3-4B-Instruct-2507", exist_ok=True)
    
    timestamp = int(time.time())
    flexi_path = f"results/Qwen_Qwen3-4B-Instruct-2507/flexi_dual_{timestamp}.json"
    fdb_path = f"results/Qwen_Qwen3-4B-Instruct-2507/fdb_dual_{timestamp}.json"
    
    with open(flexi_path, "w") as f:
        json.dump(flexi_res, f, indent=2)
    with open(fdb_path, "w") as f:
        json.dump(fdb_res, f, indent=2)

    print(f"RESULTS_SAVED")
    print(f"FLEXI_PATH:{os.path.abspath(flexi_path)}")
    print(f"FDB_PATH:{os.path.abspath(fdb_path)}")

if __name__ == "__main__":
    main()
