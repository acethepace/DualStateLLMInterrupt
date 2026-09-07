import json
import torch
import time
import os
import sys

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
    base_instructions = """You are a Semantic Voice Activity Detection (VAD) classifier.
Your ONLY job is to classify the ongoing audio stream as EXACTLY 'STOP' or 'IGNORE'.

RULES:
1. BENIGN BACKCHANNEL (Output 'IGNORE'): If the user utters a short backchannel acknowledging your speech (e.g., 'uh huh', 'okay', 'yeah', 'right', 'hmm', 'Ah, okay.', 'I see.', 'Makes sense.').
2. INTENTIONAL BARGE-IN (Output 'STOP'): If the user asks a new question, corrects a fact, changes the topic, adds a new constraint, or tells the system to wait/stop. 

CRITICAL: 
- If the user says things like "Oh, and it also needs to be gluten-free", they are adding a constraint. You MUST output 'STOP'.
- If the user says "Never mind...", they are changing the topic. You MUST output 'STOP'.
- If the user asks a question like "Hold on, who is considered...", you MUST output 'STOP'.

EXAMPLES:
[USER BARGE-IN]: Okay.
Classification: IGNORE

[USER BARGE-IN]: Right.
Classification: IGNORE

[USER BARGE-IN]: Oh, and it also needs to be gluten-free.
Classification: STOP

[USER BARGE-IN]: Never mind sci-fi, switch to the best fantasy novels instead.
Classification: STOP

[USER BARGE-IN]: Sorry to interrupt, but is it theoretically possible to travel through one?
Classification: STOP"""
    
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
    print(f"Evaluating {dataset_path} with Logit Gating...")
    with open(dataset_path, "r") as f:
        data = json.load(f)

    config = get_templates()
    tp, tn, fp, fn = 0, 0, 0, 0
    
    tp_latencies = []
    tn_latencies = []
    total_latencies = []
    
    stop_id = 50669
    ignore_id = 35045

    for idx, item in enumerate(data):
        full_text = config["system_turn_start"] + config["base_instructions"]
        if "transcript" in item:
            full_text += config["user_turn_start"] + "Input: " + item["transcript"]
        elif "interruption_chunk_index" in item:
            full_text += config["user_turn_start"] + "Input: " + "".join(item["base_stream"][:item["interruption_chunk_index"]])
        else:
            full_text += config["user_turn_start"] + "Input: " + item.get("base_stream", "")
            
        full_text += "\n[USER BARGE-IN]: " + item["user_interruption"]
        full_text += config["assessor_prompt"]

        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to("cuda")

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
        # In FDB/FLEXI, "CONTINUE" is often the ground truth label for benign backchannels.
        # But we classify it as IGNORE in prediction. So let's normalize expected to STOP vs IGNORE.
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
        "dataset": dataset_path
    }

def main():
    flexi_res = evaluate("real_flexi.json")
    fdb_res = evaluate("datasets/fdb_dataset_recovered.json")

    os.makedirs("results/Qwen_Qwen3-4B-Instruct-2507", exist_ok=True)
    
    timestamp = int(time.time())
    flexi_path = f"results/Qwen_Qwen3-4B-Instruct-2507/flexi_logit_{timestamp}.json"
    fdb_path = f"results/Qwen_Qwen3-4B-Instruct-2507/fdb_logit_{timestamp}.json"
    
    with open(flexi_path, "w") as f:
        json.dump(flexi_res, f, indent=2)
    with open(fdb_path, "w") as f:
        json.dump(fdb_res, f, indent=2)

    print(f"RESULTS_SAVED")
    print(f"FLEXI_PATH:{os.path.abspath(flexi_path)}")
    print(f"FDB_PATH:{os.path.abspath(fdb_path)}")

if __name__ == "__main__":
    main()
