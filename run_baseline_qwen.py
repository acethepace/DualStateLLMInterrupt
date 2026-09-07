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
    with open("prompt_v2.txt", "r") as f:
        base_instructions = f.read()
    
    system_turn_start = "<|im_start|>system\n"
    user_turn_start = "<|im_end|>\n<|im_start|>user\n"
    assessor_trigger = "<|im_end|>\n<|im_start|>assistant\nClassification: "

    config = {
        "stop_word": "STOP",
        "max_new_tokens": 15,
        "assessor_prompt": assessor_trigger,
        "base_instructions": base_instructions,
        "system_turn_start": system_turn_start,
        "user_turn_start": user_turn_start
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
        full_text = config["system_turn_start"] + config["base_instructions"]
        if "transcript" in item:
            full_text += config["user_turn_start"] + "Input: " + item["transcript"]
        elif "interruption_chunk_index" in item:
            full_text += config["user_turn_start"] + "Input: " + "".join(item["base_stream"][:item["interruption_chunk_index"]])
        else:
            full_text += config["user_turn_start"] + "Input: " + item["base_stream"]
            
        full_text += "\n[USER BARGE-IN]: " + item["user_interruption"]
        full_text += config["assessor_prompt"]

        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to("cuda")

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            pred = "STOP" if "STOP" in generated_text else "CONTINUE"
                
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
    flexi_path = f"results/Qwen_Qwen3-4B-Instruct-2507/flexi_baseline_{timestamp}.json"
    fdb_path = f"results/Qwen_Qwen3-4B-Instruct-2507/fdb_baseline_{timestamp}.json"
    
    with open(flexi_path, "w") as f:
        json.dump(flexi_res, f, indent=2)
    with open(fdb_path, "w") as f:
        json.dump(fdb_res, f, indent=2)

    print(f"\nSaved results to:\n- {flexi_path}\n- {fdb_path}")

if __name__ == "__main__":
    main()
