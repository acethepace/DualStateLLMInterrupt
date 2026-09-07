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

def get_templates(version):
    base_v1 = open("prompt_v2.txt").read()
    
    with open("prompt_v2.txt", "r") as f:
        base_v4 = f.read()
    
    system_turn_start = "<|im_start|>system\n"
    user_turn_start = "<|im_end|>\n<|im_start|>user\n"
    assessor_trigger = "<|im_end|>\n<|im_start|>assistant\nClassification: "

    config = {}
    if version == "v1":
        config["stop_word"] = "STOP"
        config["max_new_tokens"] = 15
        config["assessor_prompt"] = assessor_trigger
        config["base_instructions"] = base_v1
    elif version == "v2_prefill":
        config["stop_word"] = "STOP"
        config["max_new_tokens"] = 1
        config["assessor_prompt"] = assessor_trigger
        config["base_instructions"] = base_v4

    config["system_turn_start"] = system_turn_start
    config["user_turn_start"] = user_turn_start
    return config

def evaluate(dataset_path, version):
    print(f"Evaluating {dataset_path} with {version}...")
    with open(dataset_path, "r") as f:
        data = json.load(f)

    config = get_templates(version)
    tp, tn, fp, fn = 0, 0, 0, 0
    total_latency = 0
    
    stop_id = 50669
    ignore_id = 35045

    for item in data:
        full_text = config["system_turn_start"] + config["base_instructions"]
        if "interruption_chunk_index" in item:
            full_text += config["user_turn_start"] + "Input: " + "".join(item["base_stream"][:item["interruption_chunk_index"]])
        else:
            full_text += config["user_turn_start"] + "Input: " + item["base_stream"]
            
        full_text += "\n[USER BARGE-IN]: " + item["user_interruption"]
        full_text += config["assessor_prompt"]

        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to("cuda")

        start = time.time()
        with torch.no_grad():
            if version == "v1":
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                pred = "STOP" if "STOP" in generated_text else "CONTINUE"
            else:
                out = model(input_ids=inputs.input_ids)
                logits = out.logits[:, -1, :]
                stop_logit = logits[0, stop_id]
                ignore_logit = logits[0, ignore_id]
                
                combined = torch.tensor([stop_logit, ignore_logit])
                probs = torch.nn.functional.softmax(combined, dim=-1)
                
                pred = "STOP" if probs[0] > probs[1] else "CONTINUE"
                
        latency = (time.time() - start) * 1000
        total_latency += latency
                
        expected = item.get("expected_action", "CONTINUE")
        if pred == "STOP" and expected == "STOP":
            tp += 1
        elif pred == "CONTINUE" and expected == "CONTINUE":
            tn += 1
        elif pred == "STOP" and expected == "CONTINUE":
            fp += 1
        elif pred == "CONTINUE" and expected == "STOP":
            fn += 1

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    avg_latency = total_latency / len(data) if len(data) > 0 else 0
    print(f"{version} Acc: {acc:.4f} | TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn} | Avg Latency: {avg_latency:.2f} ms")
    return {"accuracy": acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "avg_latency_ms": avg_latency}

flexi_v1 = evaluate("real_flexi.json", "v1")
flexi_v4 = evaluate("real_flexi.json", "v2_prefill")
# fdb_v1 = evaluate("datasets/fdb_dataset.json", "v1")
# fdb_v4 = evaluate("datasets/fdb_dataset.json", "v2_prefill")

os.makedirs("results/Qwen_Qwen3-4B-Instruct-2507", exist_ok=True)
res = {
    "flexi": {"v1": flexi_v1, "v4": flexi_v4},
    "fdb": {"v1": fdb_v1, "v4": fdb_v4}
}
with open("results/Qwen_Qwen3-4B-Instruct-2507/final_eval.json", "w") as f:
    json.dump(res, f, indent=2)

print("Done. Saved final_eval.json.")
