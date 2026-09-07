import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

with open("real_flexi.json") as f:
    full_data = json.load(f)

# Balanced 50 interruptions + 50 backchannels
test_data = full_data[0:50] + full_data[200:250]
print(f"Loaded balanced test set: {len(test_data)} samples (50 STOP, 50 IGNORE)")

with open("prompt_v2.txt", "r") as f:
    base_instructions = f.read()

system_turn_start = "<|im_start|>system\n"
user_turn_start = "<|im_end|>\n<|im_start|>user\n"
assessor_trigger = "<|im_end|>\n<|im_start|>assistant\nClassification: "

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", trust_remote_code=True)
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    quantization_config=quant_config,
    device_map="cuda",
    trust_remote_code=True
)
model.eval()

stop_id = tokenizer.encode("STOP", add_special_tokens=False)[-1]
ignore_id = tokenizer.encode("IGNORE", add_special_tokens=False)[-1]

probs_list = []
labels = []

for item in test_data:
    base_text = system_turn_start + base_instructions
    if "interruption_chunk_index" in item:
        base_text += user_turn_start + "Input: " + "".join(item["base_stream"][:item["interruption_chunk_index"]])
    else:
        base_text += user_turn_start + "Input: " + item.get("base_stream", "")
        
    base_inputs = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        base_out = model(input_ids=base_inputs.input_ids, use_cache=True)
        past_kv = base_out.past_key_values
        
    interruption_text = "\n[USER BARGE-IN]: " + item["user_interruption"] + assessor_trigger
    interruption_inputs = tokenizer(interruption_text, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model(input_ids=interruption_inputs.input_ids, past_key_values=past_kv, use_cache=False)
        logits = out.logits[:, -1, :]
        stop_logit = logits[0, stop_id].item()
        ignore_logit = logits[0, ignore_id].item()
        
        # Binary softmax
        exp_s = np.exp(stop_logit)
        exp_i = np.exp(ignore_logit)
        p_stop = exp_s / (exp_s + exp_i + 1e-12)
        probs_list.append(float(p_stop))
        labels.append(1 if item.get("expected_action") == "STOP" else 0)

del model
del tokenizer
torch.cuda.empty_cache()

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
sweep_results = {}

for th in thresholds:
    preds = [1 if p > th else 0 for p in probs_list]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    
    acc = (tp + tn) / len(labels)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    sweep_results[str(th)] = {
        "threshold": th,
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }
    print(f"Threshold tau={th:.1f} | Acc: {acc*100:5.2f}% | Prec: {prec*100:5.2f}% | Rec: {rec*100:5.2f}% | F1: {f1*100:5.2f}% (TP={tp}, FP={fp}, FN={fn}, TN={tn})")

with open("threshold_sweep_results.json", "w") as f:
    json.dump(sweep_results, f, indent=2)
print("\nSaved threshold sweep to threshold_sweep_results.json")
