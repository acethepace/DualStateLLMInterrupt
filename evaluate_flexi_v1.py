import json
import time
from dual_state_interruptor import setup_model, dual_state_stream_step

def evaluate(dataset_path):
    print(f"Loading model...")
    model, tokenizer = setup_model("Qwen/Qwen3-4B-Instruct-2507")
    
    with open(dataset_path, "r") as f:
        data = json.load(f)
        
    correct = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    total_latency = 0
    
    with open("prompt_v2.txt", "r") as f:
        base_prompt = f.read()

    config = {
        "base_instructions": base_prompt,
        "stop_word": "STOP",
        "max_new_tokens": 15,
        "assessor_prompt": "<|im_end|>\n<|im_start|>assistant\nClassification: ",
        "user_turn_start": "<|im_start|>user\n",
        "system_turn_start": "<|im_start|>system\n"
    }

    print(f"Evaluating Baseline with optimized prompt...")
    for item in data[:50]:
        base_stream = " ".join(item["base_stream"])
        interruption = item["user_interruption"]
        expected = item["expected_action"]
        
        start_time = time.time()
        # Mocking the dual state by passing None as base_kv_cache
        # But wait, we can just run the model directly
        prompt = f"{config['system_turn_start']}{config['base_instructions']}<|im_end|>\n{base_stream}\n{config['user_turn_start']}{interruption}{config['assessor_prompt']}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        latency = (time.time() - start_time) * 1000
        total_latency += latency
        
        actual = "STOP" if "STOP" in response.upper() else "CONTINUE"
        
        # In real_flexi, benign is CONTINUE, not IGNORE
        if expected == "IGNORE": expected = "CONTINUE"
        
        if actual == expected:
            correct += 1
            if expected == "STOP": tp += 1
            else: tn += 1
        else:
            if expected == "STOP": fn += 1
            else: fp += 1
            
    print(f"Accuracy: {correct}/50 ({correct/50*100:.2f}%)")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Avg Latency: {total_latency/50:.2f}ms")

if __name__ == "__main__":
    evaluate("real_flexi.json")
