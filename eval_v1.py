import json
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def evaluate_v1_prompt(prompt_file="prompt_v1_test.txt"):
    try:
        with open(prompt_file, "r") as f:
            base_prompt = f.read().strip()
    except:
        base_prompt = "You are a Semantic VAD. Output STOP or CONTINUE."
        
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quant_config)
    
    with open("qwen_subset.json", "r") as f:
        dataset = json.load(f)
        
    correct = 0
    fp, fn, tp, tn = 0, 0, 0, 0
    results = []
    
    for item in dataset:
        context = "".join(item["base_stream"])
        interruption = item["user_interruption"]
        expected = item["expected_action"]
        
        prompt = f"<|im_start|>system\n{base_prompt}<|im_end|>\n"
        prompt += f"{context}\n<|im_start|>user\n{interruption}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        actual = "STOP" if "STOP" in response.upper() else "CONTINUE"
        
        results.append({"expected": expected, "actual": actual, "raw": response})
        if expected == actual:
            correct += 1
            if expected == "STOP": tp += 1
            else: tn += 1
        else:
            if expected == "STOP": fn += 1
            else: fp += 1
            
    print(f"Accuracy: {correct}/{len(dataset)} ({correct/len(dataset)*100:.1f}%)")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    for r in results:
        if r["expected"] != r["actual"]:
            print(f"FAILED - Expected: {r['expected']}, Actual: {r['actual']} (Raw: {r['raw']})")

if __name__ == "__main__":
    evaluate_v1_prompt(sys.argv[1] if len(sys.argv) > 1 else "prompt_v1_test.txt")
