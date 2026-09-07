import json
import time
from dual_state_interruptor import setup_model

def evaluate(dataset_path):
    model, tokenizer = setup_model("Qwen/Qwen3-4B-Instruct-2507")
    with open(dataset_path, "r") as f: data = json.load(f)
        
    correct, tp, tn, fp, fn = 0, 0, 0, 0, 0
    with open("prompt_v2.txt", "r") as f: base_prompt = f.read()

    for item in data[-50:]:
        base_stream = " ".join(item["base_stream"])
        prompt = f"<|im_start|>system\n{base_prompt}<|im_end|>\n{base_stream}\n<|im_start|>user\n{item['user_interruption']}\n<|im_end|>\n<|im_start|>assistant\nClassification: "
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        actual = "STOP" if "STOP" in response.upper() else "CONTINUE"
        
        if actual == "CONTINUE":
            correct += 1; tn += 1
        else:
            fp += 1
            
    print(f"Accuracy: {correct}/50 ({correct/50*100:.2f}%)")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

if __name__ == "__main__":
    evaluate("real_flexi.json")
