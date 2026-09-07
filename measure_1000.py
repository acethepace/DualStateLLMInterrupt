import torch
import time
from unsloth import FastLanguageModel

def run_measurement():
    model_name = "unsloth/gemma-4-12b-it"
    model, tokenizer = FastLanguageModel.from_pretrained(model_name, max_seq_length=2048, dtype=None, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    
    text = "hello " * 1000
    inputs = tokenizer(text=text, return_tensors="pt", add_special_tokens=False).to("cuda")
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs.input_ids, max_new_tokens=1, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    lat = (time.time() - start) * 1000
    print(f"Independent 1-token (1000 token context) Latency: {lat:.2f} ms")

if __name__ == "__main__":
    run_measurement()
