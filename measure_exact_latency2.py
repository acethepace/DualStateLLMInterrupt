import torch
import time
import json
from unsloth import FastLanguageModel

def run_measurement():
    model_name = "unsloth/gemma-4-12b-it"
    max_seq_length = 4096

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    with open("mini_flexi.json", "r") as f:
        scenarios = json.load(f)
    
    scenario = scenarios[0]
    base_instructions = "You are a helpful AI assistant..."
    
    user_turn_start = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    assessor_trigger = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nClassification: "
    
    base_context = user_turn_start + base_instructions + "\n\n" + scenario["system_prompt"] + "\n\n<INPUT>\n"
    current_stream = base_context

    # Get Base KV cache
    base_inputs = tokenizer(text=base_context, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        base_outputs = model(input_ids=base_inputs.input_ids, use_cache=True)
    base_kv = base_outputs.past_key_values

    dual_1_latencies = []
    
    for i, chunk in enumerate(scenario["base_stream"][:3]):
        # Update Base KV cache with chunk
        chunk_inputs = tokenizer(text=chunk, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            base_outputs = model(input_ids=chunk_inputs.input_ids, past_key_values=base_kv, use_cache=True)
        base_kv = base_outputs.past_key_values
        
        # Fork to Assessor
        assessor_inputs = tokenizer(text=assessor_trigger, return_tensors="pt", add_special_tokens=False).to("cuda")
        
        start = time.time()
        with torch.no_grad():
            out = model.generate(input_ids=assessor_inputs.input_ids, past_key_values=base_kv, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        dual_1_lat = (time.time() - start) * 1000
        dual_1_latencies.append(dual_1_lat)
        
    print(f"Dual State 1-token Avg: {sum(dual_1_latencies)/len(dual_1_latencies):.2f} ms")

if __name__ == "__main__":
    run_measurement()
