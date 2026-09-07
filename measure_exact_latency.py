import torch
import time
import json
from unsloth import FastLanguageModel

def run_measurement():
    model_name = "unsloth/gemma-4-12b-it"
    max_seq_length = 4096

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print("Loading a scenario from FLEXI...")
    with open("mini_flexi.json", "r") as f:
        scenarios = json.load(f)
    
    # Take a representative scenario
    scenario = scenarios[0]
    
    base_instructions = """You are a helpful AI assistant engaged in a voice conversation. Listen carefully to the user.
If the user interrupts to change the subject, stop you, or provide new constraints, say STOP.
If the user is just offering backchannel agreement (like 'uh-huh', 'okay', 'yes'), say CONTINUE."""
    
    user_turn_start = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    assessor_trigger = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nClassification: "
    
    # 1. Prepare Independent Context
    base_context = user_turn_start + base_instructions + "\n\n" + scenario["system_prompt"] + "\n\n<INPUT>\n"
    current_stream = base_context

    print("Starting exact measurement loop...")
    independent_15_latencies = []
    independent_1_latencies = []
    
    for i, chunk in enumerate(scenario["base_stream"][:3]): # Measure 3 chunks
        current_stream += chunk
        
        # We simulate what an independent decider does: it tokenizes everything from scratch
        full_text = current_stream + assessor_trigger
        
        inputs = tokenizer(text=full_text, return_tensors="pt", add_special_tokens=False).to("cuda")
        
        # A. Independent (15 tokens)
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs.input_ids, max_new_tokens=15, use_cache=True, pad_token_id=tokenizer.eos_token_id)
        ind_15_lat = (time.time() - start) * 1000
        independent_15_latencies.append(ind_15_lat)
        
        # B. Independent (1 token / v2 prefill equivalent)
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs.input_ids, max_new_tokens=1, use_cache=True, pad_token_id=tokenizer.eos_token_id)
        ind_1_lat = (time.time() - start) * 1000
        independent_1_latencies.append(ind_1_lat)
        
    print(f"Independent 15-tokens Avg: {sum(independent_15_latencies)/len(independent_15_latencies):.2f} ms")
    print(f"Independent 1-token Avg: {sum(independent_1_latencies)/len(independent_1_latencies):.2f} ms")

if __name__ == "__main__":
    run_measurement()
