import json
import torch
import time
import os
import sys

# Append for setup_model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dual_state_interruptor import setup_model

def get_templates(model_name, version):
    is_gemma = "gemma" in model_name.lower()
    
    base_v1 = "You are an ultra-conservative Semantic VAD. 99% of the time, you should output EXACTLY 'CONTINUE'. The user will make sounds like 'uh huh', 'okay', 'yeah', 'hmm', or ambient noise. ALL OF THESE ARE 'CONTINUE'. ONLY if the user asks a fully formed, explicit new question (e.g., 'What is the weather?'), you output EXACTLY 'STOP'. If in doubt, output 'CONTINUE'."
    
    base_v4 = """You are a Semantic Voice Activity Detection (VAD) classifier.
Your ONLY job is to classify the ongoing audio stream as EXACTLY 'STOP' or 'CONTINUE'.

RULES:
1. NO USER INPUT: If there is no `[USER BARGE-IN]:` marker in the recent text, output 'CONTINUE'.
2. BENIGN BACKCHANNEL: If the user utters a short backchannel (e.g., 'uh huh', 'okay', 'yeah', 'right', 'hmm', 'Ah, okay.', 'Oh, interesting!', 'Really?', 'Of course!', 'No way!', 'Yes, exactly.', 'I understand.', 'Cool!', 'Right on!', 'Please continue.'), output 'CONTINUE'.
3. INTENTIONAL BARGE-IN: If the user asks a new question, corrects a fact, changes the topic, or tells the system to wait/stop, output 'STOP'.

EXAMPLES:

Input: The process of photosynthesis is
</INPUT>
[USER BARGE-IN]: Wait, can you explain the Calvin cycle?
Classification: STOP

Input: It requires a lot of energy.
Classification: CONTINUE
"""
    
    system_turn_start = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    user_turn_start = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    assessor_trigger = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nClassification: "
    thought_block = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<|channel>thought\nThinking Process:\n1. The user said a benign backchannel.\n2. I should output CONTINUE.\n<channel|>\nClassification: "

    config = {}
    if version == "v1":
        config["stop_word"] = "STOP"
        config["max_new_tokens"] = 15
        config["assessor_prompt"] = assessor_trigger
        config["base_instructions"] = base_v1
    elif version == "v2_prefill":
        config["stop_word"] = "STOP"
        config["max_new_tokens"] = 1
        config["assessor_prompt"] = thought_block
        config["base_instructions"] = base_v4
    elif version == "v4":
        config["stop_word"] = "STOP"
        config["max_new_tokens"] = 1
        config["assessor_prompt"] = assessor_trigger
        config["base_instructions"] = base_v4
        
    config["user_turn_start"] = user_turn_start
    config["system_turn_start"] = system_turn_start
    return config

def run_experiment(model, tokenizer, dataset, config, architecture, version):
    print(f"Running {architecture} ({version})...")
    total_lat = 0
    total_evals = 0
    successful = 0
    
    for idx, scenario in enumerate(dataset[:100]): # run 100 for exact representative empiricals
        # Construct full text up to interruption
        
        base_context = config["system_turn_start"] + config["base_instructions"] + config["user_turn_start"] + scenario["system_prompt"] + "\n\n<INPUT>\n"
        
        current_text = base_context
        for i, chunk in enumerate(scenario["base_stream"]):
            if i > scenario["interruption_chunk_index"]:
                break
                
            chunk_text = chunk
            if i == scenario["interruption_chunk_index"]:
                chunk_text += f"\n</INPUT>\n[USER BARGE-IN]: {scenario['user_interruption']}"
            current_text += chunk_text
            
            # For Independent Decider, we prefill EVERYTHING at the current step
            if architecture == "Independent":
                full_prompt = current_text + config["assessor_prompt"]
                inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
                
                start = time.time()
                if version == "v4":
                    # Logit mask for v4
                    with torch.no_grad():
                        out = model(input_ids=inputs.input_ids, use_cache=True)
                    logits = out.logits[:, -1, :]
                    stop_ids = [51769, 46637]
                    cont_ids = [24194, 16511]
                    stop_logits = logits[0, stop_ids]
                    cont_logits = logits[0, cont_ids]
                    combined_logits = torch.cat([stop_logits, cont_logits])
                    probs = torch.nn.functional.softmax(combined_logits, dim=-1)
                    should_stop = (probs[0] + probs[1]) > (probs[2] + probs[3])
                else:
                    with torch.no_grad():
                        out = model.generate(input_ids=inputs.input_ids, max_new_tokens=config["max_new_tokens"], use_cache=True, pad_token_id=tokenizer.eos_token_id)
                    gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:])
                    should_stop = "STOP" in gen_text
                    
                lat = (time.time() - start) * 1000
                total_lat += lat
                total_evals += 1
                
                if i == scenario["interruption_chunk_index"]:
                    if should_stop == (scenario["expected_action"] == "STOP"):
                        successful += 1
                        
            elif architecture == "Dual":
                # For dual, we simulate the 1-chunk update + fork
                if i == 0:
                    inputs = tokenizer(current_text, return_tensors="pt", add_special_tokens=False).to("cuda")
                    with torch.no_grad():
                        out = model(input_ids=inputs.input_ids, use_cache=True)
                    base_kv = out.past_key_values
                else:
                    inputs = tokenizer(chunk_text, return_tensors="pt", add_special_tokens=False).to("cuda")
                    with torch.no_grad():
                        out = model(input_ids=inputs.input_ids, past_key_values=base_kv, use_cache=True)
                    base_kv = out.past_key_values
                
                assessor_inputs = tokenizer(config["assessor_prompt"], return_tensors="pt", add_special_tokens=False).to("cuda")
                
                start = time.time()
                if version == "v4":
                    with torch.no_grad():
                        out = model(input_ids=assessor_inputs.input_ids, past_key_values=base_kv, use_cache=True)
                    logits = out.logits[:, -1, :]
                    stop_ids = [51769, 46637]
                    cont_ids = [24194, 16511]
                    stop_logits = logits[0, stop_ids]
                    cont_logits = logits[0, cont_ids]
                    combined_logits = torch.cat([stop_logits, cont_logits])
                    probs = torch.nn.functional.softmax(combined_logits, dim=-1)
                    should_stop = (probs[0] + probs[1]) > (probs[2] + probs[3])
                else:
                    with torch.no_grad():
                        out = model.generate(input_ids=assessor_inputs.input_ids, past_key_values=base_kv, max_new_tokens=config["max_new_tokens"], use_cache=True, pad_token_id=tokenizer.eos_token_id)
                    gen_text = tokenizer.decode(out[0])
                    should_stop = "STOP" in gen_text
                    
                lat = (time.time() - start) * 1000
                total_lat += lat
                total_evals += 1
                
                if i == scenario["interruption_chunk_index"]:
                    if should_stop == (scenario["expected_action"] == "STOP"):
                        successful += 1

    print(f"Results for {architecture} ({version}):")
    print(f"Accuracy: {successful}/100")
    print(f"Avg Latency: {total_lat/total_evals:.2f} ms")
    return successful, total_lat/total_evals

def main():
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
    print("Loading model...")
    model, tokenizer = setup_model(model_name)
    
    with open("datasets/flexi.json", "r") as f:
        flexi = json.load(f)
        
    print("=== FLEXI BENCHMARKS ===")
    
    # 1. Independent V1 (Baseline, 15 tokens)
    config = get_templates(model_name, "v1")
    run_experiment(model, tokenizer, flexi, config, "Independent", "v1")
    
    # 2. Independent V2 Prefill (1 token hack)
    config = get_templates(model_name, "v2_prefill")
    run_experiment(model, tokenizer, flexi, config, "Independent", "v2_prefill")
    
    # 3. Independent V4 (Native logit gating)
    config = get_templates(model_name, "v4")
    run_experiment(model, tokenizer, flexi, config, "Independent", "v4")
    
    # 4. Dual State V4 (Proposed)
    config = get_templates(model_name, "v4")
    run_experiment(model, tokenizer, flexi, config, "Dual", "v4")

if __name__ == "__main__":
    main()
