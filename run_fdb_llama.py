import json
import torch
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dual_state_interruptor import setup_model

def get_templates(version):
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

def run_fdb_experiment(model, tokenizer, dataset, config, architecture, version):
    print(f"Running FDB {architecture} ({version})...")
    total_lat = 0
    total_evals = 0
    successful = 0
    
    # We evaluate 100 FDB scenarios
    for idx, scenario in enumerate(dataset[:100]):
        base_context = config["system_turn_start"] + config["base_instructions"] + config["user_turn_start"] + "<INPUT>\n"
        
        # FDB scenarios already have a single large string chunk with the user barge in at the end
        full_text = base_context + scenario["transcript"] + f"\n</INPUT>\n[USER BARGE-IN]: {scenario['user_interruption']}"
        
        if architecture == "Independent":
            full_prompt = full_text + config["assessor_prompt"]
            inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            
            start = time.time()
            if version == "v4":
                with torch.no_grad():
                    out = model(input_ids=inputs.input_ids, use_cache=True)
                logits = out.logits[:, -1, :]
                stop_ids = [11614, 21974] if "llama" in model.name_or_path.lower() else [51769, 46637]
                cont_ids = [43509, 58498] if "llama" in model.name_or_path.lower() else [24194, 16511]
                # Llama 3 STOP/CONTINUE tokens: STOP: 11614, 21974 (Stop, STOP), CONTINUE: 43509, 58498 (Continue, CONTINUE)
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
            
            if should_stop == (scenario["expected"] == "STOP"):
                successful += 1

    print(f"Results for {architecture} ({version}):")
    print(f"Accuracy: {successful}/100")
    print(f"Avg Latency: {total_lat/total_evals:.2f} ms")
    return successful, total_lat/total_evals

def main():
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
    print("Loading model...")
    model, tokenizer = setup_model(model_name)
    
    with open("datasets/fdb_dataset.json", "r") as f:
        fdb = json.load(f)["data"]
        
    print("=== FDB BENCHMARKS ===")
    
    # We just test the Independent V1 vs V4 latency to prove the FDB Pre-fill penalty
    config = get_templates("v1")
    run_fdb_experiment(model, tokenizer, fdb, config, "Independent", "v1")
    
    config = get_templates("v4")
    run_fdb_experiment(model, tokenizer, fdb, config, "Independent", "v4")

if __name__ == "__main__":
    main()
