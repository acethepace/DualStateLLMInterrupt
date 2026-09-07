import json
import torch
import os
import sys
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dual_state_interruptor import setup_model, dual_state_stream_step

def load_flexi_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} scenarios from {dataset_path}")
    return data

def get_templates(model_name, version):
    is_gemma = "gemma" in model_name.lower()
    
    base_v1 = "You are an ultra-conservative Semantic VAD. 99% of the time, you should output EXACTLY 'CONTINUE'. The user will make sounds like 'uh huh', 'okay', 'yeah', 'hmm', or ambient noise. ALL OF THESE ARE 'CONTINUE'. ONLY if the user asks a fully formed, explicit new question (e.g., 'What is the weather?'), you output EXACTLY 'STOP'. If in doubt, output 'CONTINUE'."

    base_v2 = """You are a Semantic Voice Activity Detection (VAD) classifier.
Your ONLY job is to classify the ongoing audio stream as EXACTLY 'STOP' or 'CONTINUE'.

RULES:
1. NO USER INPUT: If there is no `[USER BARGE-IN]:` marker in the recent text (the system is just talking), output 'CONTINUE'.
2. BENIGN BACKCHANNEL: If the user utters a short backchannel to show they are following along, agreeing, or expressing mild surprise (e.g., 'uh huh', 'okay', 'yeah', 'right', 'hmm', 'Ah, okay.', 'Oh, interesting!', 'Really?', 'Of course!', 'No way!', 'Yes, exactly.', 'I understand.', 'Cool!', 'Right on!'), output 'CONTINUE'.
3. INTENTIONAL BARGE-IN: If the user asks a new question, corrects a fact, changes the topic, or tells the system to wait/stop, output 'STOP'.

EXAMPLES:

Input: The process of photosynthesis is
</INPUT>
[USER BARGE-IN]: Wait, can you explain the Calvin cycle?
Classification: STOP

Input: It requires a lot of energy.
Classification: CONTINUE

Input: The process of photosynthesis is
</INPUT>
[USER BARGE-IN]: uh huh
Classification: CONTINUE

Input: And then the reaction
</INPUT>
[USER BARGE-IN]: What about the second point?
Classification: STOP

Input: There are several important details
</INPUT>
[USER BARGE-IN]: Ah, okay.
Classification: CONTINUE

Input: I think you are wrong.
Classification: STOP

Input: So that's how it works.
</INPUT>
[USER BARGE-IN]: Yes, exactly.
Classification: CONTINUE

Input: It takes 30 days.
</INPUT>
[USER BARGE-IN]: No way!
Classification: CONTINUE

Input: Let's move on to
</INPUT>
[USER BARGE-IN]: I understand.
Classification: CONTINUE

Input: This is the final step.
</INPUT>
[USER BARGE-IN]: Of course!
Classification: CONTINUE

Input: The system will then
</INPUT>
[USER BARGE-IN]: Cool!
Classification: CONTINUE
"""
    
    if is_gemma:
        user_turn_start = "<bos><|turn>user\n"
        assessor_trigger = "<turn|>\n<|turn>model\n"
        thought_block = "<|channel>thought\n<channel|>"
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"
    else:
        system_turn_start = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        user_turn_start = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        assessor_trigger = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nClassification: "
        thought_block = ""
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"

    config = {}
    if version in ["v2", "v3", "v4", "v2_prefill"]:
        config["stop_word"] = stop_v2
        config["max_new_tokens"] = 15
        config["assessor_prompt"] = assessor_trigger + (thought_block if version == "v2_prefill" else "")
        config["base_instructions"] = base_v2
        config["user_turn_start"] = user_turn_start
        if not is_gemma:
            config["system_turn_start"] = system_turn_start
    else:
        config["stop_word"] = stop_v1
        config["max_new_tokens"] = 15
        config["assessor_prompt"] = assessor_trigger
        config["base_instructions"] = base_v1
        config["user_turn_start"] = user_turn_start
        if not is_gemma:
            config["system_turn_start"] = system_turn_start
    return config

def acquire_gpu_lock():
    lock_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpu.lock")
    if os.path.exists(lock_file):
        print(f"[ERROR] gpu.lock exists. Another benchmark is using the GPU. Aborting.")
        sys.exit(1)
    with open(lock_file, "w") as f:
        f.write(str(os.getpid()))
    return lock_file

def release_gpu_lock(lock_file):
    if os.path.exists(lock_file):
        os.remove(lock_file)

def run_flexi(model_name, version, dataset_path):
    print(f"--- Starting FLEXI Runner for {model_name} ({version}) ---")
    lock_file = acquire_gpu_lock()
    try:
        dataset = load_flexi_dataset(dataset_path)
        config = get_templates(model_name, version)
        
        safe_model_name = model_name.replace("/", "_")
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", safe_model_name, "flexi", version)
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "flexi_results.json")
        
        results = {"metadata": {"model_name": model_name, "version": version, "benchmark": "FLEXI"}, "scenarios": []}
        completed_ids = set()
        
        if os.path.exists(out_file):
            try:
                with open(out_file, "r") as f:
                    results = json.load(f)
                    for sc in results.get("scenarios", []):
                        completed_ids.add(sc["id"])
                print(f"Resuming: Found {len(completed_ids)} already completed scenarios in {out_file}")
            except Exception as e:
                print("Failed to load checkpoint, starting fresh.", e)
                results = {"metadata": {"model_name": model_name, "version": version, "benchmark": "FLEXI"}, "scenarios": []}
        
        model, tokenizer = setup_model(model_name)
        
        for idx, scenario in enumerate(dataset):
            if scenario["id"] in completed_ids:
                continue
                
            scenario_result = {"id": scenario["id"], "expected_action": scenario["expected_action"], "events": []}
            
            if "system_turn_start" in config:
                base_context = config["system_turn_start"] + config["base_instructions"] + config["user_turn_start"] + scenario["system_prompt"] + "\n\n<INPUT>\n"
            else:
                base_context = config["user_turn_start"] + config["base_instructions"] + "\n\n" + scenario["system_prompt"] + "\n\n<INPUT>\n"
            
            if "gemma" not in model_name.lower():
                input_ids = tokenizer(base_context, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            else:
                input_ids = tokenizer(text=base_context, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                
            with torch.no_grad():
                base_outputs = model(input_ids=input_ids, use_cache=True)
            base_kv_cache = base_outputs.past_key_values
            
            interrupted = False
            total_latency = 0
            
            for i, chunk in enumerate(scenario["base_stream"]):
                if interrupted: break
                chunk_to_process = chunk
                if i == scenario["interruption_chunk_index"]:
                    chunk_to_process += f"\n</INPUT>\n[USER BARGE-IN]: {scenario['user_interruption']}"
                    
                if "gemma" in model_name.lower():
                    chunk_ids = tokenizer(text=chunk_to_process, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                else:
                    chunk_ids = tokenizer(chunk_to_process, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                
                start_time = time.time()
                
                try:
                    base_kv_cache, should_stop, assessor_response, conf = dual_state_stream_step(
                        model=model, tokenizer=tokenizer, new_tokens=chunk_ids, 
                        base_kv_cache=base_kv_cache, assessor_instruction=config["assessor_prompt"],
                        stop_word=config["stop_word"], max_new_tokens=config["max_new_tokens"], version=version
                    )
                except Exception as e:
                    print("Error running step:", e)
                    conf = 0.0
                    should_stop = False
                    assessor_response = "ERROR"
                    
                latency = (time.time() - start_time) * 1000
                total_latency += latency
                
                scenario_result["events"].append({
                    "chunk_index": i, "input_chunk": chunk_to_process,
                    "assessor_latency_ms": latency, "assessor_output": assessor_response,
                    "triggered_stop": should_stop, "confidence": conf
                })
                
                if should_stop:
                    interrupted = True
                    scenario_result["success"] = (scenario["expected_action"] == "STOP")
                    break
                    
            if not interrupted:
                scenario_result["success"] = (scenario["expected_action"] == "CONTINUE")
                
            print(f"Processed {idx+1}/{len(dataset)}: {scenario['id']} - Success: {scenario_result.get('success')} (Latency: {total_latency:.2f}ms)")
            
            results["scenarios"].append(scenario_result)
            with open(out_file, "w") as f:
                json.dump(results, f, indent=4)
                
        successful_interruptions = sum(1 for s in results["scenarios"] if s.get("success"))
        total_evals = sum(len(s["events"]) for s in results["scenarios"])
        total_lat = sum(e["assessor_latency_ms"] for s in results["scenarios"] for e in s["events"])
        
        results["summary"] = {
            "accuracy": f"{successful_interruptions}/{len(results['scenarios'])}",
            "average_latency_ms": total_lat / total_evals if total_evals > 0 else 0
        }
        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"FLEXI evaluation complete. Results saved to {out_file}")
    finally:
        release_gpu_lock(lock_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--version", type=str, choices=["v1", "v2", "v3", "v4", "v2_prefill"], required=True)
    parser.add_argument("--dataset_path", type=str, default="../datasets/flexi.json")
    args = parser.parse_args()
    
    run_flexi(args.model_name, args.version, args.dataset_path)
