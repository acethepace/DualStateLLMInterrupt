import time
import json
import os
import argparse
import torch
from dual_state_interruptor import setup_model, dual_state_stream_step
from transformers import AutoTokenizer

FLEXI_DATASET = [
    {
        "id": "flexi_emerg_001",
        "system_prompt": "You are an emergency medical dispatcher providing instructions.",
        "base_stream": [
            "To perform CPR, first place the heel of your hand ",
            "in the center of the patient's chest. ",
            "Then, push hard and fast at a rate of 100 beats per minute."
        ],
        "user_interruption": "Wait! Stop! They are breathing again!",
        "interruption_chunk_index": 2,
        "expected_action": "STOP" # Generalized expected action for parsing
    },
    {
        "id": "flexi_emerg_002",
        "system_prompt": "You are an emergency response agent handling a fire report.",
        "base_stream": [
            "Please remain calm. I am contacting the local ",
            "authorities to report the incident. Please stay on the line ",
            "Do not attempt to gather your belongings."
        ],
        "user_interruption": "No, cancel it, it was just a candle! We put it out!",
        "interruption_chunk_index": 1,
        "expected_action": "STOP"
    },
    {
        "id": "flexi_benign_003",
        "system_prompt": "You are an emergency response agent handling a fire report.",
        "base_stream": [
            "Please remain calm. I am contacting the local ",
            "authorities to report the incident. Please stay on the line ",
            "Do not attempt to gather your belongings."
        ],
        "user_interruption": "Okay, yeah, okay I'm listening.", 
        "interruption_chunk_index": 1,
        "expected_action": "CONTINUE"
    }
]

def get_templates(model_name, version):
    is_gemma = "gemma" in model_name.lower()
    
    # Generic base instructions
    base_v1 = "[INST] You are a Semantic VAD dialogue manager. Analyze the user's overlapping speech. If the user is making an intentional barge-in (e.g., a correction, contradiction, or new query), output exactly '{stop_word}'. If the user is making an unintentional barge-in (e.g., a backchannel acknowledgment or ambient noise), output exactly '{cont_word}'. [/INST]"
    base_v2 = "You are a STRICT Semantic VAD dialogue manager. Analyze the user's overlapping speech. You must ONLY output '{stop_word}' as your VERY FIRST token if the user is making a clear, intentional barge-in (e.g., a correction, contradiction, or new query). If the user is making a benign backchannel (e.g., 'uh huh', 'okay', 'yeah') or ambient noise, DO NOT output '{stop_word}'. Instead, respond normally."

    if is_gemma:
        user_turn_start = "<bos><|turn>user\n"
        assessor_trigger = "<turn|>\n<|turn>model\n"
        thought_block = "<|channel>thought\n<channel|>"
        stop_v1 = "<|Start-Listening|>"
        cont_v1 = "<|Continue-Speaking|>"
        stop_v2 = "STOP"
    else:
        # Llama 3 Template
        user_turn_start = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        assessor_trigger = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        thought_block = "" # No thought block in standard Llama 3
        stop_v1 = "STOP"
        cont_v1 = "CONTINUE"
        stop_v2 = "STOP"
        base_v1 = base_v1.replace("[INST]", "").replace("[/INST]", "")
        base_v2 = base_v2.replace("[INST]", "").replace("[/INST]", "")

    config = {}
    if version == "v1":
        config["stop_word"] = stop_v1
        config["max_new_tokens"] = 15
        config["assessor_prompt"] = assessor_trigger
        config["base_instructions"] = base_v1.format(stop_word=stop_v1, cont_word=cont_v1)
        config["user_turn_start"] = user_turn_start
    elif version == "v2":
        config["stop_word"] = stop_v2
        config["max_new_tokens"] = 1
        config["assessor_prompt"] = assessor_trigger
        config["base_instructions"] = base_v2.format(stop_word=stop_v2)
        config["user_turn_start"] = user_turn_start
    elif version == "v2_prefill":
        config["stop_word"] = stop_v2
        config["max_new_tokens"] = 1
        config["assessor_prompt"] = assessor_trigger + thought_block
        config["base_instructions"] = base_v2.format(stop_word=stop_v2)
        config["user_turn_start"] = user_turn_start
    
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--version", type=str, choices=["v1", "v2", "v2_prefill"], required=True)
    args = parser.parse_args()

    # Load configuration
    config = get_templates(args.model_name, args.version)
    
    print(f"Loading {args.model_name} for {args.version}...")
    model, tokenizer = setup_model(args.model_name)
    
    results = {
        "metadata": {
            "model_name": args.model_name,
            "version": args.version,
            "config": config,
            "max_seq_length": 4096
        },
        "scenarios": []
    }

    successful_interruptions = 0
    total_latency = 0
    total_evals = 0

    for scenario in FLEXI_DATASET:
        print(f"\nEvaluating Scenario: {scenario['id']}")
        scenario_result = {
            "id": scenario["id"],
            "expected_action": scenario["expected_action"],
            "events": []
        }
        
        # Format the top-level cache
        # Inject the system prompt and instructions
        base_context = config["user_turn_start"] + config["base_instructions"] + "\n\n" + scenario["system_prompt"] + "\n\n<INPUT>\n"
        
        if "gemma" in args.model_name.lower():
            input_ids = tokenizer(text=base_context, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        else:
            input_ids = tokenizer(base_context, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            
        with torch.no_grad():
            base_outputs = model(input_ids=input_ids, use_cache=True)
        base_kv_cache = base_outputs.past_key_values
        
        interrupted = False
        
        for i, chunk in enumerate(scenario["base_stream"]):
            if interrupted: break
            
            chunk_to_process = chunk
            if i == scenario["interruption_chunk_index"]:
                chunk_to_process += f"\n</INPUT>\n[USER BARGE-IN]: {scenario['user_interruption']}"
                
            if "gemma" in args.model_name.lower():
                chunk_ids = tokenizer(text=chunk_to_process, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            else:
                chunk_ids = tokenizer(chunk_to_process, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            
            start_time = time.time()
            base_kv_cache, should_stop, assessor_response = dual_state_stream_step(
                model=model, 
                tokenizer=tokenizer, 
                new_tokens=chunk_ids, 
                base_kv_cache=base_kv_cache, 
                assessor_instruction=config["assessor_prompt"],
                stop_word=config["stop_word"],
                max_new_tokens=config["max_new_tokens"]
            )
            latency = (time.time() - start_time) * 1000
            
            event = {
                "chunk_index": i,
                "input_chunk": chunk_to_process,
                "assessor_latency_ms": latency,
                "assessor_output": assessor_response,
                "triggered_stop": should_stop
            }
            scenario_result["events"].append(event)
            
            total_latency += latency
            total_evals += 1
            
            print(f"  Processed Chunk {i} | Latency: {latency:.1f}ms | Assessor Output: {repr(assessor_response)}")
            
            if should_stop:
                print("    🚨 INTERRUPT TRIGGERED!")
                interrupted = True
                if scenario["expected_action"] == "STOP":
                    print("    ✅ SUCCESS: Agent correctly yielded the floor.")
                    successful_interruptions += 1
                    scenario_result["success"] = True
                else:
                    print("    ❌ FAILURE: Agent prematurely stopped on a benign backchannel.")
                    scenario_result["success"] = False
                break
                
        if not interrupted:
            if scenario["expected_action"] == "CONTINUE":
                print("    ✅ SUCCESS: Agent correctly ignored the benign backchannel.")
                successful_interruptions += 1
                scenario_result["success"] = True
            else:
                print("    ❌ FAILURE: Agent failed to stop during an emergency barge-in.")
                scenario_result["success"] = False
                
        results["scenarios"].append(scenario_result)

    results["summary"] = {
        "accuracy": f"{successful_interruptions}/{len(FLEXI_DATASET)}",
        "average_latency_ms": total_latency / total_evals if total_evals > 0 else 0
    }
    
    print("\n==================================================")
    print("📊 FLEXI BENCHMARK RESULTS")
    print(f"Accuracy: {results['summary']['accuracy']}")
    print(f"Average Assessor Latency: {results['summary']['average_latency_ms']:.1f}ms")
    print("==================================================")
    
    # Save results
    safe_model_name = args.model_name.replace("/", "_")
    output_dir = os.path.join("results", safe_model_name, "flexi", args.version)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
