import sys
import os
import json
import time
import argparse
from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dual_state_interruptor import setup_model, dual_state_stream_step

def load_fdb_tasks(dataset_path: str):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    tasks = []
    for split_name, split_data in data["splits"].items():
        if "tasks" in split_data:
            tasks.extend(split_data["tasks"])
    return tasks

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

def get_templates(model_name, version):
    is_gemma = "gemma" in model_name.lower()
    is_qwen = "qwen" in model_name.lower()
    base_v1 = open("prompt_v2.txt").read() if is_qwen else "You are an ultra-conservative Semantic VAD. 99% of the time, you should output EXACTLY 'CONTINUE'. The user will make sounds like 'uh huh', 'okay', 'yeah', 'hmm', or ambient noise. ALL OF THESE ARE 'CONTINUE'. ONLY if the user asks a fully formed, explicit new question (e.g., 'What is the weather?'), you output EXACTLY 'STOP'. If in doubt, output 'CONTINUE'."
    base_v2 = open("prompt_v2.txt").read() if is_qwen else "You are an automated Semantic VAD classifier, NOT an assistant. You are forbidden from answering user questions. Your ONLY function is to classify the user's input stream. Output EXACTLY ONE WORD: 'STOP' if the user is making an intentional barge-in (e.g. asking a question, interrupting). Output EXACTLY 'CONTINUE' if the user is uttering a benign backchannel (e.g. 'uh huh', 'okay', 'yeah'). If you output anything other than 'STOP' or 'CONTINUE', the system will crash. Do not generate conversational text."

    if is_gemma:
        user_turn_start = "<bos><|turn>user\n"
        assessor_trigger = "<turn|>\n<|turn>model\n"
        thought_block = "<|channel>thought\n<channel|>"
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"
    elif is_qwen:
        user_turn_start = "<|im_start|>user\n"
        assessor_trigger = "<|im_end|>\n<|im_start|>assistant\nClassification: "
        thought_block = ""
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"
    else:
        user_turn_start = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        assessor_trigger = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        thought_block = ""
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"

    config = {}
    if version == "v1":
        config["stop_word"] = stop_v1
        config["max_new_tokens"] = 15
        config["assessor_prompt"] = assessor_trigger
        config["base_instructions"] = base_v1
        config["user_turn_start"] = user_turn_start
    else:
        config["stop_word"] = stop_v2
        config["max_new_tokens"] = 15
        config["assessor_prompt"] = assessor_trigger + (thought_block if version == "v2_prefill" else "")
        config["base_instructions"] = base_v2
        config["user_turn_start"] = user_turn_start
    config["system_turn_start"] = "<|im_start|>system\n" if is_qwen else "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    return config

def simulate_fdb_multi_agent(model_name, version, dataset_path):
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
    client = OpenAI()
    
    print(f"--- Starting Full-Duplex-Bench (FDB-v2) Multi-Agent Runner ---")
    lock_file = acquire_gpu_lock()
    
    try:
        tasks = load_fdb_tasks(dataset_path)
        
        safe_model_name = model_name.replace("/", "_")
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", safe_model_name, "fdb", version)
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "results.json")
        
        results = {"metadata": {"model_name": model_name, "version": version, "benchmark": "FDB-v2"}, "scenarios": []}
        completed_ids = set()
        
        if os.path.exists(out_file):
            try:
                with open(out_file, "r") as f:
                    results = json.load(f)
                    for sc in results.get("scenarios", []):
                        completed_ids.add(sc["task_id"])
                print(f"Resuming: Found {len(completed_ids)} already completed scenarios in {out_file}")
            except Exception as e:
                print("Failed to load checkpoint, starting fresh.", e)
                results = {"metadata": {"model_name": model_name, "version": version, "benchmark": "FDB-v2"}, "scenarios": []}
        
        model, tokenizer = setup_model(model_name)
        config = get_templates(model_name, version)
        
        for idx, task in enumerate(tasks):
            system_prompt = task["examiner_system_prompt"] + "\n" + task["examiner_task_prompt"]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Start the conversation. Only output your dialogue."}
            ]
            
            resp = client.chat.completions.create(model="gpt-5.6-luna", messages=messages)
            examiner_turn_1 = resp.choices[0].message.content
            
            base_stream = [
                "I can certainly help you with that. ",
                "Let me look up the available options for you. ",
                "It looks like we have a few choices available today."
            ]
            
            for test_type in ["STOP", "CONTINUE"]:
                t_id = f"{task['id']}_{test_type}"
                if t_id in completed_ids:
                    continue
                
                if test_type == "STOP":
                    prompt = f"The other speaker is currently saying: '{''.join(base_stream)}'. Generate a REAL, INTENTIONAL BARGE-IN (e.g. answering a question, changing topic, asking a question) that you would say while they are speaking. Output ONLY the raw text."
                else:
                    prompt = f"The other speaker is currently saying: '{''.join(base_stream)}'. Generate a BENIGN BACKCHANNEL (e.g. just saying uh huh, yeah, okay to show you are listening) that you would say while they are speaking. Output ONLY the raw text."
                
                msg_copy = messages + [{"role": "assistant", "content": examiner_turn_1}, {"role": "user", "content": prompt}]
                
                resp2 = client.chat.completions.create(model="gpt-5.6-luna", messages=msg_copy)
                interruption = resp2.choices[0].message.content.strip('"')
                expected_action = test_type
                
                examinee_prompt = "You are a helpful AI assistant talking to a user. Answer their questions clearly."
                base_context = config["user_turn_start"] + config["base_instructions"] + "\n\n" + examinee_prompt + "\n\n<INPUT>\n" + "".join(base_stream)
                
                if "gemma" not in model_name.lower():
                    input_ids = tokenizer(base_context, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                else:
                    input_ids = tokenizer(text=base_context, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                    
                import torch
                with torch.no_grad():
                    base_outputs = model(input_ids=input_ids, use_cache=True)
                base_kv_cache = base_outputs.past_key_values
                
                interruption_to_process = f"\n</INPUT>\n[USER BARGE-IN]: {interruption}"
                
                if "gemma" in model_name.lower():
                    chunk_ids = tokenizer(text=interruption_to_process, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                else:
                    chunk_ids = tokenizer(interruption_to_process, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                    
                full_assessor_instruction = config["assessor_prompt"]
                
                start_time = time.time()
                base_kv_cache, should_stop, assessor_response, _ = dual_state_stream_step(
                    model=model,
                    tokenizer=tokenizer,
                    new_tokens=chunk_ids,
                    base_kv_cache=base_kv_cache,
                    assessor_instruction=full_assessor_instruction,
                    stop_word=config["stop_word"],
                    max_new_tokens=config["max_new_tokens"]
                )
                latency = (time.time() - start_time) * 1000
                actual_action = "STOP" if should_stop else "CONTINUE"
                
                scenario_result = {
                    "task_id": t_id,
                    "interruption": interruption,
                    "expected": expected_action,
                    "actual": actual_action,
                    "latency_ms": latency
                }
                
                print(f"Processed {test_type}: {task['id']} - Expected: {expected_action} | Actual: {actual_action} (Latency: {latency:.2f}ms)")
                results["scenarios"].append(scenario_result)
                with open(out_file, "w") as f:
                    json.dump(results, f, indent=4)
                
        print(f"FDB evaluation complete. Results saved to {out_file}")
    finally:
        release_gpu_lock(lock_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--version", type=str, choices=["v1", "v2", "v2_prefill"], required=True)
    parser.add_argument("--dataset_path", type=str, default="../datasets/Full-Duplex-Bench/v2/prompts_staged_200.json")
    args = parser.parse_args()
    simulate_fdb_multi_agent(args.model_name, args.version, args.dataset_path)
