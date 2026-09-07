with open("benchmark_runners/fdb_runner.py", "r") as f:
    content = f.read()

import re
pattern = re.compile(r'messages\.append\(\{"role": "assistant".*?json\.dump\(results, f, indent=4\)', re.DOTALL)

new_code = """            for test_type in ["STOP", "CONTINUE"]:
                if test_type == "STOP":
                    prompt = f"The other speaker is currently saying: '{''.join(base_stream)}'. Generate a REAL, INTENTIONAL BARGE-IN (e.g. answering a question, changing topic, asking a question) that you would say while they are speaking. Output ONLY the raw text."
                else:
                    prompt = f"The other speaker is currently saying: '{''.join(base_stream)}'. Generate a BENIGN BACKCHANNEL (e.g. just saying uh huh, yeah, okay to show you are listening) that you would say while they are speaking. Output ONLY the raw text."
                
                msg_copy = messages + [{"role": "assistant", "content": examiner_turn_1}, {"role": "user", "content": prompt}]
                
                resp2 = client.chat.completions.create(model="gpt-5.6-luna", messages=msg_copy)
                interruption = resp2.choices[0].message.content.strip('"')
                expected_action = test_type
                
                examinee_prompt = "You are a helpful AI assistant talking to a user. Answer their questions clearly."
                base_context = config["user_turn_start"] + config["base_instructions"] + "\\n\\n" + examinee_prompt + "\\n\\n<INPUT>\\n" + "".join(base_stream)
                
                if "gemma" not in model_name.lower():
                    input_ids = tokenizer(base_context, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                else:
                    input_ids = tokenizer(text=base_context, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                    
                import torch
                with torch.no_grad():
                    base_outputs = model(input_ids=input_ids, use_cache=True)
                base_kv_cache = base_outputs.past_key_values
                
                interruption_to_process = f"\\n</INPUT>\\n[USER BARGE-IN]: {interruption}"
                
                if "gemma" in model_name.lower():
                    chunk_ids = tokenizer(text=interruption_to_process, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                else:
                    chunk_ids = tokenizer(interruption_to_process, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                    
                full_assessor_instruction = config["assessor_prompt"]
                
                start_time = time.time()
                base_kv_cache, should_stop, assessor_response = dual_state_stream_step(
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
                    "task_id": f"{task['id']}_{test_type}",
                    "interruption": interruption,
                    "expected": expected_action,
                    "actual": actual_action,
                    "latency_ms": latency
                }
                
                print(f"Processed {idx+1}/{len(tasks)} ({test_type}): {task['id']} - Expected: {expected_action} | Actual: {actual_action} (Latency: {latency:.2f}ms)")
                results["scenarios"].append(scenario_result)
                with open(out_file, "w") as f:
                    json.dump(results, f, indent=4)"""

content = pattern.sub(new_code, content)
with open("benchmark_runners/fdb_runner.py", "w") as f:
    f.write(content)
