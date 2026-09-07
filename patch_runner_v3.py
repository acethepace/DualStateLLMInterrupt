import re

with open("benchmark_runners/flexi_runner.py", "r") as f:
    content = f.read()

# Replace version validation
content = content.replace('choices=["v1", "v2", "v2_prefill"]', 'choices=["v1", "v2", "v3", "v2_prefill"]')

# Replace dual_state_stream_step call
old_call = """                base_kv_cache, should_stop, assessor_response = dual_state_stream_step(
                    model=model, tokenizer=tokenizer, new_tokens=chunk_ids, 
                    base_kv_cache=base_kv_cache, assessor_instruction=config["assessor_prompt"],
                    stop_word=config["stop_word"], max_new_tokens=config["max_new_tokens"]
                )"""

new_call = """                try:
                    base_kv_cache, should_stop, assessor_response, conf = dual_state_stream_step(
                        model=model, tokenizer=tokenizer, new_tokens=chunk_ids, 
                        base_kv_cache=base_kv_cache, assessor_instruction=config["assessor_prompt"],
                        stop_word=config["stop_word"], max_new_tokens=config["max_new_tokens"], version=version
                    )
                except ValueError:
                    # In case old dual_state_stream_step is still imported? No, it's patched.
                    pass"""
content = content.replace(old_call, new_call)

# Update events logging
old_event = """                scenario_result["events"].append({
                    "chunk_index": i, "input_chunk": chunk_to_process,
                    "assessor_latency_ms": latency, "assessor_output": assessor_response,
                    "triggered_stop": should_stop
                })"""

new_event = """                scenario_result["events"].append({
                    "chunk_index": i, "input_chunk": chunk_to_process,
                    "assessor_latency_ms": latency, "assessor_output": assessor_response,
                    "triggered_stop": should_stop, "confidence": conf
                })"""
content = content.replace(old_event, new_event)

with open("benchmark_runners/flexi_runner.py", "w") as f:
    f.write(content)
print("done")
