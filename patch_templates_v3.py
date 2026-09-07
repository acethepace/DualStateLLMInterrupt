import re
with open("benchmark_runners/flexi_runner.py", "r") as f: content = f.read()
old = '    if version == "v1":'
new = '    if version in ["v2", "v3", "v2_prefill"]:\n        config["stop_word"] = stop_v2\n        config["max_new_tokens"] = 1\n        config["assessor_prompt"] = assessor_trigger + (thought_block if version == "v2_prefill" else "")\n        config["base_instructions"] = base_v2\n        config["user_turn_start"] = user_turn_start\n        if not is_gemma: config["system_turn_start"] = system_turn_start\n    elif version == "v1":'
content = content.replace('    if version == "v1":\n        config["stop_word"] = stop_v1', new + '\n        config["stop_word"] = stop_v1')

# Remove old else block to avoid duplication
content = re.sub(r'    else:\n        config\["stop_word"\].*?system_turn_start', '', content, flags=re.DOTALL)
with open("benchmark_runners/flexi_runner.py", "w") as f: f.write(content)
print("done")
