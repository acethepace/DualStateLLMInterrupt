import re
for runner in ["benchmark_runners/flexi_runner.py", "benchmark_runners/fdb_runner.py"]:
    with open(runner, "r") as f:
        content = f.read()
    
    # Replace base_v2 string
    content = re.sub(r'base_v2 = ".*?"', 'base_v2 = """You are a Semantic Voice Activity Detector..."""', content)
    
    # Update max_new_tokens for v2
    new_config = """    if version == "v1":
        config["stop_word"] = stop_v1
        config["max_new_tokens"] = 15
        config["assessor_prompt"] = assessor_trigger
        config["base_instructions"] = base_v1.format(stop_word=stop_v1, cont_word=cont_v1)
        config["user_turn_start"] = user_turn_start
    else:
        config["stop_word"] = stop_v2
        config["max_new_tokens"] = 30 if not is_gemma else 1
        config["assessor_prompt"] = assessor_trigger + (thought_block if version == "v2_prefill" else "")
        config["base_instructions"] = base_v2
        config["user_turn_start"] = user_turn_start
    return config"""
    
    content = re.sub(r'    if version == "v1":.*return config', new_config, content, flags=re.DOTALL)
    
    with open(runner, "w") as f:
        f.write(content)
