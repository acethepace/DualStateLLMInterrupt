import re
with open("benchmark_runners/flexi_runner.py", "r") as f:
    content = f.read()

# find the get_templates function and replace it entirely
def_start = content.find("def get_templates")
def_end = content.find("def acquire_gpu_lock")

new_func = """def get_templates(model_name, version):
    is_gemma = "gemma" in model_name.lower()
    
    base_v1 = "You are an ultra-conservative Semantic VAD. 99% of the time, you should output EXACTLY 'CONTINUE'. The user will make sounds like 'uh huh', 'okay', 'yeah', 'hmm', or ambient noise. ALL OF THESE ARE 'CONTINUE'. ONLY if the user asks a fully formed, explicit new question (e.g., 'What is the weather?'), you output EXACTLY 'STOP'. If in doubt, output 'CONTINUE'."

    base_v2 = "You are a Semantic VAD classifier. Analyze the user's speech. If the user makes an intentional barge-in (e.g. asking a question, interrupting), output EXACTLY 'STOP'. If the user only utters a benign backchannel (e.g. 'uh huh', 'okay', 'yeah', 'right') or makes ambient noise, output EXACTLY 'CONTINUE'."
    
    if is_gemma:
        user_turn_start = "<bos><|turn>user\\n"
        assessor_trigger = "<turn|>\\n<|turn>model\\n"
        thought_block = "<|channel>thought\\n<channel|>"
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"
    else:
        system_turn_start = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n"
        user_turn_start = "<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n"
        assessor_trigger = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\nClassification: "
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
        if not is_gemma:
            config["system_turn_start"] = system_turn_start
    return config

"""

content = content[:def_start] + new_func + content[def_end:]

with open("benchmark_runners/flexi_runner.py", "w") as f:
    f.write(content)
print("done")
