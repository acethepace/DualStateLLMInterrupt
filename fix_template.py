import re

with open("benchmark_runners/flexi_runner.py", "r") as f:
    content = f.read()

# Fix config definitions
old_else = """    else:
        user_turn_start = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n"
        assessor_trigger = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
        thought_block = ""
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"

    config = {}"""

new_else = """    else:
        system_turn_start = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n"
        user_turn_start = "<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n"
        assessor_trigger = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
        thought_block = ""
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"

    config = {}"""

if old_else in content:
    content = content.replace(old_else, new_else)

content = content.replace('config["user_turn_start"] = user_turn_start', 'config["user_turn_start"] = user_turn_start\n        if not is_gemma: config["system_turn_start"] = system_turn_start')

old_base_context = 'base_context = config["user_turn_start"] + config["base_instructions"] + "\\n\\n" + scenario["system_prompt"] + "\\n\\n<INPUT>\\n"'
new_base_context = """if "system_turn_start" in config:
                base_context = config["system_turn_start"] + config["base_instructions"] + config["user_turn_start"] + scenario["system_prompt"] + "\\n\\n<INPUT>\\n"
            else:
                base_context = config["user_turn_start"] + config["base_instructions"] + "\\n\\n" + scenario["system_prompt"] + "\\n\\n<INPUT>\\n\""""

content = content.replace(old_base_context, new_base_context)

with open("benchmark_runners/flexi_runner.py", "w") as f:
    f.write(content)
print("Patched flexi_runner.py template logic!")
