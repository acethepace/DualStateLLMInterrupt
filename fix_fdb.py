import sys

with open("benchmark_runners/fdb_runner.py", "r") as f:
    code = f.read()

code = code.replace("""    else:
        user_turn_start = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n"
        assessor_trigger = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
        thought_block = ""
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"
    elif is_qwen:
        user_turn_start = "<|im_start|>user\\n"
        assessor_trigger = "<|im_end|>\\n<|im_start|>assistant\\nClassification: "
        thought_block = ""
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP\"\"\"", 
\"\"\"    elif is_qwen:
        user_turn_start = "<|im_start|>user\\n"
        assessor_trigger = "<|im_end|>\\n<|im_start|>assistant\\nClassification: "
        thought_block = ""
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"
    else:
        user_turn_start = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n"
        assessor_trigger = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
        thought_block = ""
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP\"\"\")

with open("benchmark_runners/fdb_runner.py", "w") as f:
    f.write(code)
