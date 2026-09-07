import re

with open("benchmark_runners/flexi_runner.py", "r") as f:
    content = f.read()

content = re.sub(
    r'    else:\n        system_turn_start.*?\n        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"',
    '    else:\n        system_turn_start = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n"\n        user_turn_start = "<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n"\n        assessor_trigger = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\nClassification: "\n        thought_block = ""\n        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"',
    content,
    flags=re.DOTALL
)

with open("benchmark_runners/flexi_runner.py", "w") as f:
    f.write(content)
