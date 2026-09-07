import sys

with open("benchmark_runners/fdb_runner.py", "r") as f:
    code = f.read()
code = code.replace("if is_qwen:\n        user_turn_start = \"<|im_start|>user\\n\"", "if is_qwen:\n        user_turn_start = \"<|im_start|>user\\n\"")
