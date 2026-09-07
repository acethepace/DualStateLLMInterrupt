import sys

# Patch dual_state_interruptor.py
with open("dual_state_interruptor.py", "r") as f:
    ds_code = f.read()

ds_code = ds_code.replace("stop_ids = [51769, 46637]", "stop_ids = [50669] if 'Qwen' in model.name_or_path else [51769, 46637]")
ds_code = ds_code.replace("cont_ids = [24194, 16511]", "cont_ids = [35045] if 'Qwen' in model.name_or_path else [24194, 16511]")
ds_code = ds_code.replace("stop_prob = probs[0].item() + probs[1].item()", "stop_prob = probs[0].item() if 'Qwen' in model.name_or_path else probs[0].item() + probs[1].item()")
ds_code = ds_code.replace("cont_prob = probs[2].item() + probs[3].item()", "cont_prob = probs[1].item() if 'Qwen' in model.name_or_path else probs[2].item() + probs[3].item()")

with open("dual_state_interruptor.py", "w") as f:
    f.write(ds_code)

# Patch fdb_runner.py
with open("benchmark_runners/fdb_runner.py", "r") as f:
    fdb_code = f.read()

fdb_code = fdb_code.replace('is_gemma = "gemma" in model_name.lower()', 'is_gemma = "gemma" in model_name.lower()\n    is_qwen = "qwen" in model_name.lower()')

qwen_templates = """
    elif is_qwen:
        user_turn_start = "<|im_start|>user\\n"
        assessor_trigger = "<|im_end|>\\n<|im_start|>assistant\\nClassification: "
        thought_block = ""
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"
"""

fdb_code = fdb_code.replace('        thought_block = ""\n        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"', '        thought_block = ""\n        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"' + qwen_templates)

fdb_code = fdb_code.replace('config["base_instructions"] = base_v4', 'config["base_instructions"] = open("prompt_v2.txt").read() if is_qwen else base_v4')

with open("benchmark_runners/fdb_runner.py", "w") as f:
    f.write(fdb_code)
