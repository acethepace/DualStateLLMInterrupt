import re

prompt = """You are a Semantic VAD. Read the user's speech and output a JSON object: {{"thought": "<brief thought>", "action": "<[HALT] or [PROCEED]>"}}. If the user is making an intentional interruption to stop the assistant, use [HALT]. If the user is making a benign backchannel (e.g. 'uh huh', 'okay', 'yeah'), use [PROCEED]. Keep thought under 10 words."""

for filename in ["benchmark_runners/flexi_runner.py", "benchmark_runners/fdb_runner.py"]:
    with open(filename, "r") as f:
        lines = f.readlines()
        
    for i in range(len(lines)):
        if lines[i].strip().startswith("base_v1 ="):
            lines[i] = f'    base_v1 = """{prompt}"""\n'
            
    with open(filename, "w") as f:
        f.writelines(lines)
        
print("Fixed braces!")
