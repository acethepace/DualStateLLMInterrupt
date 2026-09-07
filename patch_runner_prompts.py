import os
import sys

prompt = """You are a Semantic VAD. Read the user's speech and output a JSON object: {"thought": "<brief thought>", "action": "<[HALT] or [PROCEED]>"}. If the user is making an intentional interruption to stop the assistant, use [HALT]. If the user is making a benign backchannel (e.g. 'uh huh', 'okay', 'yeah'), use [PROCEED]. Keep thought under 10 words."""

for filename in ["benchmark_runners/flexi_runner.py", "benchmark_runners/fdb_runner.py"]:
    with open(filename, "r") as f:
        content = f.read()
    
    # We can just replace the block directly.
    import re
    content = re.sub(r'base_v1 = ".*?"', f'base_v1 = """{prompt}"""', content, flags=re.DOTALL)
    
    with open(filename, "w") as f:
        f.write(content)
print("Patched python strings safely!")
