import sys, re

with open("benchmark_runners/flexi_runner.py", "r") as f:
    content = f.read()

# Fix the syntax error manually
content = re.sub(
    r'assessor_trigger = "<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\n.*',
    'assessor_trigger = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\nClassification: "',
    content,
    flags=re.MULTILINE
)

with open("benchmark_runners/flexi_runner.py", "w") as f:
    f.write(content)
