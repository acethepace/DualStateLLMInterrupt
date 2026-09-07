#!/bin/bash
PROMPT="$1"
TRIGGER="$2"

cd benchmark_runners
rm -rf ../results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v2/
rm -f ../gpu.lock gpu.lock

python3 -c '
import sys
prompt = sys.argv[1]
trigger = sys.argv[2]
for filename in ["flexi_runner.py"]:
    with open(filename, "r") as f: content = f.read()
    import re
    
    # Replace base_instructions dynamically
    content = re.sub(r"    base_v2 = .*\n", "", content)
    content = content.replace("    if is_gemma:", f"    base_v2 = \"\"\"{prompt}\"\"\"\n    if is_gemma:")
    
    # Replace assessor_trigger for Llama
    content = re.sub(
        r"assessor_trigger = \"<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\\n\\n\".*",
        f"assessor_trigger = \"{trigger}\"",
        content
    )
    
    # Ensure stop words are correct
    content = re.sub(r"stop_v1, cont_v1, stop_v2 = .*", "stop_v1, cont_v1, stop_v2 = \"STOP\", \"CONTINUE\", \"STOP\"", content)
    
    # Max tokens
    content = re.sub(r"config\[\"max_new_tokens\"\] = .*", "config[\"max_new_tokens\"] = 15", content)
    with open(filename, "w") as f: f.write(content)
' "$PROMPT" "$TRIGGER"

source ../.venv/bin/activate
python3 flexi_runner.py --model_name unsloth/Meta-Llama-3.1-8B-Instruct --version v2 --dataset_path ../mini_flexi.json > /dev/null
cd ..
python3 evaluate_llama_v2.py
