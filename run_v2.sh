#!/bin/bash
PROMPT="$1"

cd benchmark_runners
rm -rf ../results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v2/
rm -f ../gpu.lock gpu.lock

python3 -c '
import sys
prompt = sys.argv[1]
for filename in ["flexi_runner.py", "fdb_runner.py"]:
    with open(filename, "r") as f: content = f.read()
    import re
    # Strip old base_v2 definitions completely
    content = re.sub(r"    base_v2 = .*\n", "", content)
    # Insert new base_v2 right before config
    content = content.replace("    if is_gemma:", f"    base_v2 = \"\"\"{prompt}\"\"\"\n    if is_gemma:")
    # Replace stop_v2 mapping
    content = re.sub(r"stop_v1, cont_v1, stop_v2 = .*", "stop_v1, cont_v1, stop_v2 = \"STOP\", \"CONTINUE\", \"STOP\"", content)
    # Turn off json instruction inside config
    content = content.replace("config[\"base_instructions\"] = base_v2", "config[\"base_instructions\"] = base_v2")
    with open(filename, "w") as f: f.write(content)
' "$PROMPT"

source ../.venv/bin/activate
echo "Running FLEXI V2 with prompt..."
python3 flexi_runner.py --model_name unsloth/Meta-Llama-3.1-8B-Instruct --version v2 --dataset_path ../real_flexi.json > /dev/null
