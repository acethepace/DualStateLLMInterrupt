#!/bin/bash
MODEL="unsloth/Meta-Llama-3.1-8B-Instruct"

echo "=== LLAMA 8B V1 ITERATION 4 ==="
cd benchmark_runners

rm -rf ../results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v1/
rm -rf ../results/unsloth_Meta-Llama-3.1-8B-Instruct/fdb/v1/
rm -f ../gpu.lock
rm -f gpu.lock

# Revert to standard 1-token strings
python3 -c '
import re
for filename in ["flexi_runner.py", "fdb_runner.py"]:
    with open(filename, "r") as f: content = f.read()
    content = re.sub(r"config\[\"max_new_tokens\"\] = .*", "config[\"max_new_tokens\"] = 15", content)
    content = re.sub(r"stop_v1, cont_v1, stop_v2 = .*", "stop_v1, cont_v1, stop_v2 = \"STOP\", \"CONTINUE\", \"HALT\"", content)
    with open(filename, "w") as f: f.write(content)
'

# The strictest possible prompt
PROMPT="You are an ultra-conservative Semantic VAD. 99% of the time, you should output EXACTLY 'CONTINUE'. The user will make sounds like 'uh huh', 'okay', 'yeah', 'hmm', or ambient noise. ALL OF THESE ARE 'CONTINUE'. ONLY if the user asks a fully formed, explicit new question (e.g., 'What is the weather?'), you output EXACTLY 'STOP'. If in doubt, output 'CONTINUE'."

python3 -c '
import sys, re
prompt = """'"$PROMPT"'"""
for filename in ["flexi_runner.py", "fdb_runner.py"]:
    with open(filename, "r") as f: lines = f.readlines()
    for i in range(len(lines)):
        if lines[i].strip().startswith("base_v1 ="): lines[i] = f"    base_v1 = \"\"\"{prompt}\"\"\"\n"
    with open(filename, "w") as f: f.writelines(lines)
'

source ../.venv/bin/activate
echo "Running FLEXI V1..."
python3 flexi_runner.py --model_name $MODEL --version v1 --dataset_path ../real_flexi.json > /dev/null
echo "Running FDB V1..."
python3 fdb_runner.py --model_name $MODEL --version v1 > /dev/null

cd ..
python3 append_history_v1.py "Iteration 4: Ultra-Conservative No-Reasoning (max_tokens=15)" "$PROMPT"
