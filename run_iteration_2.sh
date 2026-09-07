#!/bin/bash
MODEL="unsloth/Meta-Llama-3.1-8B-Instruct"

echo "=== LLAMA 8B V1 ITERATION 2 ==="
cd benchmark_runners

# Set up the new JSON reasoning prompt!
PROMPT="You are a Semantic VAD. Read the user's speech and output a JSON object: {\"thought\": \"<brief thought>\", \"action\": \"<[HALT] or [PROCEED]>\"}. If the user is making an intentional interruption to stop the assistant, use [HALT]. If the user is making a benign backchannel (e.g. 'uh huh', 'okay', 'yeah'), use [PROCEED]. Keep thought under 10 words."

ESCAPED_PROMPT=$(echo "$PROMPT" | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')

# Patch runners for Iteration 2
sed -i -E "s/base_v1 = \".*?\"/base_v1 = \"$ESCAPED_PROMPT\"/g" flexi_runner.py
sed -i -E "s/base_v1 = \".*?\"/base_v1 = \"$ESCAPED_PROMPT\"/g" fdb_runner.py

# Increase max_new_tokens to 35 for JSON
sed -i -E 's/config\["max_new_tokens"\] = 15/config["max_new_tokens"] = 35/g' flexi_runner.py
sed -i -E 's/config\["max_new_tokens"\] = 15/config["max_new_tokens"] = 35/g' fdb_runner.py

# Change stop and cont words to HALT and PROCEED
sed -i -E 's/stop_v1 = "STOP"/stop_v1 = "[HALT]"/g' flexi_runner.py
sed -i -E 's/cont_v1 = "CONTINUE"/cont_v1 = "[PROCEED]"/g' flexi_runner.py
sed -i -E 's/stop_v1 = "STOP"/stop_v1 = "[HALT]"/g' fdb_runner.py
sed -i -E 's/cont_v1 = "CONTINUE"/cont_v1 = "[PROCEED]"/g' fdb_runner.py

rm -rf ../results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v1/
rm -rf ../results/unsloth_Meta-Llama-3.1-8B-Instruct/fdb/v1/
rm -f ../gpu.lock

source ../.venv/bin/activate
echo "Running FLEXI V1..."
python3 flexi_runner.py --model_name $MODEL --version v1 --dataset_path ../real_flexi.json > /dev/null
echo "Running FDB V1..."
python3 fdb_runner.py --model_name $MODEL --version v1 > /dev/null

cd ..
python3 append_history_v1.py "Iteration 2: JSON Reasoning (max_tokens=35)" "$PROMPT"
