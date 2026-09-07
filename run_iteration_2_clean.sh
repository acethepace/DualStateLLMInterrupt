#!/bin/bash
MODEL="unsloth/Meta-Llama-3.1-8B-Instruct"

echo "=== LLAMA 8B V1 ITERATION 2 ==="
cd benchmark_runners

rm -rf ../results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v1/
rm -rf ../results/unsloth_Meta-Llama-3.1-8B-Instruct/fdb/v1/
rm -f ../gpu.lock
rm -f gpu.lock

source ../.venv/bin/activate
echo "Running FLEXI V1..."
python3 flexi_runner.py --model_name $MODEL --version v1 --dataset_path ../real_flexi.json > /dev/null
echo "Running FDB V1..."
python3 fdb_runner.py --model_name $MODEL --version v1 > /dev/null

cd ..
PROMPT="You are a Semantic VAD. Read the user's speech and output a JSON object: {\"thought\": \"<brief thought>\", \"action\": \"<[HALT] or [PROCEED]>\"}. If the user is making an intentional interruption to stop the assistant, use [HALT]. If the user is making a benign backchannel (e.g. 'uh huh', 'okay', 'yeah'), use [PROCEED]. Keep thought under 10 words."

python3 append_history_v1.py "Iteration 2: JSON Reasoning (max_tokens=35)" "$PROMPT"
