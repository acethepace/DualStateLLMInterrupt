#!/bin/bash
MODEL="unsloth/Meta-Llama-3.1-8B-Instruct"

echo "=== LLAMA 8B V1 ITERATION 3 ==="
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
PROMPT="You are a VAD. Output JSON: {\"t\": \"<1-word reason>\", \"a\": \"<H or P>\"}. If intentional interruption, use H. If benign backchannel, use P."

python3 append_history_v1.py "Iteration 3: Compact JSON Reasoning (max_tokens=15)" "$PROMPT"
