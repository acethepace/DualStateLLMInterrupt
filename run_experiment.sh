#!/bin/bash
PROMPT="$1"

# 1. Update prompts
cd benchmark_runners
sed -i -E "s/base_v2 = \".*\"/base_v2 = \"$PROMPT\"/g" flexi_runner.py
sed -i -E "s/base_v2 = \".*\"/base_v2 = \"$PROMPT\"/g" fdb_runner.py

# 2. Clear previous v2_prefill results
rm -rf ../results/unsloth_gemma-4-12b-it/flexi/v2_prefill/
rm -rf ../results/unsloth_gemma-4-12b-it/fdb/v2_prefill/
rm -f ../gpu.lock

# 3. Run benchmarks sequentially
source ../.venv/bin/activate
echo "Running FLEXI..."
python3 flexi_runner.py --model_name unsloth/gemma-4-12b-it --version v2_prefill --dataset_path ../real_flexi.json > /dev/null
echo "Running FDB..."
python3 fdb_runner.py --model_name unsloth/gemma-4-12b-it --version v2_prefill > /dev/null

# 4. Print confusion matrices
cd ..
python3 parse_matrices.py | grep -E "V2_Prefill" -A 1
