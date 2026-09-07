#!/bin/bash
MODEL="unsloth/Meta-Llama-3.1-8B-Instruct"

echo "=== LLAMA 8B BENCHMARK RUN ==="
source .venv/bin/activate
cd benchmark_runners

for VER in v1 v2_prefill
do
    echo "Running FLEXI $VER for $MODEL..."
    python3 flexi_runner.py --model_name $MODEL --version $VER --dataset_path ../real_flexi.json > /dev/null
    
    echo "Running FDB $VER for $MODEL..."
    python3 fdb_runner.py --model_name $MODEL --version $VER > /dev/null
done

cd ..
echo "Parsing matrices..."
python3 parse_matrices.py > llama_matrices.txt
