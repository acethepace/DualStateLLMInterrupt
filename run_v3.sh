#!/bin/bash
cd benchmark_runners
rm -rf ../results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v3/
rm -f ../gpu.lock gpu.lock

source ../.venv/bin/activate
python3 flexi_runner.py --model_name unsloth/Meta-Llama-3.1-8B-Instruct --version v3 --dataset_path ../mini_flexi.json > run.log 2>&1 & echo $! > runner.pid
