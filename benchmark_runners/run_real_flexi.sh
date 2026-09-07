#!/bin/bash
source ../.venv/bin/activate
echo "Running REAL FLEXI for Gemma (V1)..."
python3 flexi_runner.py --model_name unsloth/gemma-4-12b-it --version v1 --dataset_path ../real_flexi.json

echo "Running REAL FLEXI for Gemma (V2 Prefill)..."
python3 flexi_runner.py --model_name unsloth/gemma-4-12b-it --version v2_prefill --dataset_path ../real_flexi.json
