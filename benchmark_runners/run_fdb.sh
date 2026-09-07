#!/bin/bash
source ../.venv/bin/activate
echo "Running FDB Multi-Agent (GPT-5.6-Luna vs Gemma V2-Prefill)..."
python3 fdb_runner.py --model_name unsloth/gemma-4-12b-it --version v2_prefill

echo "Running FDB Multi-Agent (GPT-5.6-Luna vs Gemma V1)..."
python3 fdb_runner.py --model_name unsloth/gemma-4-12b-it --version v1
