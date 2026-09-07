#!/bin/bash
source .venv/bin/activate
set -e

echo "Running Gemma v2..."
python run_benchmark.py --model_name unsloth/gemma-4-12b-it --version v2
echo "Running Gemma v1..."
python run_benchmark.py --model_name unsloth/gemma-4-12b-it --version v1
echo "Running Gemma v2_prefill..."
python run_benchmark.py --model_name unsloth/gemma-4-12b-it --version v2_prefill

echo "Running Llama-3-8B-Instruct v2..."
python run_benchmark.py --model_name unsloth/llama-3-8b-Instruct-bnb-4bit --version v2
echo "Running Llama-3-8B-Instruct v1..."
python run_benchmark.py --model_name unsloth/llama-3-8b-Instruct-bnb-4bit --version v1

echo "All benchmarks completed."
