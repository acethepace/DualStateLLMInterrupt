#!/bin/bash
echo "Running Llama 3.1 8B..."
python3.11 run_handraiser.py "unsloth/Meta-Llama-3.1-8B-Instruct" > llama_full.log 2>&1
echo "Running Qwen 3 4B..."
python3.11 run_handraiser.py "Qwen/Qwen3-4B-Instruct-2507" > qwen_full.log 2>&1
echo "Finished."
