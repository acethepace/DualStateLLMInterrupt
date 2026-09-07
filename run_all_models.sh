#!/bin/bash
echo "Running Gemma 4..."
python3.11 run_handraiser.py "unsloth/gemma-4-12b-it" > gemma_out.log 2>&1
echo "Running Qwen 3..."
python3.11 run_handraiser.py "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50" > qwen_out.log 2>&1
echo "Done."
