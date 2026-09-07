#!/bin/bash
echo "Starting Gemma 4 inference..."
python3.11 run_handraiser.py "unsloth/gemma-4-12b-it" > gemma_final.log 2>&1
echo "Finished!"
