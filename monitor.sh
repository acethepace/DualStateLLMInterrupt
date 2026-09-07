#!/bin/bash
while true; do
    if [ -f results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v1/flexi_results.json ]; then
        python3 test_flexi.py
        count=$(python3 -c "import json; print(len(json.load(open('results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v1/flexi_results.json'))))" 2>/dev/null || echo 0)
        if [ "$count" -eq 400 ]; then
            echo "FLEXI finished!"
            break
        fi
    fi
    sleep 10
done
