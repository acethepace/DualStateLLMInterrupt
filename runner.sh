#!/bin/bash
source .venv/bin/activate
python3 run_all_llama.py > llama_flexi.log 2>&1
python3 run_fdb_llama.py > llama_fdb.log 2>&1
