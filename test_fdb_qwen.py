import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import benchmark_runners.fdb_runner as fdb

def custom_load(x):
    d = json.load(open(x))
    return d["splits"]["Daily"]["classes"][0]["tasks"][:2]
fdb.load_fdb_tasks = custom_load
fdb.simulate_fdb_multi_agent("Qwen/Qwen3-4B-Instruct-2507", "v2_prefill", "datasets/Full-Duplex-Bench/v2/prompts_staged_200.json")
