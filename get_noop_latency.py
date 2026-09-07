import json
import os

def get_noop_fdb(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)["scenarios"]
        latencies = [s['latency_ms'] for s in data if s.get('expected') == 'CONTINUE']
        return latencies
    except Exception as e:
        return []

def get_noop_flexi(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)["scenarios"]
        latencies = []
        for s in data:
            if s.get("expected_action") == "CONTINUE":
                for ev in s.get('events', []):
                    if 'assessor_latency_ms' in ev:
                        latencies.append(ev['assessor_latency_ms'])
        return latencies
    except Exception as e:
        return []

flexi_llama = get_noop_flexi("results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v4/flexi_results.json")
if os.path.exists("results/unsloth_Meta-Llama-3.1-8B-Instruct/fdb/v4/results.json"):
    fdb_llama = get_noop_fdb("results/unsloth_Meta-Llama-3.1-8B-Instruct/fdb/v4/results.json")
else:
    fdb_llama = []

all_llama = fdb_llama + flexi_llama
if all_llama:
    print(f"Llama Combined No-Op Latency: {sum(all_llama)/len(all_llama):.2f} ms (Count: {len(all_llama)})")
