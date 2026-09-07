import json
import sys
import os

def compute_matrix(path):
    if not os.path.exists(path):
        return {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "Total": 0, "Avg_Latency_ms": 0}
    try:
        data = json.load(open(path))
    except:
        return {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "Total": 0, "Avg_Latency_ms": 0}
        
    tp, tn, fp, fn = 0, 0, 0, 0
    is_fdb = 'fdb' in path
    latencies = []
    scenarios = data.get('scenarios', [])
    for s in scenarios:
        expected = s.get('expected_action') if not is_fdb else s.get('expected')
        if expected is None: continue
        if is_fdb:
            actual = s.get('actual')
        else:
            success = s.get('success')
            actual = expected if success else ("STOP" if expected == "CONTINUE" else "CONTINUE")
        if expected == "STOP" and actual == "STOP": tp += 1
        elif expected == "CONTINUE" and actual == "CONTINUE": tn += 1
        elif expected == "CONTINUE" and actual == "STOP": fp += 1
        elif expected == "STOP" and actual == "CONTINUE": fn += 1
        if "assessor_latency_ms" in s: latencies.append(s["assessor_latency_ms"])
        elif "latency_ms" in s: latencies.append(s["latency_ms"])
    avg_lat = sum(latencies)/len(latencies) if latencies else 0
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "Total": tp+tn+fp+fn, "Avg_Latency_ms": avg_lat}

desc = sys.argv[1]
prompt = sys.argv[2]
flexi_res = compute_matrix("results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v2_prefill/flexi_results.json")
fdb_res = compute_matrix("results/unsloth_Meta-Llama-3.1-8B-Instruct/fdb/v2_prefill/results.json")

entry = {
    "description": desc,
    "prompt": prompt,
    "results": {
        "FLEXI": flexi_res,
        "FDB": fdb_res
    }
}

history_path = "notebooklm/Agentic_Interrupt/llama_experiment_history.json"
os.makedirs(os.path.dirname(history_path), exist_ok=True)
history = []
if os.path.exists(history_path):
    history = json.load(open(history_path))
history.append(entry)
json.dump(history, open(history_path, "w"), indent=4)
print(json.dumps(entry, indent=2))
