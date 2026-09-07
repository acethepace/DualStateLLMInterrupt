import json, sys

desc = sys.argv[1]
prompt = sys.argv[2]
try:
    with open("results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v2/flexi_results.json") as f:
        res = json.load(f)
        tp = tn = fp = fn = 0
        for r in res.get("scenarios", []):
            if r.get("expected_action") == "STOP":
                if any(e.get("triggered_stop") for e in r.get("events", [])): tp += 1
                else: fn += 1
            elif r.get("expected_action") == "CONTINUE":
                if any(e.get("triggered_stop") for e in r.get("events", [])): fp += 1
                else: tn += 1
        total = tp + tn + fp + fn
except:
    tp = tn = fp = fn = total = 0

entry = {
    "description": desc,
    "prompt": prompt,
    "results": {
        "FLEXI": {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "Total": total}
    }
}

try:
    with open("notebooklm/Agentic_Interrupt/llama_v2_experiment_history.json", "r") as f:
        history = json.load(f)
except:
    history = []

history.append(entry)
with open("notebooklm/Agentic_Interrupt/llama_v2_experiment_history.json", "w") as f:
    json.dump(history, f, indent=4)
print("Logged to v2 history!")
