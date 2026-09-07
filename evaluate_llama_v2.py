import json, sys, os

def eval_results():
    if not os.path.exists("results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v2/flexi_results.json"):
        return 0, 0, 0, 0, 0, 0
    with open("results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v2/flexi_results.json") as f:
        try:
            results = json.load(f)
        except:
            return 0, 0, 0, 0, 0, 0

    tp = tn = fp = fn = 0
    for r in results.get("scenarios", []):
        if r.get("expected_action") == "STOP":
            if any(e.get("triggered_stop") for e in r.get("events", [])): tp += 1
            else: fn += 1
        elif r.get("expected_action") == "CONTINUE":
            if any(e.get("triggered_stop") for e in r.get("events", [])): fp += 1
            else: tn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total * 100 if total > 0 else 0
    return acc, tp, tn, fp, fn, total

if __name__ == "__main__":
    acc, tp, tn, fp, fn, total = eval_results()
    print(f"Total: {total} | TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn} | Accuracy: {acc:.2f}%")
