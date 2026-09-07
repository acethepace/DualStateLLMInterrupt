import json
with open("temp.json") as f:
    results = json.load(f)

tp = tn = fp = fn = 0
for r in results["scenarios"]:
    if r.get("expected_action") == "STOP":
        if any(e.get("triggered_stop") for e in r.get("events", [])): tp += 1
        else: fn += 1
    elif r.get("expected_action") == "CONTINUE":
        if any(e.get("triggered_stop") for e in r.get("events", [])): fp += 1
        else: tn += 1

total = tp + tn + fp + fn
print(f"Total: {total}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
if total > 0:
    print(f"Accuracy: {(tp + tn) / total * 100:.2f}%")
