import json

def get_accuracy(path):
    with open(path, "r") as f:
        data = json.load(f)
    correct = sum(1 for s in data["scenarios"] if s["expected"] == s["actual"])
    tp = sum(1 for s in data["scenarios"] if s["expected"] == "STOP" and s["actual"] == "STOP")
    tn = sum(1 for s in data["scenarios"] if s["expected"] == "CONTINUE" and s["actual"] == "CONTINUE")
    fp = sum(1 for s in data["scenarios"] if s["expected"] == "CONTINUE" and s["actual"] == "STOP")
    fn = sum(1 for s in data["scenarios"] if s["expected"] == "STOP" and s["actual"] == "CONTINUE")
    return correct, len(data["scenarios"]), tp, tn, fp, fn

v1 = get_accuracy("results/Qwen_Qwen3-4B-Instruct-2507/fdb/v1/results.json")
v2 = get_accuracy("results/Qwen_Qwen3-4B-Instruct-2507/fdb/v2_prefill/results.json")

print(f"v1 (Baseline): {v1[0]}/{v1[1]} ({v1[0]/v1[1]*100:.2f}%) - TP:{v1[2]} TN:{v1[3]} FP:{v1[4]} FN:{v1[5]}")
print(f"v2 (Logit Gating): {v2[0]}/{v2[1]} ({v2[0]/v2[1]*100:.2f}%) - TP:{v2[2]} TN:{v2[3]} FP:{v2[4]} FN:{v2[5]}")
