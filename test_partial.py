import json

def eval_flexi(path):
    data = json.load(open(path))
    scenarios = data.get('scenarios', [])
    tp, tn, fp, fn = 0, 0, 0, 0
    for s in scenarios:
        expected = s.get('expected_action')
        success = s.get('success')
        actual = expected if success else ("STOP" if expected == "CONTINUE" else "CONTINUE")
        
        if expected == "STOP" and actual == "STOP": tp += 1
        elif expected == "CONTINUE" and actual == "CONTINUE": tn += 1
        elif expected == "CONTINUE" and actual != "CONTINUE": fp += 1
        elif expected == "STOP" and actual != "STOP": fn += 1
    total = tp + tn + fp + fn
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    if total > 0:
        print(f"Accuracy: {(tp+tn)/total*100:.2f}%")

eval_flexi("results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v1/flexi_results.json")
