import json

def compute_matrix(path):
    data = json.load(open(path))
    scenarios = data.get('scenarios', [])
    tp, tn, fp, fn = 0, 0, 0, 0
    latencies = []
    
    for s in scenarios:
        expected = s.get('expected_action')
        success = s.get('success')
        actual = expected if success else ("STOP" if expected == "CONTINUE" else "CONTINUE")
            
        if expected == "STOP" and actual == "STOP": tp += 1
        elif expected == "CONTINUE" and actual == "CONTINUE": tn += 1
        elif expected == "CONTINUE" and actual != "CONTINUE": fp += 1
        elif expected == "STOP" and actual != "STOP": fn += 1
            
        for event in s.get('events', []):
            lat = event.get('assessor_latency_ms')
            if lat: latencies.append(lat)
            
    total = tp+tn+fp+fn
    acc = (tp+tn)/total*100 if total > 0 else 0
    avg_lat = sum(latencies)/len(latencies) if latencies else 0
    print(f"TP: {tp} (True Barge-in detected)")
    print(f"TN: {tn} (Benign Backchannel ignored)")
    print(f"FP: {fp} (False Positive: Halting on backchannel)")
    print(f"FN: {fn} (False Negative: Failing to halt on barge-in)")
    print(f"Total Accuracy: {acc:.2f}%")
    print(f"Average Latency: {avg_lat:.2f} ms")

compute_matrix("results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v1/flexi_results.json")
