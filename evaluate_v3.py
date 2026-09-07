import json

with open("results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v4/flexi_results.json") as f:
    data = json.load(f)

tp = 0; tn = 0; fp = 0; fn = 0

for sc in data["scenarios"]:
    expected = sc["expected_action"]
    success = sc["success"]
    if expected == "STOP":
        if success: tp += 1
        else: fn += 1
    else:
        if success: tn += 1
        else: fp += 1

print(f"Total: {len(data['scenarios'])} | TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn} | Accuracy: {(tp+tn)/len(data['scenarios'])*100:.2f}%")

print("\n--- Confidence Analysis ---")
for sc in data["scenarios"]:
    print(f"Scenario {sc['id']} (Expected {sc['expected_action']}):")
    for ev in sc["events"]:
        action = ev['assessor_output']
        conf = ev.get('confidence', 0.0)
        user_input = "NO USER INPUT"
        if "[USER BARGE-IN]" in ev["input_chunk"]:
            user_input = ev["input_chunk"].split("[USER BARGE-IN]:")[1].strip()
        print(f"  Chunk {ev['chunk_index']} ({user_input[:20]}...): output {action} with STOP confidence: {conf:.2%}")
