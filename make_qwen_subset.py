import json

with open("real_flexi.json", "r") as f:
    data = json.load(f)

# Take 10 True Positives (expected STOP) and 10 True Negatives (expected CONTINUE/IGNORE)
tps = [d for d in data if d['expected_action'] == 'STOP'][:10]
tns = [d for d in data if d['expected_action'] == 'CONTINUE'][:10]
subset = tps + tns

# Rewrite the 'expected_action' for tns to 'IGNORE'
for d in subset:
    if d['expected_action'] == 'CONTINUE':
        d['expected_action'] = 'IGNORE'

with open("qwen_subset.json", "w") as f:
    json.dump(subset, f, indent=2)

print(f"Created qwen_subset.json with {len(subset)} items.")
