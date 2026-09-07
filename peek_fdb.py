import json
data = json.load(open('results/unsloth_gemma-4-12b-it/fdb/v2_prefill/results.json'))
for s in data['scenarios']:
    if s['expected'] == 'STOP' and s['actual'] == 'CONTINUE':
        print(f"[{s['task_id']}] {s['interruption']}")
