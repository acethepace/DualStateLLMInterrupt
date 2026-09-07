import json

def compute_matrix(path):
    try:
        data = json.load(open(path))
        tp, tn, fp, fn = 0, 0, 0, 0
        is_fdb = 'fdb' in path
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
        return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "Total": tp+tn+fp+fn}
    except Exception as e:
        return str(e)

paths = {
    "Llama FLEXI (V1)": "results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v1/flexi_results.json",
    "Llama FDB (V1)": "results/unsloth_Meta-Llama-3.1-8B-Instruct/fdb/v1/results.json",
}

for name, path in paths.items():
    print(f"--- {name} ---")
    print(compute_matrix(path))
