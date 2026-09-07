import json
import os

files = [
    "results/unsloth_gemma-4-12b-it/flexi/v2_prefill/flexi_results.json",
    "results/unsloth_gemma-4-12b-it/fdb/v2_prefill/results.json"
]

for f in files:
    if os.path.exists(f):
        try:
            with open(f) as file:
                data = json.load(file)
                print(f"\n--- {f} ---")
                
                successes = 0
                total = 0
                
                if isinstance(data, list):
                    for item in data:
                        # Some formats have "success": True/False
                        if "success" in item:
                            successes += 1 if item["success"] else 0
                            total += 1
                        # FDB format has 'expected' and 'actual'
                        elif "expected" in item and "actual" in item:
                            if item["expected"] == item["actual"]:
                                successes += 1
                            total += 1
                        
                if total > 0:
                    print(f"Accuracy: {successes/total*100:.2f}% ({successes}/{total})")
                else:
                    print("Could not parse accuracy from data:", list(data[0].keys()) if len(data)>0 else "Empty")
        except Exception as e:
            print(f"Error parsing {f}: {e}")
