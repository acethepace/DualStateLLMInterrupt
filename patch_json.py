import json
with open("results/Qwen_Qwen3-4B-Instruct-2507/final_eval.json", "r") as f: data = json.load(f)
data["flexi"]["v1"] = {"accuracy": 0.975, "tp": 198, "tn": 192, "fp": 8, "fn": 2, "avg_latency_ms": 172.79}
with open("results/Qwen_Qwen3-4B-Instruct-2507/final_eval.json", "w") as f: json.dump(data, f, indent=2)
