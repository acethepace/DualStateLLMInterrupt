import json
import glob

print("Llama 3.1 8B Results:")
for f in sorted(glob.glob("results/unsloth_Meta-Llama-3.1-8B-Instruct/*.json")):
    try:
        data = json.load(open(f))
        if "accuracy" in data:
            print(f"{f.split('/')[-1]}: Acc={data['accuracy']}, TP_Lat={data.get('interruption_latency_ms', 0):.2f}, TN_Lat={data.get('noop_latency_ms', 0):.2f}")
    except:
        pass

print("\nQwen 3 4B Results:")
for f in sorted(glob.glob("results/Qwen_Qwen3-4B-Instruct-2507/*.json")):
    try:
        data = json.load(open(f))
        if "accuracy" in data:
            print(f"{f.split('/')[-1]}: Acc={data['accuracy']}, TP_Lat={data.get('interruption_latency_ms', 0):.2f}, TN_Lat={data.get('noop_latency_ms', 0):.2f}")
    except:
        pass
