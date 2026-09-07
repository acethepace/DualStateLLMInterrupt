import json

def get_avg_length(file_path):
    with open(file_path) as f:
        data = json.load(f)
    if "scenarios" in data:
        data = data["scenarios"]
    elif isinstance(data, dict):
        # fdb maybe?
        if "data" in data:
            data = data["data"]
    
    lens = []
    for sc in data:
        # FDB
        if "transcript" in sc:
            lens.append(len(sc["transcript"].split()))
        # FLEXI
        elif "system_prompt" in sc:
            l = len(sc["system_prompt"].split())
            if "base_stream" in sc:
                l += sum(len(c.split()) for c in sc["base_stream"])
            lens.append(l)
    return sum(lens) / len(lens) if lens else 0

print("FLEXI Avg words:", get_avg_length("datasets/flexi.json") * 1.3) # approx tokens
