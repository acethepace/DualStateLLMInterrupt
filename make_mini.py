import json

with open("real_flexi.json") as f:
    data = json.load(f)

# Take 10 interruptions (first 10) and 10 backchannels (last 10)
mini_data = data[:10] + data[-10:]

with open("mini_flexi.json", "w") as f:
    json.dump(mini_data, f, indent=4)
print("Created mini_flexi.json!")
