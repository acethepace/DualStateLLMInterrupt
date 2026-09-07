import json

with open("datasets/fdb_dataset.json", "r") as f:
    old_fdb = json.load(f)

scenarios = old_fdb["scenarios"]

new_fdb_data = []
for sc in scenarios:
    t_id = "_".join(sc["task_id"].split("_")[:-1])
    transcript = f"Here is the information you requested about {t_id}. First, it is important to note that..."
    
    new_fdb_data.append({
        "task_id": sc["task_id"],
        "transcript": transcript,
        "user_interruption": sc["interruption"],
        "expected_action": sc["expected"]
    })

with open("datasets/fdb_dataset_recovered.json", "w") as f:
    json.dump(new_fdb_data, f, indent=4)
