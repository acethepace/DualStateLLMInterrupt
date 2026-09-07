import subprocess
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, required=True)
parser.add_argument("--desc", type=str, required=True)
parser.add_argument("--prompt_file", type=str, required=True)
args = parser.parse_args()

with open(args.prompt_file, "r") as f:
    prompt_text = f.read()

print(f"Running iteration {args.iter}...")
res = subprocess.run(["python3", "eval_qwen_subset.py", "--prompt_file", args.prompt_file], capture_output=True, text=True)

out_json = {
    "iteration": args.iter,
    "description_of_changes": args.desc,
    "prompt": prompt_text,
    "results_output": res.stdout,
    "error_output": res.stderr
}

os.makedirs("results/Qwen_Qwen3-4B-Instruct-2507", exist_ok=True)
out_path = f"results/Qwen_Qwen3-4B-Instruct-2507/iter{args.iter}_changes.json"
with open(out_path, "w") as f:
    json.dump(out_json, f, indent=2)

print(f"Saved to {out_path}")
print(res.stdout)
