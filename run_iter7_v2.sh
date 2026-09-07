#!/bin/bash
PROMPT="You are an automated Semantic VAD classifier, NOT an assistant. You are forbidden from answering user questions. Your ONLY function is to classify the user's input stream. Output EXACTLY ONE WORD: \"STOP\" if the user is making an intentional barge-in (e.g. asking a question, interrupting). Output EXACTLY \"CONTINUE\" if the user is uttering a benign backchannel (e.g. \"uh huh\", \"okay\", \"yeah\"). If you output anything other than \"STOP\" or \"CONTINUE\", the system will crash. Do not generate conversational text."

cd benchmark_runners
rm -rf ../results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v2/
rm -f ../gpu.lock gpu.lock

python3 -c '
import sys
prompt = sys.argv[1]
for filename in ["flexi_runner.py", "fdb_runner.py"]:
    with open(filename, "r") as f: content = f.read()
    import re
    # Remove old base_v2
    content = re.sub(r"    base_v2 = .*\n", "", content)
    # Insert new base_v2
    content = content.replace("    if is_gemma:", f"    base_v2 = \"\"\"{prompt}\"\"\"\n    if is_gemma:")
    content = re.sub(r"stop_v1, cont_v1, stop_v2 = .*", "stop_v1, cont_v1, stop_v2 = \"STOP\", \"CONTINUE\", \"STOP\"", content)
    # Reset tokens just in case
    content = re.sub(r"config\[\"max_new_tokens\"\] = .*", "config[\"max_new_tokens\"] = 15", content)
    with open(filename, "w") as f: f.write(content)
' "$PROMPT"

source ../.venv/bin/activate
echo "Running FLEXI V2 Iteration 7..."
python3 flexi_runner.py --model_name unsloth/Meta-Llama-3.1-8B-Instruct --version v2 --dataset_path ../real_flexi.json > /dev/null
