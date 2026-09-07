#!/bin/bash
PROMPT="You are a Semantic VAD classifier.
Classify the user's input stream as EXACTLY 'STOP' or 'CONTINUE'.
RULES:
1. If the user makes an intentional barge-in (asking a question, correcting a fact, changing the topic), output 'STOP'.
2. If the user utters a benign backchannel to show they are listening (e.g., 'uh huh', 'okay', 'yeah', 'right', 'hmm', 'Ah, okay.', 'Oh, interesting!', 'Really?') or makes ambient noise, output 'CONTINUE'.

EXAMPLES:
Input: [USER BARGE-IN]: Wait, can you repeat that?
Classification: STOP

Input: [USER BARGE-IN]: uh huh
Classification: CONTINUE

Input: [USER BARGE-IN]: What about the second point?
Classification: STOP

Input: [USER BARGE-IN]: Ah, okay.
Classification: CONTINUE

Input: [USER BARGE-IN]: I think you are wrong.
Classification: STOP

Input: [USER BARGE-IN]: Oh, interesting!
Classification: CONTINUE

Input: [USER BARGE-IN]: Really?
Classification: CONTINUE

Input: [USER BARGE-IN]: yeah
Classification: CONTINUE

Input: [USER BARGE-IN]: How does that work?
Classification: STOP
"
TRIGGER="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nClassification: "

cd benchmark_runners
rm -rf ../results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v2/
rm -f ../gpu.lock gpu.lock

python3 -c '
import sys, re
prompt = sys.argv[1]
trigger = sys.argv[2]
with open("flexi_runner.py", "r") as f: content = f.read()
content = re.sub(r"    base_v2 = \".*?\"\n    \n    if is_gemma:", f"    base_v2 = \"\"\"{prompt}\"\"\"\n    \n    if is_gemma:", content, flags=re.DOTALL)
with open("flexi_runner.py", "w") as f: f.write(content)
' "$PROMPT" "$TRIGGER"

source ../.venv/bin/activate
python3 flexi_runner.py --model_name unsloth/Meta-Llama-3.1-8B-Instruct --version v2 --dataset_path ../mini_flexi.json > run.log 2>&1 & echo $! > runner.pid
