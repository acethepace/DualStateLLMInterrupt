import subprocess
import json
import re
import os

PROMPT = """You are a Semantic Voice Activity Detection (VAD) classifier.
Your ONLY job is to classify the ongoing audio stream as EXACTLY 'STOP' or 'CONTINUE'.

RULES:
1. NO USER INPUT: If there is no `[USER BARGE-IN]:` marker in the recent text (the system is just talking), output 'CONTINUE'.
2. BENIGN BACKCHANNEL: If the user utters a short backchannel to show they are following along, agreeing, or expressing mild surprise (e.g., 'uh huh', 'okay', 'yeah', 'right', 'hmm', 'Ah, okay.', 'Oh, interesting!', 'Really?', 'Of course!', 'No way!', 'Yes, exactly.', 'I understand.', 'Cool!', 'Right on!'), output 'CONTINUE'.
3. INTENTIONAL BARGE-IN: If the user asks a new question, corrects a fact, changes the topic, or tells the system to wait/stop, output 'STOP'.

EXAMPLES:

Input: The process of photosynthesis is
</INPUT>
[USER BARGE-IN]: Wait, can you explain the Calvin cycle?
Classification: STOP

Input: It requires a lot of energy.
Classification: CONTINUE

Input: The process of photosynthesis is
</INPUT>
[USER BARGE-IN]: uh huh
Classification: CONTINUE

Input: And then the reaction
</INPUT>
[USER BARGE-IN]: What about the second point?
Classification: STOP

Input: There are several important details
</INPUT>
[USER BARGE-IN]: Ah, okay.
Classification: CONTINUE

Input: I think you are wrong.
Classification: STOP

Input: So that's how it works.
</INPUT>
[USER BARGE-IN]: Yes, exactly.
Classification: CONTINUE

Input: It takes 30 days.
</INPUT>
[USER BARGE-IN]: No way!
Classification: CONTINUE

Input: Let's move on to
</INPUT>
[USER BARGE-IN]: I understand.
Classification: CONTINUE

Input: This is the final step.
</INPUT>
[USER BARGE-IN]: Of course!
Classification: CONTINUE

Input: The system will then
</INPUT>
[USER BARGE-IN]: Cool!
Classification: CONTINUE
"""

def patch_runner(prompt):
    with open("benchmark_runners/flexi_runner.py", "r") as f:
        content = f.read()
    
    # Replace the base_v2 string
    pattern = re.compile(r'    base_v2 = """(.*?)"""\n    \n    if is_gemma:', re.DOTALL)
    new_content = pattern.sub(f'    base_v2 = """{prompt}"""\n    \n    if is_gemma:', content)
    
    with open("benchmark_runners/flexi_runner.py", "w") as f:
        f.write(new_content)

print("Patching runner...")
patch_runner(PROMPT)
print("Running benchmark...")
os.system("cd benchmark_runners && rm -rf ../results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi/v3/")
os.system("cd benchmark_runners && rm -f ../gpu.lock gpu.lock")
subprocess.run(["bash", "-c", "cd benchmark_runners && source ../.venv/bin/activate && python3 flexi_runner.py --model_name unsloth/Meta-Llama-3.1-8B-Instruct --version v3 --dataset_path ../mini_flexi.json"])

print("Evaluating...")
os.system("python3 evaluate_v3.py")
