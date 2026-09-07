#!/bin/bash
PROMPT="You are a Semantic Voice Activity Detector. You MUST output exactly ONE word: either 'STOP' or 'CONTINUE'. Output 'STOP' if the user is making an intentional barge-in (asking a question, changing topic, adding information). Output 'CONTINUE' if the user is just saying a backchannel (uh huh, yeah, okay) or ambient noise."

cd benchmark_runners
sed -i -E "s/base_v2 = \".*\"/base_v2 = \"\$PROMPT\"/g" fdb_runner.py
rm -rf ../results/unsloth_gemma-4-12b-it/fdb/v2_prefill/
rm -f ../gpu.lock

source ../.venv/bin/activate
# Modify fdb_runner temporarily to only run 20 tasks
sed -i 's/tasks = load_fdb_tasks(dataset_path)/tasks = load_fdb_tasks(dataset_path)[:20]/' fdb_runner.py
python3 fdb_runner.py --model_name unsloth/gemma-4-12b-it --version v2_prefill
# Restore
sed -i 's/tasks = load_fdb_tasks(dataset_path)[:20]/tasks = load_fdb_tasks(dataset_path)/' fdb_runner.py
cd ..
python3 parse_matrices.py | grep -E "FDB \(V2_Prefill\)" -A 1
