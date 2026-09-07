with open("run_all_qwen3.py", "r") as f:
    text = f.read()

text = text.replace('base_v1 = "You are an ultra-conservative Semantic VAD. 99% of the time, you should output EXACTLY \'CONTINUE\'. The user will make sounds like \'uh huh\', \'okay\', \'yeah\', \'hmm\', or ambient noise. ALL OF THESE ARE \'CONTINUE\'. ONLY if the user asks a fully formed, explicit new question (e.g., \'What is the weather?\'), you output EXACTLY \'STOP\'. If in doubt, output \'CONTINUE\'."', 'base_v1 = open("prompt_v2.txt").read()')
text = text.replace('fdb_v1 = evaluate("datasets/fdb_dataset.json", "v1")', '# fdb_v1 = evaluate("datasets/fdb_dataset.json", "v1")')
text = text.replace('fdb_v4 = evaluate("datasets/fdb_dataset.json", "v2_prefill")', '# fdb_v4 = evaluate("datasets/fdb_dataset.json", "v2_prefill")')

with open("run_all_qwen3.py", "w") as f:
    f.write(text)
