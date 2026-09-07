with open("benchmark_runners/fdb_runner.py", "r") as f:
    text = f.read()

text = text.replace('base_v1 = "You are an ultra-conservative Semantic VAD. 99% of the time, you should output EXACTLY \'CONTINUE\'. The user will make sounds like \'uh huh\', \'okay\', \'yeah\', \'hmm\', or ambient noise. ALL OF THESE ARE \'CONTINUE\'. ONLY if the user asks a fully formed, explicit new question (e.g., \'What is the weather?\'), you output EXACTLY \'STOP\'. If in doubt, output \'CONTINUE\'."', 'base_v1 = open("prompt_v2.txt").read() if is_qwen else "You are an ultra-conservative Semantic VAD. 99% of the time, you should output EXACTLY \'CONTINUE\'. The user will make sounds like \'uh huh\', \'okay\', \'yeah\', \'hmm\', or ambient noise. ALL OF THESE ARE \'CONTINUE\'. ONLY if the user asks a fully formed, explicit new question (e.g., \'What is the weather?\'), you output EXACTLY \'STOP\'. If in doubt, output \'CONTINUE\'."')

with open("benchmark_runners/fdb_runner.py", "w") as f:
    f.write(text)
