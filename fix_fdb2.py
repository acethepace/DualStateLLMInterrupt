import sys

with open("benchmark_runners/fdb_runner.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if "def get_templates(" in line:
        skip = True
        new_lines.append(line)
        new_lines.append("""    is_gemma = "gemma" in model_name.lower()
    is_qwen = "qwen" in model_name.lower()
    base_v1 = "You are an ultra-conservative Semantic VAD. 99% of the time, you should output EXACTLY 'CONTINUE'. The user will make sounds like 'uh huh', 'okay', 'yeah', 'hmm', or ambient noise. ALL OF THESE ARE 'CONTINUE'. ONLY if the user asks a fully formed, explicit new question (e.g., 'What is the weather?'), you output EXACTLY 'STOP'. If in doubt, output 'CONTINUE'."
    base_v2 = open("prompt_v2.txt").read() if is_qwen else "You are an automated Semantic VAD classifier, NOT an assistant. You are forbidden from answering user questions. Your ONLY function is to classify the user's input stream. Output EXACTLY ONE WORD: 'STOP' if the user is making an intentional barge-in (e.g. asking a question, interrupting). Output EXACTLY 'CONTINUE' if the user is uttering a benign backchannel (e.g. 'uh huh', 'okay', 'yeah'). If you output anything other than 'STOP' or 'CONTINUE', the system will crash. Do not generate conversational text."

    if is_gemma:
        user_turn_start = "<bos><|turn>user\\n"
        assessor_trigger = "<turn|>\\n<|turn>model\\n"
        thought_block = "<|channel>thought\\n<channel|>"
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"
    elif is_qwen:
        user_turn_start = "<|im_start|>user\\n"
        assessor_trigger = "<|im_end|>\\n<|im_start|>assistant\\nClassification: "
        thought_block = ""
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"
    else:
        user_turn_start = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n"
        assessor_trigger = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
        thought_block = ""
        stop_v1, cont_v1, stop_v2 = "STOP", "CONTINUE", "STOP"

    config = {}
    if version == "v1":
        config["stop_word"] = stop_v1
        config["max_new_tokens"] = 15
        config["assessor_prompt"] = assessor_trigger
        config["base_instructions"] = base_v1
        config["user_turn_start"] = user_turn_start
    else:
        config["stop_word"] = stop_v2
        config["max_new_tokens"] = 15
        config["assessor_prompt"] = assessor_trigger + (thought_block if version == "v2_prefill" else "")
        config["base_instructions"] = base_v2
        config["user_turn_start"] = user_turn_start
    config["system_turn_start"] = "<|im_start|>system\\n" if is_qwen else "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n"
    return config
""")
        continue
    if skip and "return config" in line:
        skip = False
        continue
    if not skip:
        new_lines.append(line)

with open("benchmark_runners/fdb_runner.py", "w") as f:
    f.write("".join(new_lines))
