import json

def flexi_prompt():
    base_instructions = "You are a Semantic Voice Activity Detector..."
    user_turn_start = "<bos><|turn>user\n"
    sys_prompt = "You are an AI."
    base_context = user_turn_start + base_instructions + "\n\n" + sys_prompt + "\n\n<INPUT>\n"
    chunk = "I can help with that. "
    interruption = "Actually I want juice."
    chunk_to_process = chunk + f"\n</INPUT>\n[USER BARGE-IN]: {interruption}"
    assessor = "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    return base_context + chunk_to_process + assessor

def fdb_prompt():
    base_instructions = "You are a Semantic Voice Activity Detector..."
    user_turn_start = "<bos><|turn>user\n"
    sys_prompt = "You are a helpful AI assistant talking to a user. Answer their questions clearly."
    base_stream = ["I can certainly help you with that. ", "Let me look up the available options for you. ", "It looks like we have a few choices available today."]
    base_context = user_turn_start + base_instructions + "\n\n" + sys_prompt + "\n\n<INPUT>\n" + "".join(base_stream)
    interruption = "Could I get some juice with that too, and I'll take wheat bread, please?"
    interruption_to_process = f"\n</INPUT>\n[USER BARGE-IN]: {interruption}"
    assessor = "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    return base_context + interruption_to_process + assessor

print("--- FLEXI ---")
print(flexi_prompt())
print("\n--- FDB ---")
print(fdb_prompt())
