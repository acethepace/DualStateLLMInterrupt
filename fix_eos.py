with open("dual_state_interruptor.py", "r") as f:
    content = f.read()

old = """        current_input_ids = next_token_id
        
        if stop_word.upper() in assessor_text.upper() :
            break"""

new = """        current_input_ids = next_token_id
        
        if next_token_id[0].item() in [tokenizer.eos_token_id, 128009, 128001]:
            break
            
        if stop_word.upper() in assessor_text.upper():
            break"""

if old in content:
    content = content.replace(old, new)
    with open("dual_state_interruptor.py", "w") as f:
        f.write(content)
    print("Patched!")
else:
    print("Could not find old code!")
