with open("dual_state_interruptor.py", "r") as f:
    content = f.read()

old_code = """        if next_token_id[0].item() in [tokenizer.eos_token_id, 128009, 128001]:
            break"""

new_code = """        if next_token_id[0].item() in [tokenizer.eos_token_id, 128009, 128001]:
            break
        
        # Stop early if the model has already emitted a clear STOP or CONTINUE
        current_text = tokenizer.decode(generated_tokens)
        if "STOP" in current_text or "CONTINUE" in current_text:
            break"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open("dual_state_interruptor.py", "w") as f:
        f.write(content)
    print("Patched!")
else:
    print("Could not find old code!")
