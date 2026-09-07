with open("dual_state_interruptor.py", "r") as f:
    content = f.read()

old = """        if stop_word.upper() in assessor_text.upper():
            break"""

new = """        if stop_word.upper() in assessor_text.upper() or "CONTINUE" in assessor_text.upper() or "PROCEED" in assessor_text.upper():
            break"""

if old in content:
    content = content.replace(old, new)
    with open("dual_state_interruptor.py", "w") as f:
        f.write(content)
    print("Patched!")
else:
    print("Could not find old code!")
