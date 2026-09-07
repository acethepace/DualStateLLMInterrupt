with open("dual_state_interruptor.py", "r") as f:
    content = f.read()

import re
old_code = """        if next_token_id[0].item() == tokenizer.eos_token_id:
            break"""

new_code = """        if next_token_id[0].item() in [tokenizer.eos_token_id, 128009, 128001]:
            break"""

content = content.replace(old_code, new_code)
with open("dual_state_interruptor.py", "w") as f:
    f.write(content)
